import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import os
from tqdm import tqdm, trange
import json
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
import moviepy.editor as mpy

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6
'''
Global utilities
'''

TORCH_DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torchify = lambda x: torch.FloatTensor(x).to(TORCH_DEVICE)


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


# Soft update of target critic network
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) +
                                param.data * tau)


# Hard update of target critic network
def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

# Make gifs
def npy_to_gif(im_list, filename, fps=4):
    clip = mpy.ImageSequenceClip(im_list, fps=fps)
    clip.write_gif(filename + '.gif')


# Load trajectories
def load_trajectories(num_traj, file):
    print('Loading trajectories from %s' % file)

    if not os.path.exists(file):
        raise RuntimeError("Could not find directory %s." % file)
    trajectories = []
    iterator = range(num_traj) if num_traj <= 200 else trange(num_traj)
    for i in iterator:
        if not os.path.exists(os.path.join(file, '%d.json' % i)):
            print('Could not find %d' % i)
            continue
        im_fields = ('obs', 'next_obs')
        with open(os.path.join(file, '%d.json' % i), 'r') as f:
            trajectory = json.load(f)
        im_dat = {}

        for field in im_fields:
            f = os.path.join(file, "%d_%s.npy" % (i, field))
            if os.path.exists(file):
                dat = np.load(f)
                im_dat[field] = dat.astype(np.uint8)

        for j, frame in list(enumerate(trajectory)):
            for key in im_dat:
                frame[key] = im_dat[key][j]
        trajectories.append(trajectory)

    return trajectories


'''
Architectures for latent dynamics model for model-based recovery policy
'''


# f_dyn, model of dynamics in latent space
class TransitionModel(nn.Module):
    __constants__ = ['min_std_dev']

    def __init__(self, action_size, hidden_size=256, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(hidden_size + action_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, hidden_size)

    def forward(self, prev_hidden, action):
        hidden = torch.cat([prev_hidden, action], dim=-1)
        trajlen, batchsize = hidden.size(0), hidden.size(1)
        hidden.view(-1, hidden.size(2))
        hidden = self.act_fn(self.fc1(hidden))
        hidden = self.act_fn(self.fc2(hidden))
        hidden = self.act_fn(self.fc3(hidden))
        hidden = self.fc4(hidden)
        hidden = hidden.view(trajlen, batchsize, -1)
        return hidden


# Encoder
class VisualEncoderAttn(nn.Module):
    __constants__ = ['embedding_size']

    def __init__(self, hidden_size=256,
                 activation_function='relu',
                 ch=3):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()
        self.ch = ch
        self.conv1 = nn.Conv2d(self.ch, 32, 4, stride=2)  #3
        self.conv1_1 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv2_1 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv3_1 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
        self.conv4_1 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 2 * hidden_size)

    def forward(self, observation):
        trajlen, batchsize = observation.size(0), observation.size(1)
        observation = observation.reshape(trajlen * batchsize, 3, 64, 64)
        atn = torch.zeros_like(observation[:, :1])

        hidden = self.act_fn(self.conv1(observation))
        hidden = self.act_fn(self.conv1_1(hidden))
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv2_1(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        hidden = self.act_fn(self.conv3_1(hidden))
        hidden = self.act_fn(self.conv4(hidden))
        hidden = self.act_fn(self.conv4_1(hidden))

        hidden = hidden.view(trajlen * batchsize, -1)
        hidden = self.act_fn(self.fc1(hidden))
        hidden = self.fc2(hidden)
        hidden = hidden.view(trajlen, batchsize, -1)
        atn = atn.view(trajlen, batchsize, 1, 64, 64)
        return hidden, atn


# Decoder
class VisualReconModel(nn.Module):
    __constants__ = ['embedding_size']

    def __init__(self,
                 hidden_size=256,
                 activation_function='relu',
                 action_len=5):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(hidden_size * 1, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.ConvTranspose2d(128, 128, 5, stride=2)
        self.conv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.conv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.conv4 = nn.ConvTranspose2d(32, 3, 6, stride=2)

    def forward(self, hidden):
        trajlen, batchsize = hidden.size(0), hidden.size(1)
        hidden = hidden.view(trajlen * batchsize, -1)
        hidden = self.act_fn(self.fc1(hidden))
        hidden = self.act_fn(self.fc2(hidden))
        hidden = self.fc3(hidden)
        hidden = hidden.view(-1, 128, 1, 1)

        hidden = self.act_fn(self.conv1(hidden))
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        residual = self.sigmoid(self.conv4(hidden)) * 255.0

        residual = residual.view(trajlen, batchsize, residual.size(1),
                                 residual.size(2), residual.size(3))
        return residual


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

def train(obs_seqs,
          ac_seqs,
          encoder, 
          transition_model,
          residual_model,
          num_train_steps=20000,
          checkpoint_interval=100,
          hidden_size=256,
          batch_size=256,
          logdir='outputs',
          curric_int=6,
          beta=10):
    '''
    Offline visual dynamics training.
    Arguments:
        obs_seqs, ac_seqs, constraint_seqs: offline episodes of observations,
        actions, constraints used for visual dynamics model training
    '''
    metrics = {'trainsteps': [], 'observation_loss': [], 'teststeps': []}
    print("Number of Train Steps: ", num_train_steps)
    for s in range(num_train_steps):
        # Sample batch_size indices
        batch_idxs = np.random.randint(len(obs_seqs),
                                       size=batch_size).astype(int)
        obs_batch = torch.FloatTensor(obs_seqs[batch_idxs].transpose(
            1, 0, 2, 3, 4)).to(TORCH_DEVICE)
        action_batch = torch.FloatTensor(ac_seqs[batch_idxs].transpose(
            1, 0, 2)).to(TORCH_DEVICE)
        # Get state encoding
        encoding, atn = encoder(obs_batch)
        mu, log_std = encoding[:, :, :hidden_size], encoding[:, :, hidden_size:]
        std = torch.exp(log_std)
        samples = torch.empty(mu.shape).normal_(mean=0, std=1).cuda()
        encoding = mu + std * samples
        klloss = 0.5 * torch.mean(mu**2 + std**2 - torch.log(std**2) - 1)
        lossinc = min(curric_int - 1,
                      int(s / (num_train_steps / curric_int)))

        if s < num_train_steps:
            residuals = obs_batch
            all_losses = []
            # Pick random start frame for logging:
            sp_log = np.random.randint(obs_batch.size(0) - lossinc)
            for sp in range(obs_batch.size(0) - lossinc):
                next_step = []
                next_step_encoding = encoding[sp:sp + 1]
                next_step.append(next_step_encoding)
                for p in range(lossinc):
                    this_act = action_batch[sp + p:sp + p + 1]
                    next_step_encoding = transition_model(
                        next_step_encoding, this_act)
                    next_step.append(next_step_encoding)
                next_step = torch.cat(next_step)
                next_res = residual_model(next_step)
                if sp == sp_log:
                    log_residual_pred = next_res
                ## Reconstruction Error
                prederr = ((residuals[sp:sp + 1 + lossinc] -
                            next_res[:1 + lossinc])**2)
                all_losses.append(prederr.mean())
            r_loss = torch.stack(all_losses).mean(0)

        # Update all networks
        dynamics_optimizer.zero_grad()
        (r_loss + beta * klloss).backward()
        dynamics_optimizer.step()
        metrics['observation_loss'].append(r_loss.cpu().detach().numpy())
        metrics['trainsteps'].append(s)

        # Checkpoint models
        if s % checkpoint_interval == 0:
            print("Checkpoint: ", s)
            print("Loss Inc: ", lossinc)
            print("Observation Loss: ", r_loss.cpu().detach().numpy())
            print("KL Loss: ", klloss.cpu().detach().numpy())
            model_name = 'model_{}.pth'.format(s)

            torch.save(
                {
                    'transition_model': transition_model.state_dict(),
                    'residual_model': residual_model.state_dict(),
                    'encoder': encoder.state_dict(),
                    'dynamics_optimizer':
                    dynamics_optimizer.state_dict(),
                }, os.path.join(logdir, model_name))
            newpath = os.path.join(logdir, str(s))
            os.makedirs(newpath, exist_ok=True)
            metrics['teststeps'].append(s)
            # Save model predicttion gif
            video_frames = []
            for p in range(lossinc + 1):
                video_frames.append(
                    make_grid(torch.cat([
                        residuals[p + sp_log, :5, :, :, :].cpu().detach(),
                        log_residual_pred[p, :5, :, :, :].cpu().detach(),
                    ],
                                        dim=3),
                              nrow=1).numpy().transpose(1, 2, 0))

            npy_to_gif(video_frames,
                       os.path.join(newpath, 'train_steps_{}'.format(s)))

if __name__ == "__main__":
    traj_len = 10
    batch_size = 20
    beta = 10

    encoder = VisualEncoderAttn().to(device=TORCH_DEVICE)
    transition_model = TransitionModel(4).to(device=TORCH_DEVICE)
    residual_model = VisualReconModel().to(device=TORCH_DEVICE)

    dynamics_param_list = list(
        transition_model.parameters()) + list(
            residual_model.parameters()) + list(
                encoder.parameters())
    dynamics_optimizer = optim.Adam(dynamics_param_list,
                                    lr=3e-4,
                                    eps=1e-4)

    # Load in obs_seqs and ac_seqs
    trajs = load_trajectories(1000, 'data/Gym-Cloth')
    new_trajs = []
    obs_seqs = []
    ac_seqs = []
    for traj in trajs:
        if len(traj) == traj_len:
            obs_seqs.append([])
            ac_seqs.append([])
            for i in range(len(traj)):
                transition = traj[i]
                obs_seqs[-1].append(transition["obs"])
                if i == traj_len-1:
                    obs_seqs[-1].append(transition["next_obs"])
                ac_seqs[-1].append(transition["action"])

    obs_seqs = np.array(obs_seqs)
    obs_seqs = np.transpose(obs_seqs, (0, 1, 4, 2, 3))
    ac_seqs = np.array(ac_seqs)
    print("Total Trajectories: %d" % len(obs_seqs))
    train(obs_seqs, ac_seqs, encoder, transition_model, residual_model, batch_size=batch_size, beta=beta)
