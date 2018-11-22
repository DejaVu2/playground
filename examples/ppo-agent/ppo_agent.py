from gym.core import Wrapper
from glob import glob
import csv
import uuid
import pandas
import os.path as osp
import json
import os
import time
import functools
import gym
from gym import spaces
import numpy as np
import tensorflow as tf
from baselines import logger
from multiprocessing import Process, Pipe
from collections import deque
from abc import ABC, abstractmethod
from pommerman import make
from pommerman.agents import BaseAgent
from pommerman import characters
from pommerman import agents
from baselines.common import explained_variance, set_global_seeds
from baselines.common.policies import build_policy
from baselines.common.tf_util import get_session, save_variables, load_variables
from baselines.common.vec_env import VecEnv, VecEnvWrapper, CloudpickleWrapper
from baselines.common.vec_env.util import copy_obs_dict, dict_to_obs, obs_space_info

try:
    from baselines.common.mpi_adam_optimizer import MpiAdamOptimizer
    from mpi4py import MPI
    from baselines.common.mpi_util import sync_from_root
except ImportError:
    MPI = None

from baselines.common.tf_util import initialize


class Monitor(Wrapper):
    EXT = "monitor.csv"
    f = None

    def __init__(self, env, filename, allow_early_resets=False, reset_keywords=(), info_keywords=()):
        Wrapper.__init__(self, env=env)
        self.tstart = time.time()
        self.results_writer = ResultsWriter(
            filename,
            header={"t_start": time.time(), 'env_id': env.spec and env.spec.id},
            extra_keys=reset_keywords + info_keywords
        )
        self.reset_keywords = reset_keywords
        self.info_keywords = info_keywords
        self.allow_early_resets = allow_early_resets
        self.rewards = None
        self.needs_reset = True
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        self.total_steps = 0
        self.current_reset_info = {}  # extra info about the current episode, that was passed in during reset()
        self.id = env.id

    def reset(self, **kwargs):
        self.reset_state()
        for k in self.reset_keywords:
            v = kwargs.get(k)
            if v is None:
                raise ValueError('Expected you to pass kwarg %s into reset'%k)
            self.current_reset_info[k] = v
        return self.env.reset(**kwargs)

    def reset_state(self):
        if not self.allow_early_resets and not self.needs_reset:
            raise RuntimeError("Tried to reset an environment before done. If you want to allow early resets, wrap your env with Monitor(env, path, allow_early_resets=True)")
        self.rewards = []
        self.needs_reset = False

    def step(self, action):
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        ob, rew, done, info, alive_list = self.env.step(action)
        self.update(ob, rew, done, info)
        return (ob, rew, done, info, alive_list)

    def update(self, ob, rew, done, info):
        self.rewards.append(rew)
        if done:
            # self.needs_reset = True
            a, b, c, d = 0, 0, 0, 0
            for rl in self.rewards:
                a += rl[0]
                b += rl[1]
                c += rl[2]
                d += rl[3]
            eprew = [a, b, c, d]
            eplen = len(self.rewards)
            epinfo = {"r": [round(epr, 6) for epr in eprew], "l": eplen, "t": round(time.time() - self.tstart, 6)}
            for k in self.info_keywords:
                epinfo[k] = info[k]
            self.episode_rewards.append(eprew)
            self.episode_lengths.append(eplen)
            self.episode_times.append(time.time() - self.tstart)
            epinfo.update(self.current_reset_info)
            self.results_writer.write_row(epinfo)

            if isinstance(info, dict):
                info['episode'] = epinfo

        self.total_steps += 1

    def close(self):
        if self.f is not None:
            self.f.close()

    def get_total_steps(self):
        return self.total_steps

    def get_episode_rewards(self):
        return self.episode_rewards

    def get_episode_lengths(self):
        return self.episode_lengths

    def get_episode_times(self):
        return self.episode_times


class LoadMonitorResultsError(Exception):
    pass


class ResultsWriter(object):
    def __init__(self, filename=None, header='', extra_keys=()):
        self.extra_keys = extra_keys
        if filename is None:
            self.f = None
            self.logger = None
        else:
            if not filename.endswith(Monitor.EXT):
                if osp.isdir(filename):
                    filename = osp.join(filename, Monitor.EXT)
                else:
                    filename = filename + "." + Monitor.EXT
            self.f = open(filename, "wt")
            if isinstance(header, dict):
                header = '# {} \n'.format(json.dumps(header))
            self.f.write(header)
            self.logger = csv.DictWriter(self.f, fieldnames=('r', 'l', 't')+tuple(extra_keys))
            self.logger.writeheader()
            self.f.flush()

    def write_row(self, epinfo):
        if self.logger:
            self.logger.writerow(epinfo)
            self.f.flush()


def get_monitor_files(dir):
    return glob(osp.join(dir, "*" + Monitor.EXT))


def load_results(dir):
    import pandas
    monitor_files = (
        glob(osp.join(dir, "*monitor.json")) +
        glob(osp.join(dir, "*monitor.csv")))  # get both csv and (old) json files
    if not monitor_files:
        raise LoadMonitorResultsError("no monitor files of the form *%s found in %s" % (Monitor.EXT, dir))
    dfs = []
    headers = []
    for fname in monitor_files:
        with open(fname, 'rt') as fh:
            if fname.endswith('csv'):
                firstline = fh.readline()
                if not firstline:
                    continue
                assert firstline[0] == '#'
                header = json.loads(firstline[1:])
                df = pandas.read_csv(fh, index_col=None)
                headers.append(header)
            elif fname.endswith('json'):  # Deprecated json format
                episodes = []
                lines = fh.readlines()
                header = json.loads(lines[0])
                headers.append(header)
                for line in lines[1:]:
                    episode = json.loads(line)
                    episodes.append(episode)
                df = pandas.DataFrame(episodes)
            else:
                assert 0, 'unreachable'
            df['t'] += header['t_start']
        dfs.append(df)
    df = pandas.concat(dfs)
    df.sort_values('t', inplace=True)
    df.reset_index(inplace=True)
    df['t'] -= min(header['t_start'] for header in headers)
    df.headers = headers  # HACK to preserve backwards compatibility
    return df


def test_monitor():
    env = gym.make("CartPole-v1")
    env.seed(0)
    mon_file = "/tmp/baselines-test-%s.monitor.csv" % uuid.uuid4()
    menv = Monitor(env, mon_file)
    menv.reset()
    for _ in range(1000):
        _, _, done, _ = menv.step(0)
        if done:
            menv.reset()

    f = open(mon_file, 'rt')

    firstline = f.readline()
    assert firstline.startswith('#')
    metadata = json.loads(firstline[1:])
    assert metadata['env_id'] == "CartPole-v1"
    assert set(metadata.keys()) == {'env_id', 'gym_version', 't_start'},  "Incorrect keys in monitor metadata"

    last_logline = pandas.read_csv(f, index_col=None)
    assert set(last_logline.keys()) == {'l', 't', 'r'}, "Incorrect keys in monitor logline"
    f.close()
    os.remove(mon_file)


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                ob, reward, done, info, alive_list = env.step(data)
                if (env.id + 10) not in alive_list or done:
                    ob = env.reset()
                    done = True
                remote.send((ob, reward, done, info))
            elif cmd == 'reset':
                ob = env.reset()
                remote.send(ob)
            elif cmd == 'render':
                remote.send(env.render(mode='rgb_array'))
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        env.close()


class SubprocVecEnv(VecEnv):
    """
    VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
    Recommended to use when num_envs > 1 and step() can be a bottleneck.
    """
    def __init__(self, env_fns, spaces=None):
        """
        Arguments:

        env_fns: iterable of callables -  functions that create environments to run in subprocesses. Need to be cloud-pickleable
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        self.viewer = None
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        self._assert_not_closed()
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        self._assert_not_closed()
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close_extras(self):
        self.closed = True
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def get_images(self):
        self._assert_not_closed()
        for pipe in self.remotes:
            pipe.send(('render', None))
        imgs = [pipe.recv() for pipe in self.remotes]
        return imgs

    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"


class VecFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack):
        self.venv = venv
        self.nstack = nstack
        wos = venv.observation_space  # wrapped ob space
        low = np.repeat(wos.low, self.nstack, axis=-1)
        high = np.repeat(wos.high, self.nstack, axis=-1)
        self.stackedobs = np.zeros((venv.num_envs, 4,) + low.shape, low.dtype)
        observation_space = spaces.Box(low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        for i in range(4):
            self.stackedobs[i] = np.roll(self.stackedobs[i], shift=-1, axis=-1)
            if news[i]:
                self.stackedobs[i] = 0
            self.stackedobs[i][..., -obs.shape[-1]:] = obs[i]
        return self.stackedobs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        for i in range(4):
            self.stackedobs[i][...] = 0
            self.stackedobs[i][..., -obs.shape[-1]:] = obs[i]
        return self.stackedobs


def make_vec_env(agent_list, num_env, seed=None, wrapper_kwargs=None, start_index=0, gamestate=None):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari and MuJoCo.
    """
    if wrapper_kwargs is None:
        wrapper_kwargs = {}
    mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
    seed = seed + 10000 * mpi_rank if seed is not None else None

    def make_thunk(rank):
        return lambda: make_env(
            agent_list=agent_list,
            subrank=rank,
            seed=seed,
            gamestate=gamestate,
            wrapper_kwargs=wrapper_kwargs
        )

    set_global_seeds(seed)
    env = SubprocVecEnv([make_thunk(i + start_index) for i in range(num_env)])
    return VecFrameStack(env, 4)


def featurize(obs):
    board = obs['board']

    # convert board items into bitmaps
    maps = [board == i for i in range(10)]
    maps.append(obs['bomb_blast_strength'])
    maps.append(obs['bomb_life'])

    # duplicate ammo, blast_strength and can_kick over entire map
    maps.append(np.full(board.shape, obs['ammo']))
    maps.append(np.full(board.shape, obs['blast_strength']))
    maps.append(np.full(board.shape, obs['can_kick']))

    # add my position as bitmap
    position = np.zeros(board.shape)
    position[obs['position']] = 1
    maps.append(position)

    # add teammate
    if obs['teammate'] is not None:
        maps.append(board == obs['teammate'].value)
    else:
        maps.append(np.zeros(board.shape))

    # add enemies
    enemies = [board == e.value for e in obs['enemies']]
    maps.append(np.any(enemies, axis=0))

    return np.stack(maps, axis=2)


class WrappedEnv(gym.Wrapper):
    '''An Env Wrapper used to make it easier to work
    with multiple agents'''
    def __init__(self, env, subrank, visualize=False):
        super(WrappedEnv, self).__init__(env)
        self.visualize = visualize
        self.can_kick_dict = {10: 0, 11: 0, 12: 0, 13: 0}
        self.blast_strength_dict = {10: 2, 11: 2, 12: 2, 13: 2}
        self.ammo_record = {10: [], 11: [], 12: [], 13: []}
        self.ammo_num_list = [1, 1, 1, 1]
        self.alive_list = [10, 11, 12, 13]
        self.num_steps = 0
        self.observation_space = spaces.Box(
            low=0, high=25,
            shape=(11, 11, 18), dtype=env.observation_space.dtype.name)
        self.id = subrank

    def step(self, actions):
        if self.visualize:
            self.env.render()

        self.num_steps += 1

        obs, reward, done, info = self.env.step(actions)
        neo_obs = []
        neo_reward = [0, 0, 0, 0]
        for i, ob in enumerate(obs):
            if (i + 10) not in ob['alive']:
                neo_obs.append(np.zeros(self.observation_space.shape))
                if (i + 10) not in self.alive_list:
                    pass
                else:
                    # neo_reward[i] += ((self.num_steps - 1) / 800 - 1) * 10
                    neo_reward[i] += -5
                    self.alive_list.remove(i + 10)
            else:
                neo_obs.append(featurize(ob))
                neo_reward[i] += 0.1

                if ob['can_kick'] and not self.can_kick_dict[i + 10]:
                    neo_reward[i] += 1
                    self.can_kick_dict[i + 10] = 1

                if ob['blast_strength'] > self.blast_strength_dict[i + 10]:
                    neo_reward[i] += 1
                    self.blast_strength_dict[i + 10] = ob['blast_strength']

                if ob['ammo'] < self.ammo_num_list[i]:
                    self.ammo_record[i + 10].append(self.num_steps - 1)
                    neo_reward[i] += 1
                elif ob['ammo'] > self.ammo_num_list[i]:
                    if (self.num_steps - 10) in self.ammo_record[i + 10]:
                        self.ammo_record[i + 10].remove(self.num_steps - 10)
                    else:
                        neo_reward[i] += 1
                self.ammo_num_list[i] = ob['ammo']

                if done:
                    if reward[i] == 0:
                        neo_reward[i] += -5

                    if reward[i] == 1 and (i + 10) in ob['alive']:
                        neo_reward[i] += (1 - (self.num_steps - 1) / 800) * 10

        # obs = [self.env.featurize(ob) for ob in obs]
        # return obs, reward, done, info
        return neo_obs, neo_reward, done, info, self.alive_list

    def reset(self):
        self.can_kick_dict = {10: 0, 11: 0, 12: 0, 13: 0}
        self.blast_strength_dict = {10: 2, 11: 2, 12: 2, 13: 2}
        self.ammo_record = {10: [], 11: [], 12: [], 13: []}
        self.ammo_num_list = [1, 1, 1, 1]
        self.alive_list = [10, 11, 12, 13]
        self.num_steps = 0

        obs = self.env.reset()
        obs = [featurize(ob) for ob in obs]
        return obs


def make_env(agent_list, subrank=0, seed=None, gamestate=None, wrapper_kwargs={}):
    mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
    env = make('PommeTeamCompetitionFast-v0', agent_list)
    env = WrappedEnv(env, subrank)

    # env.seed(seed + subrank if seed is not None else None)
    env = Monitor(env,
                  logger.get_dir() and os.path.join(logger.get_dir(), str(mpi_rank) + '.' + str(subrank)),
                  allow_early_resets=True)

    return env


class PPOAgent(BaseAgent):
    """The PPOAgent. Acts through the algorith, not here."""

    def __init__(self, character=characters.Bomber):
        super(PPOAgent, self).__init__(character)

    def act(self, obs, action_space):
        """This agent has its own way of inducing actions"""
        self.stackedobs = np.roll(self.stackedobs, shift=-1, axis=-1)
        obs = featurize(obs)
        self.stackedobs[..., -obs.shape[-1]:] = obs
        action, _, _, _ = model.step(self.stackedobs)
        return int(action)

    def initialize(self, model):
        self.model = model
        self.stackedobs = np.zeros((1, 11, 11, 72))


class Model(object):
    """
    We use this object to :
    __init__:
    - Creates the step_model
    - Creates the train_model

    train():
    - Make the training part (feedforward and retropropagation of gradients)

    save/load():
    - Save load the model
    """
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                 nsteps, ent_coef, vf_coef, max_grad_norm):
        sess = get_session()

        with tf.variable_scope('ppo2_model', reuse=tf.AUTO_REUSE):
            # CREATE OUR TWO MODELS
            # act_model that is used for sampling
            act_model = policy(nbatch_act, 1, sess)

            # Train model for training
            train_model = policy(nbatch_train, nsteps, sess)

        # CREATE THE PLACEHOLDERS
        A = train_model.pdtype.sample_placeholder([None])
        ADV = tf.placeholder(tf.float32, [None])
        R = tf.placeholder(tf.float32, [None])
        # Keep track of old actor
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        # Keep track of old critic
        OLDVPRED = tf.placeholder(tf.float32, [None])
        LR = tf.placeholder(tf.float32, [])
        # Cliprange
        CLIPRANGE = tf.placeholder(tf.float32, [])

        neglogpac = train_model.pd.neglogp(A)

        # Calculate the entropy
        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        entropy = tf.reduce_mean(train_model.pd.entropy())

        # CALCULATE THE LOSS
        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss

        # Clip the value to reduce variability during Critic training
        # Get the predicted value
        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        # Unclipped value
        vf_losses1 = tf.square(vpred - R)
        # Clipped value
        vf_losses2 = tf.square(vpredclipped - R)

        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

        # Calculate ratio (pi current policy / pi old policy)
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)

        # Defining Loss = - J is equivalent to max J
        pg_losses = -ADV * ratio

        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)

        # Final PG loss
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))

        # Total loss
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        # UPDATE THE PARAMETERS USING LOSS
        # 1. Get the model parameters
        params = tf.trainable_variables('ppo2_model')
        # 2. Build our trainer
        if MPI is not None:
            trainer = MpiAdamOptimizer(MPI.COMM_WORLD, learning_rate=LR, epsilon=1e-5)
        else:
            trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        # 3. Calculate the gradients
        grads_and_var = trainer.compute_gradients(loss, params)
        grads, var = zip(*grads_and_var)

        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads_and_var = list(zip(grads, var))
        # zip aggregate each gradient with parameters associated
        # For instance zip(ABCD, xyza) => Ax, By, Cz, Da

        _train = trainer.apply_gradients(grads_and_var)
        print(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

        def train(lr, cliprange, obs, returns, masks, actions, values, neglogpacs, states=None):
            # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
            # Returns = R + yV(s')
            advs = returns - values

            # Normalize the advantages
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            td_map = {train_model.X: obs, A: actions, ADV: advs, R: returns, LR: lr,
                      CLIPRANGE: cliprange, OLDNEGLOGPAC: neglogpacs, OLDVPRED: values}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            return sess.run(
                [pg_loss, vf_loss, entropy, approxkl, clipfrac, _train],
                td_map
            )[:-1]
        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

        self.train = train
        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state

        self.save = functools.partial(save_variables, sess=sess)
        self.load = functools.partial(load_variables, sess=sess)

        if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
            initialize()
        global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="")

        if MPI is not None:
            sync_from_root(sess, global_variables)  # pylint: disable=E1101


class AbstractEnvRunner(ABC):
    def __init__(self, *, env, model, nsteps):
        self.env = env
        self.model = model
        self.nenv = nenv = env.num_envs if hasattr(env, 'num_envs') else 1
        self.batch_ob_shape = (nenv*nsteps, 4,) + env.observation_space.shape
        self.obs = np.zeros(  # pylint: disable=E0632
            (nenv, 4,) + env.observation_space.shape,
            dtype=env.observation_space.dtype.name
        )
        self.obs[:] = env.reset()  # pylint: disable=E0632
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

    @abstractmethod
    def run(self):
        raise NotImplementedError


class Runner(AbstractEnvRunner):
    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner

    run():
    - Make a mini batch
    """
    def __init__(self, *, env, model, nsteps, gamma, lam):
        super().__init__(env=env, model=model, nsteps=nsteps)
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma

    def run(self):
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        mb_states = self.states
        epinfos = []
        # For n in range number of steps
        for _ in range(self.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            tmp_obs = np.zeros((4, 11, 11, 72))
            tmp_actions = []
            tmp_values = []
            tmp_neglogpacs = []
            tmp_dones = []
            all_actions = []
            for i in range(4):
                actions, values, self.states, neglogpacs = self.model.step(self.obs[i], S=self.states, M=self.dones[i])
                tmp_obs[i] = self.obs[i][i].copy()
                tmp_actions.append(actions[i])
                tmp_values.append(values[i])
                tmp_neglogpacs.append(neglogpacs[i])
                tmp_dones.append(self.dones[i])
                all_actions.append(actions)
            mb_obs.append(tmp_obs)
            mb_actions.append(tmp_actions)
            mb_values.append(tmp_values)
            mb_neglogpacs.append(tmp_neglogpacs)
            mb_dones.append(tmp_dones)

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            self.obs[:], rewards, self.dones, infos = self.env.step(all_actions)
            epinfos.append([infos])
            mb_rewards.append([reward[i] for i, reward in enumerate(rewards)])
        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = []
        for i in range(4):
            values = self.model.value(self.obs[i], S=self.states, M=self.dones[i])
            last_values.append(values[i])
        last_values = np.asarray(last_values, dtype=np.float32)

        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
                mb_states, epinfos)
# obs, returns, masks, actions, values, neglogpacs, states = runner.run()


def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


def constfn(val):
    def f(_):
        return val
    return f


def learn(*, network, env, total_timesteps, eval_env=None, seed=None, nsteps=2048, ent_coef=0.0, lr=3e-4, vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95, log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2, save_interval=0, load_path=None, **network_kwargs):
    '''
    Learn policy using PPO algorithm (https://arxiv.org/abs/1707.06347)

    Parameters:
    ----------

    network:                          policy network architecture. Either string (mlp, lstm, lnlstm, cnn_lstm, cnn, cnn_small, conv_only - see baselines.common/models.py for full list)
                                      specifying the standard network architecture, or a function that takes tensorflow tensor as input and returns
                                      tuple (output_tensor, extra_feed) where output tensor is the last network layer output, extra_feed is None for feed-forward
                                      neural nets, and extra_feed is a dictionary describing how to feed state into the network for recurrent neural nets.
                                      See common/models.py/lstm for more details on using recurrent nets in policies

    env: baselines.common.vec_env.VecEnv     environment. Needs to be vectorized for parallel environment simulation.
                                      The environments produced by gym.make can be wrapped using baselines.common.vec_env.DummyVecEnv class.


    nsteps: int                       number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv where
                                      nenv is number of environment copies simulated in parallel)

    total_timesteps: int              number of timesteps (i.e. number of actions taken in the environment)

    ent_coef: float                   policy entropy coefficient in the optimization objective

    lr: float or function             learning rate, constant or a schedule function [0,1] -> R+ where 1 is beginning of the
                                      training and 0 is the end of the training.

    vf_coef: float                    value function loss coefficient in the optimization objective

    max_grad_norm: float or None      gradient norm clipping coefficient

    gamma: float                      discounting factor

    lam: float                        advantage estimation discounting factor (lambda in the paper)

    log_interval: int                 number of timesteps between logging events

    nminibatches: int                 number of training minibatches per update. For recurrent policies,
                                      should be smaller or equal than number of environments run in parallel.

    noptepochs: int                   number of training epochs per update

    cliprange: float or function      clipping range, constant or schedule function [0,1] -> R+ where 1 is beginning of the training
                                      and 0 is the end of the training

    save_interval: int                number of timesteps between saving events

    load_path: str                    path to load the model from

    **network_kwargs:                 keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network
                                      For instance, 'mlp' network architecture has arguments num_hidden and num_layers.



    '''

    set_global_seeds(seed)

    if isinstance(lr, float):
        lr = constfn(lr)
    else:
        assert callable(lr)
    if isinstance(cliprange, float):
        cliprange = constfn(cliprange)
    else:
        assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    policy = build_policy(env, network, **network_kwargs)

    # Get the nb of env
    nenvs = env.num_envs

    # Get state_space and action_space
    ob_space = env.observation_space
    ac_space = env.action_space

    # Calculate the batch_size
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches

    # Instantiate the model object (that creates act_model and train_model)
    make_model = lambda : Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=1, nbatch_train=nbatch_train, nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm)
    model = make_model()
    if load_path is not None:
        model.load(load_path)
    # Instantiate the runner object
    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)
    if eval_env is not None:
        eval_runner = Runner(env=eval_env, model=model, nsteps=nsteps, gamma=gamma, lam= lam)

    epinfobuf = deque(maxlen=100)
    if eval_env is not None:
        eval_epinfobuf = deque(maxlen=100)

    # Start total timer
    tfirststart = time.time()

    nupdates = total_timesteps//nbatch
    for update in range(1, nupdates+1):
        assert nbatch % nminibatches == 0
        # Start timer
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        # Calculate the learning rate
        lrnow = lr(frac)
        # Calculate the cliprange
        cliprangenow = cliprange(frac)
        # Get minibatch
        obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run()  # pylint: disable=E0632
        if eval_env is not None:
            eval_obs, eval_returns, eval_masks, eval_actions, eval_values, eval_neglogpacs, eval_states, eval_epinfos = eval_runner.run()  # pylint: disable=E0632

        epinfobuf.extend(epinfos)
        if eval_env is not None:
            eval_epinfobuf.extend(eval_epinfos)

        # Here what we're going to do is for each minibatch calculate the loss and append it.
        mblossvals = []
        if states is None:  # nonrecurrent version
            # Index of each element of batch_size
            # Create the indices array
            inds = np.arange(nbatch)
            for _ in range(noptepochs):
                # Randomize the indexes
                np.random.shuffle(inds)
                # 0 to batch_size with batch_train_size step
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices))
        else:  # recurrent version
            assert nenvs % nminibatches == 0
            envsperbatch = nenvs // nminibatches
            envinds = np.arange(nenvs)
            flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
            envsperbatch = nbatch_train // nsteps
            for _ in range(noptepochs):
                np.random.shuffle(envinds)
                for start in range(0, nenvs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mbflatinds = flatinds[mbenvinds].ravel()
                    slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mbstates = states[mbenvinds]
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))

        # Feedforward --> get losses --> update
        lossvals = np.mean(mblossvals, axis=0)
        # End timer
        tnow = time.time()
        # Calculate the fps (frame per second)
        fps = int(nbatch / (tnow - tstart))
        if update % log_interval == 0 or update == 1:
            # Calculates if value function is a good predicator of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            ev = explained_variance(values, returns)
            logger.logkv("serial_timesteps", update*nsteps)
            logger.logkv("nupdates", update)
            logger.logkv("total_timesteps", update*nbatch)
            logger.logkv("fps", fps)
            logger.logkv("explained_variance", float(ev))
            # logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
            # logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            if eval_env is not None:
                logger.logkv('eval_eprewmean', safemean([epinfo['r'] for epinfo in eval_epinfobuf]) )
                logger.logkv('eval_eplenmean', safemean([epinfo['l'] for epinfo in eval_epinfobuf]) )
            logger.logkv('time_elapsed', tnow - tfirststart)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv(lossname, lossval)
            if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
                logger.dumpkvs()
        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir() and (MPI is None or MPI.COMM_WORLD.Get_rank() == 0):
            checkdir = osp.join(logger.get_dir(), 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            savepath = osp.join(checkdir, '%.5i'%update)
            print('Saving to', savepath)
            model.save(savepath)
    return model
# Avoid division error when calculate the mean (in our case if epinfo is empty returns np.nan, not return an error)
def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)


def play(model):
    # Print all possible environments in the Pommerman registry
    # print(pommerman.registry)
    # Create a set of agents (exactly four)
    agent1 = PPOAgent()
    agent1.initialize(model)
    agent2 = PPOAgent()
    agent2.initialize(model)
    agent3 = PPOAgent()
    agent3.initialize(model)
    agent4 = PPOAgent()
    agent4.initialize(model)
    agent_list = [
        agent1,
        agent2,
        # agents.SimpleAgent(),
        agent3,
        agent4,
        # agents.SimpleAgent(),
    ]
    # Make the "Free-For-All" environment using the agent list
    env = make('PommeTeamCompetition-v0', agent_list)

    # Run the episodes just like OpenAI Gym
    for i_episode in range(1):
        state = env.reset()
        done = False
        while not done:
            actions = env.act(state)
            env.render()
            state, reward, done, info = env.step(actions)
        print('Episode {} finished'.format(i_episode))
        print('  Result {}'.format(info))
        # Result {'result': <Result.Win: 0>, 'winners': [0, 2]}
        # Result {'result': <Result.Tie: 2>}
    env.close()


"""
if __name__ == '__main__':
    logger.configure('pom_log', ['stdout', 'tensorboard'])
    agent_list = [
        PPOAgent(),
        PPOAgent(),
        PPOAgent(),
        PPOAgent(),
    ]
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config):
        model = learn(network='cnn',
                      env=make_vec_env(agent_list, 4),
                      nsteps=8192,
                      nminibatches=1024,
                      log_interval=1,
                      ent_coef=0.01,
                      lr=lambda _: 2e-4,
                      cliprange=lambda _: 0.1,
                      # total_timesteps=int(1e7),
                      total_timesteps=int(0),
                      save_interval=10,
                      load_path='00025',
                      )
        play(model)
"""
