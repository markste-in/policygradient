import gym
import tensorflow as tf
import numpy as np
from timeit import default_timer as Timer
import os

FORCE_CPU_USAGE = True

if FORCE_CPU_USAGE:

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if tf.test.gpu_device_name():
        print('GPU found')
    else:
        print("No GPU found")


print(f"Tensorflow Version: {tf.__version__}")
print(f"Gym Version: {gym.__version__}")
print(f"Numpy Version: {np.__version__}")

tboard_drive = 'RL'
ENV_NAME = 'LunarLander-v2'
COMMENT = 'Nadam_April'

env = gym.make(ENV_NAME)
env.reset()

ACTIVATION = 'tanh'
KERNEL_INIT = 'he_normal'
NUM_ACTIONS = env.action_space.n
NUM_HIDDEN_UNITS = 40
OBS_SPACE = env.observation_space.shape[0]
FILE_WEIGHTS = os.path.join(tboard_drive , ENV_NAME + COMMENT+ KERNEL_INIT + '_weights' + ACTIVATION + str(NUM_HIDDEN_UNITS) +'.h5')
TENSORBOARD_LOG = os.path.join(tboard_drive , ENV_NAME + COMMENT + KERNEL_INIT + str(NUM_HIDDEN_UNITS) + ACTIVATION + '_units_'+str(Timer()))


RENDER = True
RESUME = True
class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__(self)
        self.num_actions = NUM_ACTIONS
        self.hidden0 = tf.keras.layers.Dense(units=NUM_HIDDEN_UNITS, activation='relu',kernel_initializer=KERNEL_INIT)
        self.hidden1 = tf.keras.layers.Dense(units=NUM_HIDDEN_UNITS, activation='relu',kernel_initializer=KERNEL_INIT)
        self.logits = tf.keras.layers.Dense(self.num_actions,kernel_initializer=KERNEL_INIT, activation = ACTIVATION)

    def call(self, inputs):
        x = tf.convert_to_tensor(inputs)
        hidden0 = self.hidden0(x)
        hidden1 = self.hidden1(hidden0)
        out = self.logits(hidden1)
        return out



class Agent:
    def __init__(self, model):
        self.model = model
        self.memory = self.Memory()
        self.model.build((1,OBS_SPACE))
        self.optimizer = tf.optimizers.Nadam(0.0004)

    def _loss(self,rewards, neg_loglike): #(y_true, y_pred)
        loss = tf.reduce_mean(rewards*neg_loglike)
        return loss

    def neg_loglike(self, action: object, logits: object) -> object:
        return tf.squeeze(tf.nn.softmax_cross_entropy_with_logits(labels=action, logits=logits))

    def action(self, obs):
        logits = self.model(obs.reshape(1,-1))
        action = self.PickActionFromLogits(logits=logits)
        action = tf.keras.utils.to_categorical(action,num_classes=NUM_ACTIONS)
        return action, logits

    def PickActionFromLogits(self, logits):
        return tf.random.categorical(logits,1)

    def flushBuffer(self,gradBuffer):
        for ix, grad in enumerate(gradBuffer):
            gradBuffer[ix] = grad * 0

    def train(self, env, resume = True):
        if resume:
            try: self.model.load_weights(FILE_WEIGHTS)
            except Exception as e: print(e)
        writer = tf.summary.create_file_writer(TENSORBOARD_LOG)
        obs = env.reset()
        gradBuffer = self.model.trainable_weights
        self.flushBuffer(gradBuffer)
        current_run = 0
        update_every = 5
        running_reward = None
        max_pos = []
        while True:
            if RENDER:
                env.render()
            with tf.GradientTape() as tape:
                a, logits = self.action(obs)
                n_obs, reward, done, info = env.step(np.argmax(a))
                neg_loglike = self.neg_loglike(a,logits)
            grads = tape.gradient(neg_loglike,self.model.trainable_weights)

            self.memory.save(reward,grads)
            
            
            obs = n_obs
            if done:
                print(logits,a)
                max_pos = []
                
                running_reward = self.memory.sum_rewards if running_reward == None else running_reward*0.99+self.memory.sum_rewards*0.01
                for grads, r in zip(self.memory.gradients,self.memory.discounted_rewards):
                    for ix, grad in enumerate(grads):
                        gradBuffer[ix] += grad * r

                with writer.as_default():
                    tf.summary.scalar('score', self.memory.sum_rewards, current_run)
                    tf.summary.scalar('running_reward', running_reward,current_run)
                    for w in self.model.trainable_variables:
                        tf.summary.histogram(name=w.name, data=w, step=current_run)

                if current_run % update_every == 0:
                    self.optimizer.apply_gradients(zip(gradBuffer, model.trainable_variables))
                    self.flushBuffer(gradBuffer)
                    self.model.save_weights(FILE_WEIGHTS)

                print(f'run {current_run} done with score {self.memory.sum_rewards} and running mean {running_reward}')
                current_run += 1
                self.memory.clear()
                obs = env.reset()





    class Memory():
        def __init__(self):
            self.__rewards=[]
            self.__gradients=[]

        def save(self,reward, gradient):
            gradient = [item.numpy() for item in gradient]
            self.__rewards.append(reward)
            self.__gradients.append(gradient)

        def clear(self):
            self.__rewards=[]
            self.__gradients=[]

        @property
        def sum_rewards(self):
            return np.sum(self.rewards)

        @property
        def rewards(self):
            return np.array(self.__rewards)

        @property
        def gradients(self):
            return np.array(self.__gradients)

        @property
        def discounted_rewards(self, gamma=0.99):
            discounted_r = np.zeros_like(self.rewards)
            running_add = 0
            for t in reversed(range(0, len(self.rewards))):
                running_add = running_add * gamma + self.rewards[t]
                discounted_r[t] = running_add
            return np.array(discounted_r)

model = Model()
agent = Agent(model)
model.summary()
agent.train(env,resume=RESUME)