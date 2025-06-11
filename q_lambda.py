import numpy as np
import matplotlib.pyplot as plt
import random

def softmax(a):
    # 式(3.10)の計算
    c = np.max(a) # 最大値
    exp_a = np.exp(a - c) # 分子:オーバーフロー対策
    sum_exp_a = np.sum(exp_a) # 分母
    y = exp_a / sum_exp_a # 式(3.10)
    return y

def get_fuzzy_sound_status(true_sound_status_code):
    """真の音状態コードから、ノイズの乗った知覚状態コードを返す"""
    if true_sound_status_code == 0:
        # 音が鳴っていない時は、曖昧さはない
        return 0

    # 真の経過ステップ数を計算
    true_time_steps = true_sound_status_code - 1
    
    # 知覚の標準偏差をステップ数に変換
    sigma_in_steps = TIME_PERCEPTION_SIGMA_SECONDS / TIME_STEP_DURATION
    
    # 正規分布から「知覚された」経過ステップ数をサンプリング
    perceived_time_steps = (np.random.normal(loc=0, scale=sigma_in_steps)) + true_time_steps
    
    # 整数に丸め、有効範囲内に収める
    perceived_time_steps = int(round(perceived_time_steps))
    perceived_time_steps = max(0, min(perceived_time_steps, MAX_SOUND_STEPS -1))
    
    return 1 + perceived_time_steps

def calculate_state_value(Q_table, state):
    """指定された状態の価値V(s)を計算する"""

    # softmax方策に基づいた期待値を計算
    q_values = Q_table[state[0], state[1], state[2] :]
    policy_probs = softmax(q_values)
    state_value = np.sum(policy_probs * q_values)
    return state_value

def get_cfg():
    ret = {
        "RL_PARAM":{"ALPHA": ALPHA,
                    "GAMMA": GAMMA,
                    "LAMBDA": LAMBDA,
                    "EPSILON_START": EPSILON_START,
                    "EPSILON_END": EPSILON_END,
                    "EPSILON_DECAY_RATE": EPSILON_DECAY_RATE},
        "MOOD_PARAM":{"WITH_MOOD": WITH_MOOD,
                      "MOOD_CONSTANT": MOOD_CONSTANT,
                      "MOOD_N": MOOD_N},
        "REWARD_PARAM":{"REWARD_LICK_COST": REWARD_LICK_COST,
                        "CONDITION_SETTING":CONDITION_SETTING,
                        "NUM_CONDITION":NUM_CONDITION},
        "TASK_PARAM":{"NUM_EPISODES": NUM_EPISODES,
                        "TIME_STEP_DURATION": TIME_STEP_DURATION,
                        "ITI_DURATION_SECONDS": ITI_DURATION_SECONDS,
                        "REWARD_DELAY_OPTIONS_SECONDS": REWARD_DELAY_OPTIONS_SECONDS,
                        "MAX_TRIAL_DURATION_SECONDS": MAX_TRIAL_DURATION_SECONDS,
                        "TIME_PERCEPTION_SIGMA_SECONDS": TIME_PERCEPTION_SIGMA_SECONDS,},
        "STATE_PARAM":{"MAX_SOUND_STEPS": MAX_SOUND_STEPS,
                        "NUM_STATES": NUM_STATES,
                        "ACTION_NOLICK": ACTION_NOLICK,
                        "ACTION_LICK": ACTION_LICK,
                        "NUM_ACTIONS": NUM_ACTIONS,
                        "N_IS_SOUND": N_IS_SOUND,
                        "N_SOUND_STATE": N_SOUND_STATE,
                        "N_IS_DROP": N_IS_DROP,
                        "N_STEP": N_STEP,}
    }
    return ret

def make_coniditions():
    assert NUM_EPISODES%NUM_CONDITION == 0
    conditions = np.concatenate([[i for _ in range(int(NUM_EPISODES/NUM_CONDITION))] for i in range(NUM_CONDITION)])
    random.shuffle(conditions)
    ret = [[] for _ in range(NUM_CONDITION)]
    for i, c in enumerate(conditions):
        ret[c].append(i)

    return conditions, ret
        
# --- パラメータ設定 ---
SEED=42
# 強化学習パラメータ
ALPHA = 0.05
GAMMA = 0.99
LAMBDA = 0.996
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY_RATE = 0.999 #0.999 2つ

WITH_MOOD = False
MOOD_CONSTANT = 0.0
MOOD_N = 100

# 報酬設定 <<< 変更点
REWARD_LICK_COST = -0.5 # 報酬がない時に舐めた時のコスト 0.05

# シミュレーションパラメータ
NUM_EPISODES = 3000
TIME_STEP_DURATION = 0.1

# 環境パラメータ
ITI_DURATION_SECONDS = [5, 5]
REWARD_DELAY_OPTIONS_SECONDS = [2.0]
MAX_TRIAL_DURATION_SECONDS = max(ITI_DURATION_SECONDS) + max(REWARD_DELAY_OPTIONS_SECONDS) + 3.0
TIME_PERCEPTION_SIGMA_SECONDS = 1.0

# 状態空間の定義
MAX_SOUND_STEPS = int(3.0 / TIME_STEP_DURATION)
NUM_STATES = 1 + (MAX_SOUND_STEPS + 1)
CONDITION_SETTING = {
    "reward":[0,1,10],
    "omission":[1,1,1],
    "condition_name":["reward:0", "reward:1", "reward:10"]
}
NUM_CONDITION = len(CONDITION_SETTING["reward"])

# 行動空間の定義
ACTION_NOLICK = 0
ACTION_LICK = 1
NUM_ACTIONS = 2

# --- 状態エンコーディング ---
N_IS_SOUND = 2
N_SOUND_STATE = int((1 + 4) / TIME_STEP_DURATION) + 1
N_IS_DROP = 2
N_STEP = int(MAX_TRIAL_DURATION_SECONDS / 0.1) + 1

random.seed(SEED)
np.random.seed(SEED)


def learn(verbose=True):
    # --- Qテーブルと適格度トレースの初期化 ---
    Q_table = np.zeros((N_SOUND_STATE, NUM_CONDITION+1, N_IS_DROP, NUM_ACTIONS))
    E_traces = np.zeros((N_SOUND_STATE, NUM_CONDITION+1, N_IS_DROP, NUM_ACTIONS))
    # --- 学習記録用 ---
    rewards_per_episode = []
    anticipatory_licks_per_episode = []
    consummatory_licks_per_episode = []

    conditions_list, conditions_index = make_coniditions()
    epsilon_values = []
    episode_licks = []
    episode_rpes = []
    episode_moods = []
    vs_list = []
    mood = MOOD_CONSTANT
    # --- シミュレーションメインループ ---
    epsilon = EPSILON_START

    for episode in range(NUM_EPISODES):
        condition = conditions_list[episode]
        E_traces.fill(0.0)

        current_time_seconds = 0.0
        is_sound_on = False
        time_since_sound_onset_steps = 0
        
        current_reward_delay_seconds = random.choice(REWARD_DELAY_OPTIONS_SECONDS)
        #current_reward_delay_seconds = max(1, np.random.normal(REWARD_DELAY_OPTIONS_SECONDS[0], 0.5))
        sound_start_seconds = random.randint(ITI_DURATION_SECONDS[0], ITI_DURATION_SECONDS[1])
        reward_delivery_time_seconds = sound_start_seconds + current_reward_delay_seconds
        
        episode_anticipatory_licks = 0
        episode_consummatory_licks = 0
        episode_total_reward = 0

        episode_lick = []
        episode_rpe = []
        episode_mood = []
        vs = []
        
        sucrose_done = False
        current_step = 0
        current_state = [0, 0, 0]
        n_lick=0
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY_RATE)
        time_from_sucrose = 0
        while current_time_seconds < MAX_TRIAL_DURATION_SECONDS:
            is_reward_window = current_time_seconds > reward_delivery_time_seconds and not sucrose_done
            v_current = calculate_state_value(Q_table, current_state)
            # 報酬提示時以外は、方策に基づいて行動を選択
            if np.random.rand() < epsilon:
                # εの確率でランダムに行動を選択
                #action = np.random.randint(0, NUM_ACTIONS)
                if is_reward_window:
                    action = np.random.choice(
                        [ACTION_NOLICK, ACTION_LICK], 
                        p=[0.1, 0.9] 
                    )
                else:
                    action = np.random.choice(
                        [ACTION_NOLICK, ACTION_LICK], 
                        p=[0.9, 0.1] 
                    )
            else:
                # 1-εの確率でQ値が最大の行動を選択
                action = np.argmax(Q_table[current_state[0],  current_state[1], current_state[2], :])
            # 報酬の計算
            if action == ACTION_LICK:
                # 報酬がない時に舐めたらコストを与える 
                if is_reward_window:
                    omission_rate = CONDITION_SETTING["omission"][condition]
                    if np.random.random() < omission_rate:
                        reward_qty = CONDITION_SETTING["reward"][condition]
                        reward = reward_qty
                    else:
                        reward = 0
                    sucrose_done = True
                else:
                    if not is_sound_on:
                        reward = REWARD_LICK_COST
                    else:
                        reward = 0
                if current_time_seconds >= sound_start_seconds and current_time_seconds > sound_start_seconds + 1:
                    episode_anticipatory_licks += 1
                elif is_reward_window:
                    episode_consummatory_licks += 1
                n_lick += 1
            else: # NoLick
                reward = 0

            episode_total_reward += reward
            episode_lick.append(action)
            # --- 1タイムステップ進める ---
            current_time_seconds += TIME_STEP_DURATION
            if not is_sound_on and current_time_seconds >= sound_start_seconds:
                is_sound_on = True
                E_traces.fill(0.0)
            
            if current_time_seconds >= sound_start_seconds:
                next_condition = condition + 1
            else:
                next_condition = 0

            # --- TD学習の更新プロセス (変更なし) ---
            next_sound_state = 0 if not is_sound_on else get_fuzzy_sound_status(1 + int((current_time_seconds - sound_start_seconds) / TIME_STEP_DURATION))
            next_state = [next_sound_state, next_condition, int(is_reward_window),]

            v_next = calculate_state_value(Q_table, next_state)
            rpe_v = reward + GAMMA * v_next - v_current
            episode_rpe.append(rpe_v)

            if current_time_seconds >= MAX_TRIAL_DURATION_SECONDS:
                td_target = reward
            else:
                td_target = reward + GAMMA * np.max(Q_table[next_state[0], next_state[1], current_state[2]  :])
            
            td_error = td_target - Q_table[current_state[0], current_state[1], current_state[2], action]
            mood = mood + 2/(MOOD_N+1) * (ALPHA * td_error - mood) + MOOD_CONSTANT

            E_traces[current_state[0], current_state[1], current_state[2], action] = 1.0 
            Q_table += ALPHA * td_error * E_traces + WITH_MOOD * (1-ALPHA) * mood   
            E_traces *= GAMMA * LAMBDA

            #episode_rpe.append(td_error)
            current_state = next_state
            episode_mood.append(mood)
            vs.append(v_next)
            current_step += 1

        lick_hz = [np.mean(episode_lick[i-1:i+1])/0.3 for i in range(1, len(episode_lick)-2)]
        lick_hz.insert(0, np.mean(episode_lick[0:1]))
        lick_hz.append(np.mean(episode_lick[-2:-1]))
        #episode_licks.append(lick_hz)
        episode_licks.append(episode_lick)
        episode_rpes.append(episode_rpe)
        episode_moods.append(episode_mood)
        vs_list.append(vs)
        rewards_per_episode.append(episode_total_reward)
        anticipatory_licks_per_episode.append(episode_anticipatory_licks)
        consummatory_licks_per_episode.append(episode_consummatory_licks) # 参考として記録
        epsilon_values.append(epsilon)

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY_RATE)

        if (episode + 1) % 100 == 0 and verbose:
            print(f"Episode {episode + 1}/{NUM_EPISODES} - Avg Reward: {np.mean(rewards_per_episode[-100:]) :.2f} "
                f"- Avg Anticip. Licks: {np.mean(anticipatory_licks_per_episode[-100:]) :.2f} "
                f"- Epsilon: {epsilon:.3f}")
    #print(Q_table[1:, 1, 0,:])
    # print(Q_table[1:, 1, 0, 1])
    #print(Q_table[1:, 2, 0, :])
    #  print(Q_table[1:, 2, 0, 1])
    return np.array(episode_licks), np.array(vs_list), np.array(episode_rpes), np.array(episode_moods), np.array(conditions_index), get_cfg()

# --- 結果のプロット (変更なし) ---
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def single():
    episode_licks, vs_list, episode_rpes, episode_moods, conditions, cfg = learn()
    window = 50
    print(len(conditions), len(conditions[0]))

    mag = 100

    fig, ax = plt.subplots(2, 5, figsize=(40, 10))
    episode_licks = np.array(episode_licks)
    episode_licks = np.array([np.mean(episode_licks[:, i-2:i+2], axis=1) for i in range(2,episode_licks.shape[1]-2)]).T
    for c in range(NUM_CONDITION):
        episode_licks_ = episode_licks[conditions[c]]
        for i, n in enumerate([ii*mag for ii in range(10)]):
            m = np.mean(episode_licks_[n:50+n, :]/0.5, axis=0)
            se = np.std(episode_licks_[n:50+n, :]/0.5, axis=0) / (50**0.5)
            #lim = max(np.max(m+se), lim)
            x = [i for i in range(len(m))]
            ax[i//5][i%5].axvline(50, color="blue")
            ax[i//5][i%5].axvline(70, color="orange")
            ax[i//5][i%5].fill_between(x, m + se,  m - se, alpha=0.2, color='gray')
            ax[i//5][i%5].set_title(f"licking rate: {n}-{n+50} episode")
            ax[i//5][i%5].plot(x, m, label=CONDITION_SETTING["condition_name"][c])
            ax[i//5][i%5].set_ylim(0, 4)
    plt.legend()
    plt.savefig("result_q_lambda_lick_trace.jpg")

    fig, ax = plt.subplots(2, 5, figsize=(40, 10))
    vs_list = np.array(vs_list)

    
    for c in range(NUM_CONDITION):
        vs_list_ = vs_list[conditions[c]]
        for i, n in enumerate([ii*mag for ii in range(10)]):
            m = np.mean(vs_list_[n:50+n, :], axis=0)
            se = np.std(vs_list_[n:50+n, :], axis=0) / (50**0.5)
            #lim = max(np.max(m+se), lim)
            x = [i for i in range(len(m))]
            ax[i//5][i%5].axvline(50, color="blue")
            ax[i//5][i%5].axvline(70, color="orange")
            ax[i//5][i%5].fill_between(x, m + se,  m - se, alpha=0.2, color='gray')
            ax[i//5][i%5].set_title(f"state value: {n}-{n+50} episode")
            ax[i//5][i%5].plot(x, m, label=CONDITION_SETTING["condition_name"][c])
    
    plt.legend()
    plt.savefig("result_q_lambda_vs.jpg")

    fig, ax = plt.subplots(2, 5, figsize=(40, 10))
    episode_rpes = np.array(episode_rpes)

    for c in range(NUM_CONDITION):
        episode_rpes_ = episode_rpes[conditions[c]]
        for i, n in enumerate([ii*mag for ii in range(10)]):
            m = np.mean(episode_rpes_[n:2+n, :], axis=0)
            se = np.std(episode_rpes_[n:2+n, :], axis=0) / (50**0.5)
            #lim = max(np.max(m+se), lim)
            x = [i for i in range(len(m))]
            ax[i//5][i%5].axvline(50, color="blue")
            ax[i//5][i%5].axvline(70, color="orange")
            ax[i//5][i%5].fill_between(x, m + se,  m - se, alpha=0.2, color='gray')
            ax[i//5][i%5].set_title(f"rpe: {n}-{n+50} episode")
            ax[i//5][i%5].plot(x, m, label=CONDITION_SETTING["condition_name"][c])

    plt.legend()
    plt.savefig("result_q_lambda_rpe.jpg")

    fig, ax = plt.subplots(2, 5, figsize=(40,10))
    episode_moods = np.array(episode_moods)
    for c in range(NUM_CONDITION):
        episode_moods_ = episode_moods[conditions[c]]
        for i, n in enumerate([ii*mag for ii in range(10)]):
            m = np.mean(episode_moods_[n:50+n, :], axis=0)
            se = np.std(episode_moods_[n:50+n, :], axis=0) / (50**0.5)
            #lim = max(np.max(m+se), lim)
            x = [i for i in range(len(m))]
            ax[i//5][i%5].axvline(50, color="blue")
            ax[i//5][i%5].axvline(70, color="orange")
            ax[i//5][i%5].fill_between(x, m + se,  m - se, alpha=0.2, color='gray')
            ax[i//5][i%5].set_title(f"rpe: {n}-{n+50} episode")
            ax[i//5][i%5].plot(x, m, label=CONDITION_SETTING["condition_name"][c])

    plt.legend()
    plt.savefig("result_q_lambda_mood.jpg")

if __name__=="__main__":

    single()