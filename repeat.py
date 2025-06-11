import os 
from q_lambda import learn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml

NUM_REPEAT = 50
exp_name = "reward_0_1_10"

def plot(l, conditions, num_condition, ax, row, col, title, plot_max=None):
    l = np.array(l, dtype=float)
    for c in range(num_condition):
        l_ = np.zeros((l.shape[0], l.shape[1]//num_condition), dtype=float)
        for i in range(len(l)):
            l_[i] = l[i, conditions[i, c]]
        m = np.mean(l_, axis=0)
        se = np.std(l_, axis=0) / (NUM_REPEAT ** 0.5)
        x = [i for i in range(len(m))]
        ax[row][col].fill_between(x, m + se,  m - se, alpha=0.2, color='gray')
        ax[row][col].set_title(title)
        ax[row][col].plot(x, m)
        if plot_max is not None:
            ax[row][col].set_ylim(-0.2, plot_max)

os.makedirs(exp_name, exist_ok=True)

anticipatory_licks = []
comsumptory_licks = []
anticipatory_vss = []
comsumptory_vss = []
anticipatory_rpes = []
comsumptory_rpes = []
conditions = []

for i in tqdm(range(NUM_REPEAT)):
    lick, vs, rpe, mood, condition, cfg = learn(verbose=False)

    anticipatory_licks.append(np.sum(lick[:, 50:70], axis=1)/2)
    comsumptory_licks.append(np.sum(lick[:, 70:90], axis=1)/2)

    anticipatory_vss.append(np.mean(vs[:, 50:70], axis=1))
    comsumptory_vss.append(np.mean(vs[:, 70:90], axis=1))

    anticipatory_rpes.append(np.max(rpe[:, 50:70], axis=1))
    comsumptory_rpes.append(np.max(rpe[:, 70:90], axis=1))

    conditions.append(condition)

with open(f'{exp_name}/cfg.yaml', mode='w', encoding='utf-8')as f:
    yaml.safe_dump(cfg, f)

conditions = np.array(conditions)
num_condition = cfg["REWARD_PARAM"]["NUM_CONDITION"]

fig, ax = plt.subplots(3, 2, figsize=(25, 30))
plot(np.array(anticipatory_licks), conditions, num_condition, ax, 0, 0, f"anticipatory_lick: n={NUM_REPEAT}")
plot(np.array(comsumptory_licks), conditions, num_condition, ax, 0, 1, f"comsumptiory_lick: n={NUM_REPEAT}")
plot(np.array(anticipatory_vss), conditions, num_condition, ax, 1, 0, f"anticipatory_vs: n={NUM_REPEAT}")
plot(np.array(comsumptory_vss), conditions, num_condition, ax, 1, 1,f"comsumptory_vs: n={NUM_REPEAT}")
plot(np.array(anticipatory_rpes), conditions, num_condition, ax, 2, 0, f"anticipatory_rpes: n={NUM_REPEAT}")
plot(np.array(comsumptory_rpes), conditions, num_condition, ax, 2, 1,f"comsumptory_rpes: n={NUM_REPEAT}")

plt.savefig(f"{exp_name}/result_repeat.jpg")

np.save(f"{exp_name}/anticipatory_licks", anticipatory_licks)
np.save(f"{exp_name}/comsumptory_licks", comsumptory_licks)
np.save(f"{exp_name}/anticipatory_vss", anticipatory_vss)
np.save(f"{exp_name}/comsumptory_vss", comsumptory_vss)
np.save(f"{exp_name}/anticipatory_rpes", anticipatory_rpes)
np.save(f"{exp_name}/comsumptory_rpes", comsumptory_rpes)
np.save(f"{exp_name}/conditions", conditions)