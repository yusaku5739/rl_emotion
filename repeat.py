import os 
from q_lambda import learn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml

NUM_REPEAT = 50
exp_name = "mood_constant1_n100"

def plot(l, ax, row, col, title, plot_max=None):
    l = np.array(l) 
    m = np.mean(l, axis=0)
    se = np.std(l, axis=0) / (NUM_REPEAT ** 0.5)
    x = [i for i in range(len(m))]
    ax[row][col].fill_between(x, m + se,  m - se, alpha=0.2, color='gray')
    ax[row][col].set_title(title)
    ax[row][col].plot(x, m, color="black")
    if plot_max is not None:
        ax[row][col].set_ylim(-0.2, plot_max)

os.makedirs(exp_name, exist_ok=True)

anticipatory_licks = []
comsumptory_licks = []
anticipatory_vss = []
comsumptory_vss = []
anticipatory_rpes = []
comsumptory_rpes = []

for i in tqdm(range(NUM_REPEAT)):
    lick, vs, rpe, mood, cfg = learn(verbose=False)

    anticipatory_licks.append(np.sum(lick[:, 50:70], axis=1)/2)
    comsumptory_licks.append(np.sum(lick[:, 70:90], axis=1)/2)

    anticipatory_vss.append(np.mean(vs[:, 50:70], axis=1))
    comsumptory_vss.append(np.mean(vs[:, 70:90], axis=1))

    anticipatory_rpes.append(np.max(rpe[:, 50:70], axis=1))
    comsumptory_rpes.append(np.max(rpe[:, 70:90], axis=1))

with open(f'{exp_name}/cfg.yaml', mode='w', encoding='utf-8')as f:
    yaml.safe_dump(cfg, f)

fig, ax = plt.subplots(3, 2, figsize=(25, 30))
plot(anticipatory_licks, ax, 0, 0, f"anticipatory_lick: n={NUM_REPEAT}")
plot(comsumptory_licks, ax, 0, 1, f"comsumptiory_lick: n={NUM_REPEAT}")
plot(anticipatory_vss, ax, 1, 0, f"anticipatory_vs: n={NUM_REPEAT}")
plot(comsumptory_vss, ax, 1, 1,f"comsumptory_vs: n={NUM_REPEAT}")
plot(anticipatory_rpes, ax, 2, 0, f"anticipatory_rpes: n={NUM_REPEAT}")
plot(comsumptory_rpes, ax, 2, 1,f"comsumptory_rpes: n={NUM_REPEAT}")

plt.savefig(f"{exp_name}/result_repeat.jpg")

np.save(f"{exp_name}/anticipatory_licks", anticipatory_licks)
np.save(f"{exp_name}/comsumptory_licks", comsumptory_licks)
np.save(f"{exp_name}/anticipatory_vss", anticipatory_vss)
np.save(f"{exp_name}/comsumptory_vss", comsumptory_vss)
np.save(f"{exp_name}/anticipatory_rpes", anticipatory_rpes)
np.save(f"{exp_name}/comsumptory_rpes", comsumptory_rpes)