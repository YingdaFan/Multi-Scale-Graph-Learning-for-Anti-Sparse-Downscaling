
import subprocess

#basins = ['Brandywine', 'LowerDelaware', 'Neversink', 'LowerWestBranchDelaware', 'Rancocas']
#maskpercentages = [0]
#random_seeds_default = [42, 61, 71]
#model_seeds = [1, 2, 3]

# basins = ['Brandywine', 'Rancocas']
# maskpercentages = [round(i * 0.03, 2) for i in range(34)]
# random_seeds_default = [42]
# model_seeds = [1]

#mask 曲线 #2025.2.17
basins = ['LowerDelaware', 'Neversink', 'LowerWestBranchDelaware', 'Rancocas']
random_seeds_default = [42]
maskpercentages = [0,0.8,0.9,0.91,0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995]
model_seeds = [1]
model_names = ['RGCN_v1']

for basin in basins:
    for maskpercentage in maskpercentages:
        if maskpercentage == 0:
            random_seeds = [42]
        else:
            random_seeds = random_seeds_default

        for random_seed in random_seeds:
            for model_seed in model_seeds:
                command = [
                    'C://Users//admin//.conda//envs//river//python.exe', 'multi.py',  # 注意修改这里！！
                    '--basin', str(basin),
                    '--maskpercentage', str(maskpercentage),
                    '--random_seed', str(random_seed),
                    '--model_seed', str(model_seed)
                ]
                subprocess.run(command)
