import os
import sys
import yaml
current_dir = os.path.dirname(os.path.abspath(__file__))
config_file_path = os.path.join(current_dir, '../config_multiscale.yml')
with open(config_file_path, 'r') as f:
    config = yaml.safe_load(f)
sys.path.insert(0, os.path.abspath(os.path.join(current_dir, '..')))
sys.path.insert(0, os.path.abspath(os.path.join(current_dir, '../..')))

import numpy as np
import torch
import torch.optim as optim
code_dir = config['code_dir']

import sys

sys.path.insert(1, code_dir)
from multiscale.multi.model import RGCN_Multi
from multiscale.multi.predict import mypredict_from_io_data
from multiscale.multi.torch_utils import my_train_torch
from multiscale.multi.torch_utils import pretrain

from river_dl.mypreproc_utils import prep_all_data
from river_dl.my_torch_utilsMulti import rmse_masked
from river_dl.evaluate import combined_metrics
from river_dl.mypreproc_utils import reduce_training_data_random
from river_dl.create_mixed_distance_matrix import mixed_dixtance_matrix
outdir = config['out_dir']
from pandas import pandas as pd

import argparse
parser = argparse.ArgumentParser(description='Running multi-parameter experiments')
parser.add_argument('--basin', type=str, help='basin', default='Rancocas')
parser.add_argument('--maskpercentage', type=float, help='maskpercentage', default=0)
parser.add_argument('--random_seed', type=int, help='random_seed', default=42)
parser.add_argument('--model_seed', type=int, help='model_seed', default=1)
args = parser.parse_args()


basin = args.basin
maskpercentage = args.maskpercentage
random_seed = args.random_seed
model_seed = args.model_seed
if maskpercentage > 0.99:       
    maskpercentage_formatted = int(maskpercentage * 1000)
else:
    maskpercentage_formatted = int(maskpercentage * 100)   



def getSegs(x):
    #x = 'COMID' or 'seg_id_nat'
    crossDF = pd.read_csv("../../DRB_NHD_on_NHM_20240119/NHD_on_NHM_crosswalk.csv")
    crossDF = crossDF.loc[crossDF.Basin==basin]
    return np.unique(crossDF[x].values)

if maskpercentage > 0:
    y_train_sparse = reduce_training_data_random(
        config['obs_file_NHD'],
        train_start_date=config['train_start_date'],
        train_end_date=config['train_end_date'],
        #val_start_date=config['val_start_date'],
        #val_end_date=config['val_end_date'],
        reduce_amount=maskpercentage,
        out_file=f"../{outdir}/spare_obs_temp_flow",
        #segs=None
        segs=getSegs('COMID'),
        random_seed=random_seed,
    )


prepped = prep_all_data(
          x_data_file=config['sntemp_file'],
          pretrain_file=config['sntemp_file'],
          y_data_file=config['obs_file'],
          distfile=config['dist_matrix_file'],
          x_vars=config['x_vars'],
          y_vars_pretrain=config['y_vars_pretrain'],
          y_vars_finetune=config['y_vars_finetune'],
          spatial_idx_name="seg_id_nat",
          catch_prop_file=None,
          train_start_date=config['train_start_date'],
          train_end_date=config['train_end_date'],
          val_start_date=config['val_start_date'],
          val_end_date=config['val_end_date'],
          test_start_date=config['test_start_date'],
          test_end_date=config['test_end_date'],
          #segs=None,
          segs = getSegs('seg_id_nat'),
          out_file=f"../{outdir}/prepped.npz",
          trn_offset = config['trn_offset'],
          tst_val_offset = config['tst_val_offset'],
          check_pre_partitions=False,
          fill_batch = False)



prepped_NHD =     prep_all_data(
                  x_data_file=config['sntemp_file_NHD'],
                  pretrain_file=config['sntemp_file_NHD'],
                  y_data_file=f"../{outdir}/spare_obs_temp_flow" if maskpercentage > 0 else config['obs_file_NHD'],
                  distfile=config['dist_matrix_file_NHD'],
                  x_vars=config['x_vars_NHD'],
                  y_vars_pretrain=config['y_vars_pretrain_NHD'],
                  y_vars_finetune=config['y_vars_finetune'],
                  spatial_idx_name='COMID',
                  catch_prop_file=None,
                  train_start_date=config['train_start_date'],
                  train_end_date=config['train_end_date'],
                  val_start_date=config['val_start_date'],
                  val_end_date=config['val_end_date'],
                  test_start_date=config['test_start_date'],
                  test_end_date=config['test_end_date'],
                  #segs=None,
                  segs=getSegs('COMID'),
                  out_file=f"../{outdir}/prepped_NHD.npz",
                  trn_offset=config['trn_offset'],
                  tst_val_offset=config['tst_val_offset'],
                  check_pre_partitions=False,
                  fill_batch=False)


'''
#Pre-Train
data = np.load(f"../{outdir}/dataset/prepped_CRW_pretrain.npz") #这里与微调不同
data_NHD = np.load(f"../{outdir}/dataset/prepped_CRW_NHD_pretrain.npz") #这里与微调不同
x_num_segs = len(np.unique(data['ids_trn']))
x_num_segs_NHD = len(np.unique(data_NHD['ids_trn']))
y_num_segs = len(np.unique(data['ids_trn']))
y_num_segs_NHD = len(np.unique(data_NHD['ids_trn']))
adj_mx = data['dist_matrix']
adj_mx_NHD = data_NHD['dist_matrix']
adj_mx_mixed = mixed_dixtance_matrix(data,data_NHD,config['dist_matrix_file_NHD'], config['crosswalk'])
in_dim = len(data['x_vars'])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = RGCN_Multi(in_dim, config['hidden_size'], adj_matrix=adj_mx, adj_matrix_NHD=adj_mx_NHD, adj_matrix_mixed=adj_mx_mixed,device=device, seed=config['seed'])
opt = optim.Adam(model.parameters(),lr=config['pretrain_learning_rate'])

pre_train =        pretrain(model,
                    loss_function = rmse_masked,
                    optimizer= opt,
                    x_train= data['x_trn'],
                    y_train = data_NHD['y_obs_trn'],
                    max_epochs = config['pt_epochs'],
                    x_batch_size = x_num_segs,
                    y_batch_size = y_num_segs_NHD,
                    weights_file = f"../{outdir}/pretrained_weights.pth",
                    log_file = f"../{outdir}/pretrain_log.csv",
                    device=device)
'''
'''
#new pretrain with multi-objective learning
data = np.load(f"../{outdir}/dataset/prepped_CRW_pretrain.npz") #这里与微调不同
data_NHD = np.load(f"../{outdir}/dataset/prepped_CRW_NHD_pretrain.npz") #这里与微调不同
x_num_segs = len(np.unique(data['ids_trn']))
x_num_segs_NHD = len(np.unique(data_NHD['ids_trn']))
y_num_segs = len(np.unique(data['ids_trn']))
y_num_segs_NHD = len(np.unique(data_NHD['ids_trn']))
adj_mx = data['dist_matrix']
adj_mx_NHD = data_NHD['dist_matrix']
adj_mx_mixed = mixed_dixtance_matrix(data,data_NHD,config['dist_matrix_file_NHD'], config['crosswalk'])
in_dim = len(data['x_vars'])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = RGCN_Multi(in_dim, config['hidden_size'], adj_matrix=adj_mx, adj_matrix_NHD=adj_mx_NHD, adj_matrix_mixed=adj_mx_mixed,device=device, seed=config['seed'])
opt = optim.Adam(model.parameters(),lr=config['pretrain_learning_rate'])
pre_train =      my_train_torch(model,
                loss_function=rmse_masked,
                optimizer=opt,
                x_train=data['x_trn'],
                x_train_NHD=data_NHD['x_trn'],
                y_train=data['y_obs_trn'],
                y_train_NHD=data_NHD['y_obs_trn'],
                x_val=data['x_val'],
                x_val_NHD= data_NHD['x_val'],
                y_val=data['y_obs_val'],
                y_val_NHD=data_NHD['y_obs_val'],
                x_tst=data['x_tst'],
                x_tst_NHD=data_NHD['x_tst'],
                y_tst=data['y_obs_tst'],
                y_tst_NHD=data_NHD['y_obs_tst'],
                max_epochs=config['pt_epochs'],
                x_batch_size = x_num_segs,
                x_batch_size_NHD = x_num_segs_NHD,
                y_batch_size = y_num_segs,
                y_batch_size_NHD = y_num_segs_NHD,
                weights_file=f"../{outdir}/pretrained_weights.pth",
                log_file = f"../{outdir}/pretrain_log.csv",
                device=device,
                y_std=data_NHD["y_std"],
                y_mean=data_NHD["y_mean"])
'''




#Fine-Tune
data = np.load(f"../{outdir}/prepped.npz")
data_NHD = np.load(f"../{outdir}/prepped_NHD.npz")
x_num_segs = len(np.unique(data['ids_trn']))
x_num_segs_NHD = len(np.unique(data_NHD['ids_trn']))
y_num_segs = len(np.unique(data['ids_trn']))
y_num_segs_NHD = len(np.unique(data_NHD['ids_trn']))
adj_mx = data['dist_matrix']
adj_mx_NHD = data_NHD['dist_matrix']
adj_mx_mixed = mixed_dixtance_matrix(data,data_NHD, config['dist_matrix_file_NHD'], config['crosswalk'])
in_dim = len(data['x_vars'])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = RGCN_Multi(in_dim, config['hidden_size'], adj_matrix=adj_mx, adj_matrix_NHD=adj_mx_NHD, adj_matrix_mixed=adj_mx_mixed,device=device, seed=model_seed)
opt = optim.Adam(model.parameters(),lr=config['finetune_learning_rate'])
model.load_state_dict(torch.load(f"../{outdir}/pretrained_weights.pth", map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')))
if 'pre_train' in locals():
    model.load_state_dict(torch.load(f"../{outdir}/pretrained_weights.pth", map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')))



train =      my_train_torch(model,
                loss_function=rmse_masked,
                optimizer=opt,
                x_train=data['x_trn'],
                x_train_NHD=data_NHD['x_trn'],
                y_train=data['y_obs_trn'],
                y_train_NHD=data_NHD['y_obs_trn'],
                x_val=data['x_val'],
                x_val_NHD= data_NHD['x_val'],
                y_val=data['y_obs_val'],
                y_val_NHD=data_NHD['y_obs_val'],
                x_tst=data['x_tst'],
                x_tst_NHD=data_NHD['x_tst'],
                y_tst=data['y_obs_tst'],
                y_tst_NHD=data_NHD['y_obs_tst'],
                max_epochs=config['ft_epochs'],
                early_stopping_patience=config['early_stopping'],
                x_batch_size = x_num_segs,
                x_batch_size_NHD = x_num_segs_NHD,
                y_batch_size = y_num_segs,
                y_batch_size_NHD = y_num_segs_NHD,
                weights_file=f"../{outdir}/finetuned_weights_Multi.pth",
                log_file=f"../{outdir}/multi_log.csv",
                device=device)






model = RGCN_Multi(in_dim, config['hidden_size'], adj_matrix=adj_mx, adj_matrix_NHD=adj_mx_NHD, adj_matrix_mixed=adj_mx_mixed,device=device, seed=config['seed'])
model.load_state_dict(torch.load(f"../{outdir}/finetuned_weights_Multi.pth"))
#partitions = ['trn','val','tst']
partitions = ['tst']



for partition in partitions:
    mypredict_from_io_data(model=model,
                         io_data_x=f"../{outdir}/prepped.npz",
                         io_data_x2=f"../{outdir}/prepped_NHD.npz",
                         io_data_y=f"../{outdir}/prepped_NHD.npz",
                         partition=partition,
                         outfile=f"../{outdir}/{partition}_preds_Multi.feather",
                         log_vars=False,
                         trn_offset=config['trn_offset'],
                         tst_val_offset=config['tst_val_offset'],
                         spatial_idx_name="COMID")



def get_grp_arg(metric_type):
    if metric_type == 'overall':
        return None
    elif metric_type == 'month':
        return 'month'
    elif metric_type == 'reach':
        return 'COMID'
    elif metric_type == 'month_reach':
        return ['COMID', 'month']

#metric_types = ['overall', 'month', 'reach', 'month_reach']
metric_types = ['overall']

for metric_type in metric_types:
    grp_arg = get_grp_arg(metric_type)
    combined_metrics(obs_file=config['obs_file_NHD'],
                     pred_trn=f"../{outdir}/trn_preds_Multi.feather",
                     pred_val=f"../{outdir}/val_preds_Multi.feather",
                     pred_tst=f"../{outdir}/tst_preds_Multi.feather",
                     group_spatially=False if not grp_arg else True if "COMID" in grp_arg else False,
                     group_temporally=False if not grp_arg else 'M' if "month" in grp_arg else False,
                     #outfile=f"{outdir}/{metric_type}_metrics.csv",
                     outfile=f"path to your river-dl/results_multiscale/saved_results_nopretrain/metrics/{basin}_{maskpercentage_formatted}_{random_seed}_{model_seed}_metrics.csv",
                     spatial_idx_name="COMID")





