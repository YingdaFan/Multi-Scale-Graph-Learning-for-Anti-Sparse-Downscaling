import numpy as np
import torch
import torch.utils.data
import pandas as pd
import time
from tqdm import tqdm
from river_dl.min_norm_solvers import MinNormSolver, gradient_normalizers
from torch.utils.data import TensorDataset, DataLoader


def train_loop(epoch_index, x_train_loader, y_train_loader, model, loss_function, optimizer, device='cpu'):
    train_loss = []

    with tqdm(x_train_loader, ncols=100, desc=f"Epoch {epoch_index + 1}", unit="batch") as x_tepoch, tqdm(
            y_train_loader, ncols=100,
            desc=f"Epoch {epoch_index + 1}", unit="batch") as y_tepoch:
        for x, y in zip(x_tepoch, y_tepoch):
            trainx = x.to(device)
            trainy = y.to(device)
            if torch.isnan(trainy).all():
                continue
            optimizer.zero_grad()
            output, _ = model.forward_task1(trainx)
            loss = loss_function(trainy, output)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()
            train_loss.append(loss.item())
            x_tepoch.set_postfix(loss=loss.item())
    mean_loss = np.mean(train_loss)
    return mean_loss


def pretrain(model,
             loss_function,
             optimizer,
             x_train,
             y_train,
             x_batch_size,
             y_batch_size,
             max_epochs,
             shuffle=False,
             weights_file=None,
             log_file=None,
             device='cpu'):
    print(f"Training on {device}")
    print("start training...", flush=True)

    x_train_data = []
    y_train_data = []
    for i in range(len(x_train)):
        x_train_data.append(torch.from_numpy(x_train[i]).float())
    for i in range(len(y_train)):
        y_train_data.append(torch.from_numpy(y_train[i]).float())

    x_train_loader = torch.utils.data.DataLoader(x_train_data, batch_size=x_batch_size, shuffle=shuffle,
                                                 pin_memory=True)
    y_train_loader = torch.utils.data.DataLoader(y_train_data, batch_size=y_batch_size, shuffle=shuffle,
                                                 pin_memory=True)

    model.to(device)
    log_cols = ['epoch', 'loss', 'val_loss']
    train_log = pd.DataFrame(columns=log_cols)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 50], gamma=0.3)
    for i in range(max_epochs):
        scheduler.step()
        model.train()
        epoch_loss = train_loop(i, x_train_loader, y_train_loader, model, loss_function, optimizer, device)
        train_log = pd.concat(
            [train_log, pd.DataFrame([[i, epoch_loss, np.nan]], columns=log_cols, index=[i])])
    train_log.to_csv(log_file)
    torch.save(model.state_dict(), weights_file)
    print("Finished Pretrain")
    return model


def my_train_torch(model,
                   loss_function,
                   optimizer,
                   x_train,
                   x_train_NHD,
                   y_train,
                   y_train_NHD,
                   x_batch_size,
                   x_batch_size_NHD,
                   y_batch_size,
                   y_batch_size_NHD,
                   max_epochs,
                   early_stopping_patience=False,
                   x_val=None,
                   x_val_NHD=None,
                   y_val=None,
                   y_val_NHD=None,
                   x_tst=None,
                   x_tst_NHD=None,
                   y_tst=None,
                   y_tst_NHD=None,
                   shuffle=False,
                   weights_file=None,
                   log_file=None,
                   device='cpu',
                   keep_portion=None):
    print(f"Training on {device}")
    print("start training...", flush=True)

    if not early_stopping_patience:
        early_stopping_patience = max_epochs

    epochs_since_best = 0
    best_loss = 1000

    if keep_portion is not None:
        if keep_portion > 1:
            period = int(keep_portion)
        else:
            period = int(keep_portion * y_train.shape[1])
        y_train[:, :-period, ...] = np.nan
        if y_val is not None:
            y_val[:, :-period, ...] = np.nan

    x_train_data = []
    x_train_data_NHD = []
    y_train_data = []
    y_train_data_NHD = []

    for i in range(len(x_train)):
        x_train_data.append(torch.from_numpy(x_train[i]).float())
    for i in range(len(x_train_NHD)):
        x_train_data_NHD.append(torch.from_numpy(x_train_NHD[i]).float())
    for i in range(len(y_train)):
        y_train_data.append(torch.from_numpy(y_train[i]).float())
    for i in range(len(y_train_NHD)):
        y_train_data_NHD.append(torch.from_numpy(y_train_NHD[i]).float())

    x_train_loader = torch.utils.data.DataLoader(x_train_data, batch_size=x_batch_size, shuffle=shuffle,
                                                 pin_memory=True)
    x_train_loader_NHD = torch.utils.data.DataLoader(x_train_data_NHD, batch_size=x_batch_size_NHD, shuffle=shuffle,
                                                     pin_memory=True)
    y_train_loader = torch.utils.data.DataLoader(y_train_data, batch_size=y_batch_size, shuffle=shuffle,
                                                 pin_memory=True)
    y_train_loader_NHD = torch.utils.data.DataLoader(y_train_data_NHD, batch_size=y_batch_size_NHD, shuffle=shuffle,
                                                     pin_memory=True)

    if x_val is not None:
        x_val_data = []
        y_val_data = []
        for i in range(len(x_val)):
            x_val_data.append(torch.from_numpy(x_val[i]).float())
        for i in range(len(y_val)):
            y_val_data.append(torch.from_numpy(y_val[i]).float())
        x_val_loader = torch.utils.data.DataLoader(x_val_data, batch_size=x_batch_size, shuffle=shuffle,
                                                   pin_memory=True)
        y_val_loader = torch.utils.data.DataLoader(y_val_data, batch_size=y_batch_size, shuffle=shuffle,
                                                   pin_memory=True)

    if x_val_NHD is not None:
        x_val_data_NHD = []
        y_val_data_NHD = []
        for i in range(len(x_val_NHD)):
            x_val_data_NHD.append(torch.from_numpy(x_val_NHD[i]).float())
        for i in range(len(y_val_NHD)):
            y_val_data_NHD.append(torch.from_numpy(y_val_NHD[i]).float())
        x_val_loader_NHD = torch.utils.data.DataLoader(x_val_data_NHD, batch_size=x_batch_size_NHD, shuffle=shuffle,
                                                       pin_memory=True)
        y_val_loader_NHD = torch.utils.data.DataLoader(y_val_data_NHD, batch_size=y_batch_size_NHD, shuffle=shuffle,
                                                       pin_memory=True)

    if x_tst is not None:
        x_tst_data = []
        y_tst_data = []
        for i in range(len(x_tst)):
            x_tst_data.append(torch.from_numpy(x_tst[i]).float())
        for i in range(len(y_tst)):
            y_tst_data.append(torch.from_numpy(y_tst[i]).float())
        x_tst_loader = torch.utils.data.DataLoader(x_tst_data, batch_size=x_batch_size, shuffle=shuffle,
                                                   pin_memory=True)
        y_tst_loader = torch.utils.data.DataLoader(y_tst_data, batch_size=y_batch_size, shuffle=shuffle,
                                                   pin_memory=True)

    if x_tst_NHD is not None:
        x_tst_data_NHD = []
        y_tst_data_NHD = []
        for i in range(len(x_tst_NHD)):
            x_tst_data_NHD.append(torch.from_numpy(x_tst_NHD[i]).float())
        for i in range(len(y_tst_NHD)):
            y_tst_data_NHD.append(torch.from_numpy(y_tst_NHD[i]).float())
        x_tst_loader_NHD = torch.utils.data.DataLoader(x_tst_data_NHD, batch_size=x_batch_size_NHD, shuffle=shuffle,
                                                       pin_memory=True)
        y_tst_loader_NHD = torch.utils.data.DataLoader(y_tst_data_NHD, batch_size=y_batch_size_NHD, shuffle=shuffle,
                                                       pin_memory=True)

    val_time = []
    train_time = []
    log_cols = ['epoch', 'trn_loss', 'trn_loss_NHD', 'val_loss_NHD', 'tst_loss_NHD']
    train_log = pd.DataFrame(columns=log_cols)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 50], gamma=0.3)

    # 测试动态学习率，记得删掉

    tasks = [0, 1, 2]


    for epoch in range(max_epochs):
        scheduler.step()  # 测试动态学习率，记得删掉
        model.train()

        t1 = time.time()
        combined_loaders = zip(zip(x_train_loader, y_train_loader), zip(x_train_loader_NHD, y_train_loader_NHD))
        trainloss1 = []
        trainloss2 = []

        with tqdm(combined_loaders, ncols=100, desc=f"Epoch {epoch + 1}", unit="batch") as tepoch:
            for ((x1, y1), (x2, y2)) in tepoch:

                trainx1 = x1.to(device)
                trainy1 = y1.to(device)
                trainx2 = x2.to(device)
                trainy2 = y2.to(device)
                if torch.isnan(trainy1).all():
                    continue

                loss_data = {}
                grads = {}
                scale = {}

                optimizer.zero_grad()
                output = model.forward_task0(trainx1)
                loss = loss_function(trainy1, output)
                loss_data[0] = loss.item()

                # 原文在这里使用loss.data[0]来尝试该值与计算图分离并将其转换为标准Python数字
                # 访问张量的数据，而不跟踪历史（或梯度）
                loss.backward()
                grads[0] = []
                for param in model.get_shared_parameters():
                    if param.grad is not None:
                        grads[0].append(param.grad.clone().detach())
                        # 克隆参数的梯度，并创建一个新的张量，这个新张量与原始的计算图分离，它不会对反向传播产生影响

                if torch.isnan(trainy2).all():
                    # 2.28 我认为这里应该加上单独用y1训练的梯度更新参数的情况
                    # optimizer.step()
                    continue
                optimizer.zero_grad()
                output, hidden_seq1 = model.forward_task1(trainx1)
                hidden_seq = hidden_seq1.detach()
                # 加了一行这个，希望有用
                loss = loss_function(trainy2, output)

                loss_data[1] = loss.item()
                loss.backward()
                grads[1] = []
                for param in model.get_shared_parameters():
                    if param.grad is not None:
                        grads[1].append(param.grad.clone().detach())

                tepoch.set_postfix(loss=loss.item())
                trainloss1.append(loss.item())

                # output1, hidden_seq = model.forward_task1(trainx1)
                # 重要：重新获取 hidden_seq，否则报错'the saved intermediate results have already been freed'
                optimizer.zero_grad()
                output = model.forward_task2(trainx2, hidden_seq)
                loss = loss_function(trainy2, output)
                loss_data[2] = loss.item()
                loss.backward()
                grads[2] = []
                for param in model.get_shared_parameters():
                    if param.grad is not None:
                        grads[2].append(param.grad.clone().detach())

                tepoch.set_postfix(loss=loss.item())
                trainloss2.append(loss.item())

                gn = gradient_normalizers(grads, loss_data, 'loss+')
                for t in tasks:
                    for gr_i in range(len(grads[t])):
                        grads[t][gr_i] = grads[t][gr_i] / gn[t]

                # Frank-Wolfe iteration to compute scales.
                sol, min_norm = MinNormSolver.find_min_norm_element([grads[t] for t in tasks])
                print('solis', sol)
                for i, t in enumerate(tasks):
                    scale[t] = float(sol[i])

                # Scaled back-propagation
                optimizer.zero_grad()

                output = model.forward_task0(trainx1)
                loss_t = loss_function(trainy1, output)
                loss_data[0] = loss_t.item()
                loss = scale[0] * loss_t

                output, hidden_seq = model.forward_task1(trainx1)
                hidden_seq1 = hidden_seq.detach()
                loss_t = loss_function(trainy2, output)
                loss_data[1] = loss_t.item()
                loss = loss + scale[1] * loss_t

                output = model.forward_task2(trainx2, hidden_seq1)
                loss_t = loss_function(trainy2, output)
                loss_data[2] = loss_t.item()
                loss = loss + scale[2] * loss_t

                loss.backward()
                optimizer.step()

        epoch_trn_loss = np.mean(trainloss1)
        epoch_trn_loss_NHD = np.mean(trainloss2)
        train_time.append(time.time() - t1)

#        if epoch == 59:  # 2024.4.5 我把模型存贮函数改在了这里，跑完实验记得删除
#            print('hhhhhhhhhhhhhhhhhhhhhhhhhhhhsavesavasavesave savesave save!!!!')
#            torch.save(model.state_dict(), weights_file)


        # Val
        valloss = []
        s1 = time.time()
        model.eval()
        for ((x1val, y1val), (x2val, y2val)) in zip(zip(x_val_loader, y_val_loader),
                                                    zip(x_val_loader_NHD, y_val_loader_NHD)):
            valx1 = x1val.to(device)
            valx2 = x2val.to(device)
            valy2 = y2val.to(device)
            output, hidden_seq = model.forward_task1(valx1)

            output2 = model.forward_task2(valx2, hidden_seq)
            loss2 = loss_function(valy2, output2)
            valloss.append(loss2.item())
        epoch_val_loss_NHD = np.mean(valloss)
        print(f"Valid loss_NHD: {epoch_val_loss_NHD:.2f}")

        # Tst
        tstloss = []
        s1 = time.time()
        model.eval()
        for ((x1tst, y1tst), (x2tst, y2tst)) in zip(zip(x_tst_loader, y_tst_loader),
                                                    zip(x_tst_loader_NHD, y_tst_loader_NHD)):
            tstx1 = x1tst.to(device)
            tstx2 = x2tst.to(device)
            tsty2 = y2tst.to(device)
            output, hidden_seq = model.forward_task1(tstx1)
            output2 = model.forward_task2(tstx2, hidden_seq)
            loss2 = loss_function(tsty2, output2)
            tstloss.append(loss2.item())
        epoch_tst_loss_NHD = np.mean(tstloss)
        print(f"Test loss_NHD: {epoch_tst_loss_NHD:.2f}")


        if epoch_val_loss_NHD < best_loss:
            torch.save(model.state_dict(), weights_file)
            best_loss = epoch_val_loss_NHD
            epochs_since_best = 0
        else:
            epochs_since_best += 1
        if epochs_since_best > early_stopping_patience:
            print(f"Early Stopping at Epoch {epoch}")
            break

        epoch_data = {
            'epoch': epoch,
            'trn_loss': epoch_trn_loss,
            'trn_loss_NHD': epoch_trn_loss_NHD,
            'val_loss_NHD': epoch_val_loss_NHD,
            'tst_loss_NHD': epoch_tst_loss_NHD,
        }
        current_epoch_log = pd.DataFrame([epoch_data], columns=log_cols)
        train_log = pd.concat([train_log, current_epoch_log], ignore_index=True)

    train_log.to_csv(log_file)

    if x_val is None:
        torch.save(model.state_dict(), weights_file)
        print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    else:
        print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
        print("Average Validation (Inference) Time: {:.4f} secs/epoch".format(np.mean(val_time)))
    return model


def rmse_masked(y_true, y_pred):
    num_y_true = torch.count_nonzero(~torch.isnan(y_true))
    if num_y_true > 0:
        zero_or_error = torch.where(
            torch.isnan(y_true), torch.zeros_like(y_true), y_pred - y_true
        )
        sum_squared_errors = torch.sum(torch.square(zero_or_error))
        rmse_loss = torch.sqrt(sum_squared_errors / num_y_true)
    else:
        # rmse_loss = 0.0
        rmse_loss = torch.tensor(0.0, device=y_true.device, dtype=y_true.dtype)
    return rmse_loss


def rmse_eval(y_true, y_pred):
    y_pred = y_pred[~np.isnan(y_true)]
    y_true = y_true[~np.isnan(y_true)]
    n = len(y_true)
    sum_squared_error = np.sum(np.square(y_pred - y_true))
    rmse = np.sqrt(sum_squared_error / n)
    return rmse

