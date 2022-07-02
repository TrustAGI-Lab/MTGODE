import argparse
import math
import time
import torch.nn as nn
import torch.optim as optim

from util import *
# from trainer import Optim
from model import DyGODE


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


parser = argparse.ArgumentParser(description='MTGODE')

# general settings
parser.add_argument('--expid', type=int, default=0, help='experiment id when saving best model')
parser.add_argument('--runs', type=int, default=1, help='number of runs')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--data', type=str, default='../data/solar_AL.txt')
parser.add_argument('--save', type=str, default='./save/', help='save path')
parser.add_argument('--save_preds', type=str_to_bool, default=True, help='whether to save prediction results')
parser.add_argument('--save_preds_path', type=str, default='./results/', help='predictions save path')
parser.add_argument('--num_nodes', type=int, default=137, help='number of nodes/variables')
parser.add_argument('--normalize', type=int, default=2, help='raw data normalization')
parser.add_argument('--in_dim', type=int, default=1, help='inputs dimension')
parser.add_argument('--seq_in_len', type=int, default=24*7, help='input sequence length')
parser.add_argument('--seq_out_len', type=int, default=1, help='output sequence length')
parser.add_argument('--horizon', type=int, default=3)

# training related
parser.add_argument('--epochs', type=int, default=40, help='')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.00001, help='weight decay rate')
parser.add_argument('--lr_decay', type=str_to_bool, default=False, help='whether to decrease lr during training')
parser.add_argument('--lr_decay_steps', type=str, default='20,40', help='lr decay at these steps')
parser.add_argument('--lr_decay_rate', type=float, default=0.5, help='how much lr will decay')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--clip', type=int, default=5, help='clip')
parser.add_argument('--L1Loss', type=str_to_bool, default=True, help='whether to use L1loss as criterion')
parser.add_argument('--optim', type=str, default='adam')

# model related
parser.add_argument('--buildA_true', type=str_to_bool, default=True, help='whether to construct adaptive adjacency matrix')
parser.add_argument('--subgraph_size', type=int, default=20, help='k')
parser.add_argument('--tanhalpha', type=float, default=3, help='tanh alpha')
parser.add_argument('--node_dim', type=int, default=40, help='dim of nodes')
parser.add_argument('--num_split', type=int, default=1, help='number of splits for graphs')
parser.add_argument('--step_size', type=int, default=100, help='step_size')
parser.add_argument('--dilation_exponential', type=int, default=2, help='dilation exponential')
parser.add_argument('--conv_channels', type=int, default=64, help='convolution channels')
parser.add_argument('--end_channels', type=int, default=64, help='end channels')
parser.add_argument('--solver_1', type=str, default='euler', help='CTA Solver')
parser.add_argument('--time_1', type=float, default=1.0, help='CTA integration time')
parser.add_argument('--step_1', type=float, default=0.167, help='CTA step size')
parser.add_argument('--solver_2', type=str, default='euler', help='CGP Solver')
parser.add_argument('--time_2', type=float, default=1.0, help='CGP integration time')
parser.add_argument('--step_2', type=float, default=0.25, help='CGP step size')
parser.add_argument('--alpha', type=float, default=2.0, help='CGP alpha to control eigenvalues range: [0, alpha]')
parser.add_argument('--rtol', type=float, default=1e-4, help='rtol')
parser.add_argument('--atol', type=float, default=1e-3, help='atol')
parser.add_argument('--adjoint', type=str_to_bool, default=False, help='whether to use adjoint method')
parser.add_argument('--perturb', type=str_to_bool, default=False, help='whether to use adjoint method')

args = parser.parse_args()
device = torch.device(args.device)
torch.set_num_threads(4)

print(args)


def evaluate(data, X, Y, model, evaluateL2, evaluateL1, batch_size, runid, save_prediction=False):
    model.eval()
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None

    for X, Y in data.get_batches(X, Y, batch_size, False):
        X = torch.unsqueeze(X,dim=1)
        X = X.transpose(2,3)
        with torch.no_grad():
            output = model(X)
            # RESET NFE
            model.ODE.odefunc.nfe = 0  # reset CTA nfe
            model.ODE.odefunc.stnet.gconv_1.CGPODE.odefunc.nfe = 0  # reset CGP 1 nfe
            model.ODE.odefunc.stnet.gconv_2.CGPODE.odefunc.nfe = 0  # reset CGP 2 nfe
        output = torch.squeeze(output)
        if len(output.shape)==1:
            output = output.unsqueeze(dim=0)
        if predict is None:
            predict = output
            test = Y
        else:
            predict = torch.cat((predict, output))
            test = torch.cat((test, Y))

        scale = data.scale.expand(output.size(0), data.m)
        total_loss += evaluateL2(output * scale, Y * scale).item()
        total_loss_l1 += evaluateL1(output * scale, Y * scale).item()
        n_samples += (output.size(0) * data.m)

    rse = math.sqrt(total_loss / n_samples) / data.rse
    rae = (total_loss_l1 / n_samples) / data.rae

    all_preds = predict
    all_reals = test.data
    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()
    sigma_p = (predict).std(axis=0)
    sigma_g = (Ytest).std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0)
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    correlation = (correlation[index]).mean()

    if save_prediction:
        all_preds = all_preds * data.scale.expand(all_preds.size(0), data.m)
        all_reals = all_reals * data.scale.expand(all_reals.size(0), data.m)
        all_preds = all_preds.data.cpu().numpy()
        all_reals = all_reals.data.cpu().numpy()
        print(all_preds.shape)
        print(all_reals.shape)
        np.save(args.save_preds_path + args.data.replace('data/', '').replace('.txt', '') + "_horizon" + str(args.horizon)
                + "_exp" + str(args.expid) + "_" + str(runid) + "_pred.npy", all_preds)
        np.save(args.save_preds_path + args.data.replace('data/', '').replace('.txt', '') +  "_horizon" + str(args.horizon)
                + "_exp" + str(args.expid) + "_" + str(runid) + "_true.npy", all_reals)

    return rse, rae, correlation


def train(data, X, Y, model, criterion, optim, batch_size, clip=None):
    model.train()
    total_loss = 0
    n_samples = 0
    iter = 0
    for X, Y in data.get_batches(X, Y, batch_size, True):
        optim.zero_grad()
        X = torch.unsqueeze(X,dim=1)
        X = X.transpose(2, 3)
        if iter % args.step_size == 0:
            perm = np.random.permutation(range(args.num_nodes))
        num_sub = int(args.num_nodes / args.num_split)

        for j in range(args.num_split):
            if j != args.num_split - 1:
                id = perm[j * num_sub:(j + 1) * num_sub]
            else:
                id = perm[j * num_sub:]
            id = torch.LongTensor(id).to(device)
            tx = X[:, :, id, :]
            ty = Y[:, id]  # (B, N)
            output = model(tx, id)
            output = torch.squeeze(output)  # (B, N)

            # GET/RESET NFE
            nfe_1 = model.ODE.odefunc.nfe  # get CTA nfe
            nfe_2 = model.ODE.odefunc.stnet.gconv_1.CGPODE.odefunc.nfe // nfe_1  # get CPG nfe
            model.ODE.odefunc.nfe = 0  # reset CTA nfe
            model.ODE.odefunc.stnet.gconv_1.CGPODE.odefunc.nfe = 0  # reset CGP 1 nfe
            model.ODE.odefunc.stnet.gconv_2.CGPODE.odefunc.nfe = 0  # reset CGP 2 nfe

            scale = data.scale.expand(output.size(0), data.m)  # (B, N)
            scale = scale[:,id]

            loss = criterion(output * scale, ty * scale)
            loss.backward()
            total_loss += loss.item()
            n_samples += (output.size(0) * data.m)

            if clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            optim.step()

        if iter % 100 == 0:
            print('iter:{:3d} | lr {:.6f} | loss: {:.3f} | CTA nfe:{:2d} | CGP nfe:{:2d}'
                  .format(iter, optim.param_groups[0]['lr'], loss.item()/(output.size(0) * data.m), nfe_1, nfe_2), flush=True)

        iter += 1

    return total_loss / n_samples


def main(runid):

    # train 60%, valid 20%, test 20%
    Data = DataLoaderS(args.data, 0.6, 0.2, device, args.horizon, args.seq_in_len, args.normalize)

    model = DyGODE(buildA_true=args.buildA_true, num_nodes=args.num_nodes, device=device,
                   dropout=args.dropout, subgraph_size=args.subgraph_size, node_dim=args.node_dim,
                   dilation_exponential=args.dilation_exponential, conv_channels=args.conv_channels,
                   end_channels=args.end_channels, seq_length=args.seq_in_len, in_dim=args.in_dim,
                   out_dim=args.seq_out_len, tanhalpha=args.tanhalpha, method_1=args.solver_1, time_1=args.time_1,
                   step_size_1=args.step_1, method_2=args.solver_2, time_2=args.time_2, step_size_2=args.step_2,
                   alpha=args.alpha, rtol=args.rtol, atol=args.atol, adjoint=args.adjoint, perturb=args.perturb,
                   ln_affine=False).to(device)

    print('The recpetive field size is', model.receptive_field)
    nParams = sum([p.nelement() for p in model.parameters()])
    print('Number of model parameters is', nParams, flush=True)

    if args.L1Loss:
        criterion = nn.L1Loss(size_average=False).to(device)
    else:
        criterion = nn.MSELoss(size_average=False).to(device)

    evaluateL2 = nn.MSELoss(size_average=False).to(device)
    evaluateL1 = nn.L1Loss(size_average=False).to(device)

    best_val = 10000000
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.lr_decay:
        lr_decay_steps = args.lr_decay_steps.split(',')
        lr_decay_steps = [int(i) for i in lr_decay_steps]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_steps, gamma=args.lr_decay_rate)

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        print('begin training')
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            train_loss = train(Data, Data.train[0], Data.train[1], model, criterion, optimizer, args.batch_size,
                               args.clip)
            val_loss, val_rae, val_corr = evaluate(Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1,
                                               args.batch_size, runid)
            print(
                '| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f}'.format(
                    epoch, (time.time() - epoch_start_time), train_loss, val_loss, val_rae, val_corr), flush=True)

            # Save the model if the validation loss is the best we've seen so far.
            if val_loss < best_val:
                torch.save(model, args.save + args.data.replace('data/', '').replace('.txt', '') +
                           "_exp" + str(args.expid) + "_" + str(runid) + ".pt")
                best_val = val_loss

            if epoch % 5 == 0:
                test_acc, test_rae, test_corr = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1,
                                                     args.batch_size, runid)
                print("test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_acc, test_rae, test_corr), flush=True)

            if args.lr_decay:
                scheduler.step()

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    model = torch.load(args.save + args.data.replace('data/', '').replace('.txt', '') + "_exp" + str(args.expid) +
                       "_" + str(runid) + ".pt")

    vtest_acc, vtest_rae, vtest_corr = evaluate(Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1,
                                         args.batch_size, runid)
    test_acc, test_rae, test_corr = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1,
                                         args.batch_size, runid, save_prediction=args.save_preds)
    print("final test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_acc, test_rae, test_corr))

    return vtest_acc, vtest_rae, vtest_corr, test_acc, test_rae, test_corr


if __name__ == "__main__":
    vacc = []
    vrae = []
    vcorr = []
    acc = []
    rae = []
    corr = []
    for i in range(args.runs):
        val_acc, val_rae, val_corr, test_acc, test_rae, test_corr = main(i)
        vacc.append(val_acc)
        vrae.append(val_rae)
        vcorr.append(val_corr)
        acc.append(test_acc)
        rae.append(test_rae)
        corr.append(test_corr)
    print('\n\n')
    print('multiple runs average')
    print('\n\n')
    print("valid\trse\trae\tcorr")
    print("mean\t{:5.4f}\t{:5.4f}\t{:5.4f}".format(np.mean(vacc), np.mean(vrae), np.mean(vcorr)))
    print("std\t{:5.4f}\t{:5.4f}\t{:5.4f}".format(np.std(vacc), np.std(vrae), np.std(vcorr)))
    print('\n\n')
    print("test\trse\trae\tcorr")
    print("mean\t{:5.4f}\t{:5.4f}\t{:5.4f}".format(np.mean(acc), np.mean(rae), np.mean(corr)))
    print("std\t{:5.4f}\t{:5.4f}\t{:5.4f}".format(np.std(acc), np.std(rae), np.std(corr)))