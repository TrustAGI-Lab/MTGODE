import torch
import torch.optim as optim
import util


class Trainer():
    def __init__(self, model, lrate, wdecay, clip, step_size, seq_out_len, scaler, device, cl=True):
        self.scaler = scaler
        self.model = model
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = util.masked_mae
        # self.loss = util.masked_mse
        # self.loss = util.masked_rmses
        self.clip = clip
        self.step = step_size
        self.iter = 1
        self.task_level = 1
        self.seq_out_len = seq_out_len
        self.cl = cl

    def train(self, input, real_val, idx=None):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input, idx=idx).transpose(1, 3)
        nfe_1 = self.model.ODE.odefunc.nfe  # get CTA nfe
        nfe_2 = self.model.ODE.odefunc.stnet.gconv_1.CGPODE.odefunc.nfe // nfe_1  # get CPG nfe
        self.model.ODE.odefunc.nfe = 0  # reset CTA nfe
        self.model.ODE.odefunc.stnet.gconv_1.CGPODE.odefunc.nfe = 0  # reset CGP 1 nfe
        self.model.ODE.odefunc.stnet.gconv_2.CGPODE.odefunc.nfe = 0  # reset CGP 2 nfe
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)
        if self.iter % self.step == 0 and self.task_level <= self.seq_out_len:
            self.task_level += 1
        if self.cl:
            loss = self.loss(predict[:, :, :, :self.task_level], real[:, :, :, :self.task_level], 0.0)
        else:
            loss = self.loss(predict, real, 0.0)

        loss.backward()

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

        self.optimizer.step()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        self.iter += 1

        return loss.item(), mape, rmse, nfe_1, nfe_2

    def eval(self, input, real_val):
        self.model.eval()
        output = self.model(input)
        self.model.ODE.odefunc.nfe = 0  # reset CTA nfe
        self.model.ODE.odefunc.stnet.gconv_1.CGPODE.odefunc.nfe = 0  # reset CGP 1 nfe
        self.model.ODE.odefunc.stnet.gconv_2.CGPODE.odefunc.nfe = 0  # reset CGP 2 nfe
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()

        return loss.item(), mape, rmse