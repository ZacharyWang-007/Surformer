from models.attention_modules import *
from torch.nn import init


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')  # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
    if hasattr(m, 'bias') and m.bias is not None:
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        # init.constant_(m.bias.data, 0.0)


class MIL_Sum_FC_surv(nn.Module):
    def __init__(self, size_arg="small", dropout=0.25, n_classes=4):
        r"""
        Deep Sets Implementation.

        Args:
            size_arg (str): Size of NN architecture (Choices: small or large)
            dropout (float): Dropout rate
            n_classes (int): Output shape of NN
        """
        super(MIL_Sum_FC_surv, self).__init__()
        self.size_dict_path = {"small": [1024, 512, 256], "big": [1024, 512, 384]}

        # Deep Sets Architecture Construction
        size = self.size_dict_path[size_arg]
        self.phi = nn.Sequential(*[nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)])
        self.rho = nn.Sequential(*[nn.Linear(size[1], size[2]), nn.ReLU(), nn.Dropout(dropout)])

        self.classifier = nn.Linear(size[2], n_classes)

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() >= 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.phi = nn.DataParallel(self.phi, device_ids=device_ids).to('cuda:0')

        self.rho = self.rho.to(device)
        self.classifier = self.classifier.to(device)

    def forward(self, **kwargs):
        x_path = kwargs['x_path']

        h_path = self.phi(x_path).sum(axis=0)
        h_path = self.rho(h_path)
        h = h_path  # [256] vector

        logits = self.classifier(h).unsqueeze(0)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)

        return hazards, S, Y_hat, h, None


class MIL_Attention_FC_surv(nn.Module):
    def __init__(self, size_arg="small", dropout=0.25, n_classes=4):
        super(MIL_Attention_FC_surv, self).__init__()

        self.num_prototype = 5
        self.dropout_rate = 0.25
        self.c_local = 128
        self.c_global = 512
        self.p_threshold = 1 / (self.num_prototype-1)

        self.prototype = nn.Parameter(torch.randn((1, self.num_prototype, 512), requires_grad=True))

        self.non_lin = nn.Sequential(nn.Linear(512 * 2, 512),
                                     nn.LayerNorm(512),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, 512))

        self.ln_0 = nn.LayerNorm(512)
        self.ln_1 = nn.LayerNorm(512)
        self.ln_2 = nn.LayerNorm(512)
        self.ln_3 = nn.LayerNorm(512)
        self.ln_4 = nn.LayerNorm(512)
        self.ln_5 = nn.LayerNorm(512)
        self.ln_6 = nn.LayerNorm(512)
        self.ln_7 = nn.LayerNorm(512)
        self.ln_8 = nn.LayerNorm(512)
        self.ln_local = nn.LayerNorm((self.num_prototype - 1) * self.c_local)

        self.cross_attention_0 = Cross_Attention(dropout=self.dropout_rate)
        self.cross_attention_1 = Cross_Attention(dropout=self.dropout_rate)

        self.self_attention_0 = Self_Attention(dropout=self.dropout_rate)

        self.ffn_0 = FeedForward(dim=512, dropout=self.dropout_rate)
        self.ffn_1 = FeedForward(dim=512, dropout=self.dropout_rate)
        self.ffn_2 = FeedForward(dim=512, dropout=self.dropout_rate)
        self.ffn_3 = FeedForward(dim=512, dropout=self.dropout_rate)

        self.compress_local = nn.Sequential(nn.Linear(512, self.c_local),
                                            nn.LayerNorm(self.c_local),
                                            nn.ReLU(inplace=True))

        self.classifier_global = nn.Linear(
            self.c_global, n_classes, bias=False)
        self.classifier_local = nn.Linear(
            self.c_local*(self.num_prototype-1), n_classes, bias=False)
        self.classifier_overall = nn.Linear(
            self.c_local * (self.num_prototype - 1) + self.c_global, n_classes, bias=False)

        # self.classifier_global = nn.Linear(self.c_global, n_classes)
        # self.classifier_local = nn.Linear(self.c_local * (self.num_prototype - 1), n_classes)
        # self.classifier_overall = nn.Linear(self.c_local * (self.num_prototype - 1) + self.c_global, n_classes)

        self.compress_local.apply(weights_init_kaiming)
        self.classifier_global.apply(weights_init_classifier)
        self.classifier_local.apply(weights_init_classifier)
        self.classifier_overall.apply(weights_init_classifier)

    def forward_global(self, x):
        logits = self.classifier_global(x)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        return hazards, S, Y_hat

    def forward_local(self, x):
        logits = self.classifier_local(x)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        return hazards, S, Y_hat

    def forward_overall(self, x):
        logits = self.classifier_overall(x)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        return hazards, S, Y_hat

    def forward(self, **kwargs):
        x_path = kwargs['x_path']
        x_ori = self.non_lin(x_path)[None, ]

        x, attn = self.cross_attention_0(self.ln_0(self.prototype), self.ln_1(x_ori))

        # for weight generation
        attn = attn[:, 1:, :]
        attn = F.softmax(attn, dim=1)
        value, attn = torch.max(attn, dim=1)
        value = (value > self.p_threshold).int()

        attn_weight = []
        for i in range(self.num_prototype-1):
            temp = ((attn == i).int()*value).sum()
            attn_weight.append(temp[None, ])

        attn_weight = torch.cat(attn_weight, dim=0)
        attn_weight = attn_weight / value.sum()

        # print(attn_weight)

        x = self.ffn_0(self.ln_2(x)) + x

        x = self.self_attention_0(self.ln_3(x)) + x
        x = self.ffn_1(self.ln_4(x)) + x

        x = self.cross_attention_0(self.ln_0(self.prototype), self.ln_5(x))[0] + x
        x = self.ffn_2(self.ln_6(x)) + x

        x_global, x_local = x[:, 0, :][:, None, :], x[:, 1:, :]
        x_local = (attn_weight[None, :, None] + 1) * x_local
        x_local = x_local.view(1, self.num_prototype-1, 512)

        x = torch.cat((x_global, x_local), dim=1)
        x = self.self_attention_0(self.ln_7(x)) + x
        x = self.ffn_3(self.ln_8(x)) + x

        x_global, x_local = x[:, 0, :], x[:, 1:, :]

        x_local = self.compress_local(x_local).view(1, -1)
        x_local = self.ln_local(x_local)

        if self.training:
            hazards_gl, S_gl, Y_hat_gl = self.forward_global(x_global)
            hazards_lo, S_lo, Y_hat_lo = self.forward_local(x_local)
            hazards_ov, S_ov, Y_hat_ov = self.forward_overall(torch.cat((x_global, x_local), dim=-1))

            return hazards_gl, S_gl, hazards_lo, S_lo, hazards_ov, S_ov, x_local.view(self.num_prototype-1, -1)
        else:
            hazards_ov, S_ov, Y_hat_ov = self.forward_overall(torch.cat((x_global, x_local), dim=-1))
            return hazards_ov, S_ov, Y_hat_ov, x, None
