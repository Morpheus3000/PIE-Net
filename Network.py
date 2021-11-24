import torch
import torch.nn as nn
import torch.nn.functional as F


class ratioCals(nn.Module):
    """
    Class to handle all the extra ratio calculations. Exposed as layers to a
    network for future reuse.
    """

    def __init__(self, level1_probThreshold=1e-2, level2_probThreshold=1e-4, log_space=False):
        super(ratioCals, self).__init__()
        self.level1_probThreshold = level1_probThreshold
        self.level2_probThreshold = level2_probThreshold
        self.log_space = log_space
        self.register_buffer('filter', torch.Tensor([[[[0, 1, 0],
                                                     [1, 0, -1],
                                                     [0, -1, 0]]]]))

        self.register_buffer('filter2', torch.Tensor([[[[0, -1, 0],
                                                      [-1, 0, 1],
                                                      [0, 1, 0]]]]))

    def getGaussCrossRatios(self, img):

        zeroMasks = torch.zeros_like(img)
        zeroMasks[img == 0] = 1
        crossed_img = torch.zeros_like(img)

        log_img = torch.log(img + 1e-7)

        red_chan = log_img[:, 0, :, :].unsqueeze(1)
        green_chan = log_img[:, 1, :, :].unsqueeze(1)
        blue_chan = log_img[:, 2, :, :].unsqueeze(1)

        # Red-Green
        filt_r1 = F.conv2d(red_chan, weight=self.filter, padding=1)
        filt_g1 = F.conv2d(green_chan, weight=self.filter2, padding=1)
        filt_rg = filt_r1 + filt_g1
        filt_rg = torch.clamp(filt_rg, -1.0, 1.0)
        filt_rg.squeeze_(1)

        # Green-Blue
        filt_g2 = F.conv2d(green_chan, weight=self.filter, padding=1)
        filt_b1 = F.conv2d(blue_chan, weight=self.filter2, padding=1)
        filt_gb = filt_g2 + filt_b1
        filt_gb = torch.clamp(filt_gb, -1.0, 1.0)
        filt_gb.squeeze_(1)

        # Red-Blue
        filt_r2 = F.conv2d(red_chan, weight=self.filter, padding=1)
        filt_b2 = F.conv2d(blue_chan, weight=self.filter2, padding=1)
        filt_rb = filt_r2 + filt_b2
        filt_rb = torch.clamp(filt_rb, -1.0, 1.0)
        filt_rb.squeeze_(1)

        if self.log_space:
            crossed_img[:, 0, :, :] = filt_rg
            crossed_img[:, 2, :, :] = filt_gb
            crossed_img[:, 1, :, :] = filt_rb
        else:
            crossed_img[:, 0, :, :] = torch.exp(filt_rg)
            crossed_img[:, 1, :, :] = torch.exp(filt_gb)
            crossed_img[:, 2, :, :] = torch.exp(filt_rb)
            crossed_img = crossed_img - 1e-7

        crossed_img[zeroMasks == 1]=0
        return crossed_img

        def forward(self, img):
            shadowMasks = self.intrinsicBordersMasks(img)
            output_dict = {'shadow_regions': shadowMasks}
            return output_dict

class gaussCrossRatioCal(ratioCals):
    """
    Class to calculate the cross ratio, using a discrete filter.
    """

    def __init__(self):
        super(gaussCrossRatioCal, self).__init__()

    def forward(self, img):
        crossRatio = self.getGaussCrossRatios(img)
        return {'cross': crossRatio}

class AttentionLayer(nn.Module):

    def __init__(self, learnable=False):
        super(AttentionLayer, self).__init__()
        self.learnable = learnable
        self.makeAttention()

    def makeAttention(self):
        if self.learnable:
            self.learn = nn.Conv2d(3, 3, 3, 1, 1)

        self.sig = nn.Sigmoid()

    def forward(self, left_inp, right_inp):
        # print('Left: ', left_inp.shape)
        # print('Right: ', right_inp.shape)
        sigged = self.sig(left_inp)
        mulled = sigged * right_inp
        attentioned = mulled + right_inp
        return attentioned

class VGGEncoderBatchNorm(nn.Module):

    def __init__(self):
        super(VGGEncoderBatchNorm, self).__init__()
        self.makeImgEncoder()
        self.makeCCREncoder()
        self.crossRatio = gaussCrossRatioCal()

    def makeImgEncoder(self):
        self.conv0 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

    def makeCCREncoder(self):
        self.cross0 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.cross1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )

        self.cross2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )

        self.cross3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

        self.cross4 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

    def forward(self, img):
        return_dict = {}
        cross_img = self.crossRatio(img)

        conv00 = self.conv0(img)
        conv10 = self.conv1(conv00)
        conv20 = self.conv2(conv10)
        conv30 = self.conv3(conv20)
        conv40 = self.conv4(conv30)

        return_dict['conv00'] = conv00
        return_dict['conv10'] = conv10
        return_dict['conv20'] = conv20
        return_dict['conv30'] = conv30
        return_dict['conv40'] = conv40

        cross00 = self.cross0(cross_img['cross'])
        cross10 = self.cross1(cross00)
        cross20 = self.cross2(cross10)
        cross30 = self.cross3(cross20)
        cross40 = self.cross4(cross30)

        return_dict['cross00'] = cross00
        return_dict['cross10'] = cross10
        return_dict['cross20'] = cross20
        return_dict['cross30'] = cross30
        return_dict['cross40'] = cross40
        return_dict['cross_img'] = cross_img['cross']

        return return_dict

class VGGScaleClampEdgeDecoderBatchNorm(nn.Module):

    def __init__(self):
        super(VGGScaleClampEdgeDecoderBatchNorm, self).__init__()
        self.makeLinkedEdgeDecoder()

    def makeLinkedEdgeDecoder(self):
        self.edge_deconvs0 = nn.ModuleList()
        for i in range(2):
            edge0 = nn.Sequential(
                nn.ConvTranspose2d(512 * 2, 512, 4, 2, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(True)
            )
            self.edge_deconvs0.append(edge0)

        self.edge_deconvs1 = nn.ModuleList()
        for i in range(2):
            edge1 = nn.Sequential(
                nn.ConvTranspose2d(512 * 4, 512, 4, 2, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(True)
            )
            self.edge_deconvs1.append(edge1)

        self.edge_deconvs2 = nn.ModuleList()
        for i in range(2):
            edge2 = nn.Sequential(
                nn.ConvTranspose2d((512 * 2) + (256 * 2), 256, 4, 2, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(True)
            )
            self.edge_deconvs2.append(edge2)

        self.edge_deconvs3 = nn.ModuleList()
        for i in range(2):
            edge3 = nn.Sequential(
                nn.ConvTranspose2d((256 * 2) + (128 * 2), 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(True)
            )
            self.edge_deconvs3.append(edge3)

        self.edge_output = nn.ModuleList()
        for i in range(2):
            output = nn.Sequential(
                nn.Conv2d((128 * 2) + (64 * 2), 64, 3, 1, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(True),

                nn.Conv2d(64, 3, 3, 1, 1),
                nn.BatchNorm2d(3),
                nn.ReLU(True)
            )
            self.edge_output.append(output)

        self.edge_side_output_1 = nn.Sequential(
            nn.Conv2d(512, 3, 3, 1, 1),
            nn.BatchNorm2d(3),
            nn.ReLU(True)
        )

        self.edge_side_output_2 = nn.Sequential(
            nn.Conv2d(256, 3, 3, 1, 1),
            nn.BatchNorm2d(3),
            nn.ReLU(True)
        )

        self.edge_side_output_3 = nn.Sequential(
            nn.Conv2d(128, 3, 3, 1, 1),
            nn.BatchNorm2d(3),
            nn.ReLU(True)
        )

        self.illum_edge_side_output_1 = nn.Sequential(
            nn.Conv2d(512, 3, 3, 1, 1),
            nn.BatchNorm2d(3),
            nn.ReLU(True)
        )

        self.illum_edge_side_output_2 = nn.Sequential(
            nn.Conv2d(256, 3, 3, 1, 1),
            nn.BatchNorm2d(3),
            nn.ReLU(True)
        )

        self.illum_edge_side_output_3 = nn.Sequential(
            nn.Conv2d(128, 3, 3, 1, 1),
            nn.BatchNorm2d(3),
            nn.ReLU(True)
        )

    def forward(self, encoderDict):

        edge_dict = {}
        mid_comb = torch.cat([encoderDict['conv40'], encoderDict['cross40']], 1)

        edge_deconvs0_list = [deconvs0(mid_comb) for deconvs0 in
                              self.edge_deconvs0]
        edge_deconvs0_list.append(encoderDict['conv30'])
        edge_deconvs0_list.append(encoderDict['cross30'])
        edge_deconvs0_comb = torch.cat(edge_deconvs0_list, 1)

        edge_deconvs1_list = [deconvs1(edge_deconvs0_comb) for deconvs1 in
                              self.edge_deconvs1]
        edge_deconvs1_list.append(encoderDict['conv20'])
        edge_deconvs1_list.append(encoderDict['cross20'])
        edge_deconvs1_comb = torch.cat(edge_deconvs1_list, 1)

        edge_deconvs2_list = [deconvs2(edge_deconvs1_comb) for deconvs2 in
                              self.edge_deconvs2]
        edge_deconvs2_list.append(encoderDict['conv10'])
        edge_deconvs2_list.append(encoderDict['cross10'])
        edge_deconvs2_comb = torch.cat(edge_deconvs2_list, 1)

        edge_deconvs3_list = [deconvs3(edge_deconvs2_comb) for deconvs3 in
                              self.edge_deconvs3]
        edge_deconvs3_list.append(encoderDict['conv00'])
        edge_deconvs3_list.append(encoderDict['cross00'])
        edge_deconvs3_comb = torch.cat(edge_deconvs3_list, 1)

        edge_output_list = [output(edge_deconvs3_comb) for output in
                              self.edge_output]

        reflec_side_output_1 = self.edge_side_output_1(edge_deconvs1_list[0])
        reflec_side_output_1 = torch.clamp(reflec_side_output_1, 0, 1)
        reflec_side_output_1_m = torch.mean(reflec_side_output_1, dim=1, keepdim=True)
        reflec_side_output_1_m = reflec_side_output_1_m / reflec_side_output_1_m.max()
        reflec_side_output_1_mask = torch.zeros_like(reflec_side_output_1_m)
        reflec_side_output_1_mask[reflec_side_output_1_m > 0.1] = 1
        reflec_side_output_1 = reflec_side_output_1 * reflec_side_output_1_mask
        edge_dict['reflec_edge_side_output1'] = reflec_side_output_1

        illum_side_output_1 = self.edge_side_output_1(edge_deconvs1_list[1])
        edge_dict['illum_edge_side_output1'] = illum_side_output_1

        reflec_side_output_2 = self.edge_side_output_2(edge_deconvs2_list[0])
        reflec_side_output_2 = torch.clamp(reflec_side_output_2, 0, 1)
        reflec_side_output_2_m = torch.mean(reflec_side_output_2, dim=1, keepdim=True)
        reflec_side_output_2_m = reflec_side_output_2_m / reflec_side_output_2_m.max()
        reflec_side_output_2_mask = torch.zeros_like(reflec_side_output_2_m)
        reflec_side_output_2_mask[reflec_side_output_2_m > 0.1] = 1
        reflec_side_output_2 = reflec_side_output_2 * reflec_side_output_2_mask
        edge_dict['reflec_edge_side_output2'] = reflec_side_output_2

        illum_side_output_2 = self.edge_side_output_2(edge_deconvs2_list[1])
        edge_dict['illum_edge_side_output2'] = illum_side_output_2

        reflec_edge_output = edge_output_list[0]
        reflec_edge_output = torch.clamp(reflec_edge_output, 0, 1)
        reflec_edge_output_m = torch.mean(reflec_edge_output, dim=1, keepdim=True)
        reflec_edge_output_m = reflec_edge_output_m / reflec_edge_output_m.max()
        reflec_edge_output_mask = torch.zeros_like(reflec_edge_output_m)
        reflec_edge_output_mask[reflec_edge_output_m > 0.1] = 1
        reflec_edge_output = reflec_edge_output * reflec_edge_output_mask

        edge_dict['reflect_edge_deconvs0_list'] = edge_deconvs0_list[0]
        edge_dict['illum_edge_deconvs0_list'] = edge_deconvs0_list[1]
        edge_dict['reflect_edge_deconvs1_list'] = edge_deconvs1_list[0]
        edge_dict['illum_edge_deconvs1_list'] = edge_deconvs1_list[1]
        edge_dict['reflect_edge_deconvs2_list'] = edge_deconvs2_list[0]
        edge_dict['illum_edge_deconvs2_list'] = edge_deconvs2_list[1]
        edge_dict['reflect_edge_deconvs3_list'] = edge_deconvs3_list[0]
        edge_dict['illum_edge_deconvs3_list'] = edge_deconvs3_list[1]
        edge_dict['reflect_edge_output'] = reflec_edge_output
        edge_dict['illum_edge_output'] = edge_output_list[1]

        return edge_dict

class VGGUnrefinedDecoderBatchNorm(nn.Module):

    def __init__(self):
        super(VGGUnrefinedDecoderBatchNorm, self).__init__()
        self.makeLinkedUnrefinedDecoder()
        self.attention = AttentionLayer()

    def makeLinkedUnrefinedDecoder(self):
        self.unrefined_deconvs0 = nn.ModuleList()
        for i in range(2):
            unrefined0 = nn.Sequential(
                nn.ConvTranspose2d(512 * 1, 512, 4, 2, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(True)
            )
            self.unrefined_deconvs0.append(unrefined0)

        self.unrefined_deconvs1 = nn.ModuleList()
        for i in range(2):
            unrefined1 = nn.Sequential(
                nn.ConvTranspose2d(512 * 3, 512, 4, 2, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(True)
            )
            self.unrefined_deconvs1.append(unrefined1)

        self.unrefined_deconvs2 = nn.ModuleList()
        for i in range(2):
            unrefined2 = nn.Sequential(
                nn.ConvTranspose2d((512 * 2) + (256 * 1), 256, 4, 2, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(True)
            )
            self.unrefined_deconvs2.append(unrefined2)

        self.unrefined_deconvs3 = nn.ModuleList()
        for i in range(2):
            unrefined3 = nn.Sequential(
                nn.ConvTranspose2d((256 * 2) + (128 * 1), 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(True)
            )
            self.unrefined_deconvs3.append(unrefined3)

        self.unrefined_output = nn.ModuleList()
        alb_output = nn.Sequential(
            nn.Conv2d((128 * 2) + (64 * 1), 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 3, 3, 1, 1),
            nn.BatchNorm2d(3),
            nn.ReLU(True)
        )
        self.unrefined_output.append(alb_output)

        shd_output = nn.Sequential(
            nn.Conv2d((128 * 2) + (64 * 1), 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(True)
        )
        self.unrefined_output.append(shd_output)

    def forward(self, encoderDict, edgeDict):
        unrefined_dict = {}

        unrefined_deconvs0_list = []
        store_unrefined_deconvs0 = [deconvs0(encoderDict['conv40']) for deconvs0 in
                              self.unrefined_deconvs0]
        attn_reflec_edge0 = self.attention(edgeDict['reflect_edge_deconvs0_list'],
                       store_unrefined_deconvs0[0])
        attn_illum_edge0 = self.attention(edgeDict['illum_edge_deconvs0_list'],
                       store_unrefined_deconvs0[1])
        unrefined_deconvs0_list.append(attn_reflec_edge0)
        unrefined_deconvs0_list.append(attn_illum_edge0)
        unrefined_deconvs0_list.append(encoderDict['conv30'])
        unrefined_deconvs0_comb = torch.cat(unrefined_deconvs0_list, 1)


        unrefined_deconvs1_list = []
        store_unrefined_deconvs1 = [deconvs1(unrefined_deconvs0_comb) for deconvs1 in
                              self.unrefined_deconvs1]
        attn_reflec_edge1 = self.attention(edgeDict['reflect_edge_deconvs1_list'],
                       store_unrefined_deconvs1[0])
        attn_illum_edge1 = self.attention(edgeDict['illum_edge_deconvs1_list'],
                       store_unrefined_deconvs1[1])
        unrefined_deconvs1_list.append(attn_reflec_edge1)
        unrefined_deconvs1_list.append(attn_illum_edge1)
        unrefined_deconvs1_list.append(encoderDict['conv20'])
        unrefined_deconvs1_comb = torch.cat(unrefined_deconvs1_list, 1)

        unrefined_deconvs2_list = []
        store_unrefined_deconvs2 = [deconvs2(unrefined_deconvs1_comb) for deconvs2 in
                              self.unrefined_deconvs2]
        attn_reflec_edge2 = self.attention(edgeDict['reflect_edge_deconvs2_list'],
                       store_unrefined_deconvs2[0])
        attn_illum_edge2 = self.attention(edgeDict['illum_edge_deconvs2_list'],
                       store_unrefined_deconvs2[1])
        unrefined_deconvs2_list.append(attn_reflec_edge2)
        unrefined_deconvs2_list.append(attn_illum_edge2)
        unrefined_deconvs2_list.append(encoderDict['conv10'])
        unrefined_deconvs2_comb = torch.cat(unrefined_deconvs2_list, 1)

        unrefined_deconvs3_list = []
        store_unrefined_deconvs3 = [deconvs3(unrefined_deconvs2_comb) for deconvs3 in
                              self.unrefined_deconvs3]
        attn_reflec_edge3 = self.attention(edgeDict['reflect_edge_deconvs3_list'],
                       store_unrefined_deconvs3[0])
        attn_illum_edge3 = self.attention(edgeDict['illum_edge_deconvs3_list'],
                       store_unrefined_deconvs3[1])
        unrefined_deconvs3_list.append(attn_reflec_edge3)
        unrefined_deconvs3_list.append(attn_illum_edge3)
        unrefined_deconvs3_list.append(encoderDict['conv00'])
        unrefined_deconvs3_comb = torch.cat(unrefined_deconvs3_list, 1)

        store_output = [output(unrefined_deconvs3_comb) for output in
                              self.unrefined_output]
        attn_unrefined_alb_output = self.attention(edgeDict['reflect_edge_output'],
                       store_output[0])
        attn_unrefined_shd_output = self.attention(torch.mean(edgeDict['illum_edge_output'], dim=1,
                                  keepdim=True),
                       store_output[1])

        unrefined_dict['alb_deconvs0_list'] = unrefined_deconvs0_list[0]
        unrefined_dict['alb_deconvs1_list'] = unrefined_deconvs1_list[0]
        unrefined_dict['alb_deconvs2_list'] = unrefined_deconvs2_list[0]
        unrefined_dict['alb_deconvs3_list'] = unrefined_deconvs3_list[0]

        unrefined_dict['shd_deconvs0_list'] = unrefined_deconvs0_list[1]
        unrefined_dict['shd_deconvs1_list'] = unrefined_deconvs1_list[1]
        unrefined_dict['shd_deconvs2_list'] = unrefined_deconvs2_list[1]
        unrefined_dict['shd_deconvs3_list'] = unrefined_deconvs3_list[1]

        unrefined_dict['alb_output_unrefined'] = attn_unrefined_alb_output
        unrefined_dict['shd_output_unrefined'] = attn_unrefined_shd_output

        return unrefined_dict

class VGGDecRefinerBatchNorm(nn.Module):

    def __init__(self):
        super(VGGDecRefinerBatchNorm, self).__init__()
        self.makeAttentionEncoder()
        self.makeLinkedDecoder()
        self.makeFeatureRecalibrator()
        self.attention = AttentionLayer()

    def makeAttentionEncoder(self):
        self.conv0 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(64 * 1, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(128 * 1, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(256 * 1, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(512 * 1, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

    def makeLinkedDecoder(self):
        self.linked_deconvs0 = nn.ModuleList()
        for i in range(2):
            deconvs0 = nn.Sequential(
                nn.ConvTranspose2d(512 * 1, 512, 4, 2, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(True)
            )
            self.linked_deconvs0.append(deconvs0)

        self.linked_deconvs1 = nn.ModuleList()
        for i in range(2):
            deconvs1 = nn.Sequential(
                nn.ConvTranspose2d(512 * 5, 512, 4, 2, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(True)
            )
            self.linked_deconvs1.append(deconvs1)

        self.linked_deconvs2 = nn.ModuleList()
        for i in range(2):
            deconvs2 = nn.Sequential(
                nn.ConvTranspose2d((512 * 2)+ (256 * 3), 256, 4, 2, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(True)
            )
            self.linked_deconvs2.append(deconvs2)

        self.linked_deconvs3 = nn.ModuleList()
        for i in range(2):
            deconvs3 = nn.Sequential(
                nn.ConvTranspose2d((256 * 2) + (128 * 3), 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(True)
            )
            self.linked_deconvs3.append(deconvs3)

        self.linked_output = nn.ModuleList()
        output = nn.Sequential(
            nn.Conv2d((128 * 2) + (64 * 3), 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 3, 3, 1, 1),
            nn.BatchNorm2d(3),
            nn.ReLU(True)
        )
        self.linked_output.append(output)

        shd_output = nn.Sequential(
            nn.Conv2d((128 * 2) + (64 * 3), 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(True)
        )
        self.linked_output.append(shd_output)

    def makeFeatureRecalibrator(self):
        self.alb_recalibrator = nn.Sequential(
            nn.Conv2d(6, 8, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )

        self.shd_recalibrator = nn.Sequential(
            nn.Conv2d(4, 8, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )

    def forward(self, x, x1, imgEncDict):
        intrinsic_dict = {}
        concat_reflec = torch.cat(x, 1)
        concat_shd = torch.cat(x1, 1)

        calibrated_reflec = self.alb_recalibrator(concat_reflec)
        calibrated_shd = self.shd_recalibrator(concat_shd)

        concat_edge_reflec = torch.cat([calibrated_reflec, calibrated_shd], 1)

        convs00 = self.conv0(concat_edge_reflec)
        convs10 = self.conv1(convs00)
        convs20 = self.conv2(convs10)
        convs30 = self.conv3(convs20)
        convs40 = self.conv4(convs30)

        deconvs0_list = [deconv0(convs40) for deconv0 in
                         self.linked_deconvs0]
        deconvs0_list.append(convs30)
        deconvs0_list.append(self.attention(imgEncDict['conv30'], convs30))
        deconvs0_list.append(self.attention(imgEncDict['cross30'], convs30))
        deconvs0_comb = torch.cat(deconvs0_list, 1)

        deconvs1_list = [deconv1(deconvs0_comb) for deconv1 in
                         self.linked_deconvs1]
        deconvs1_list.append(convs20)
        deconvs1_list.append(self.attention(imgEncDict['conv20'], convs20))
        deconvs1_list.append(self.attention(imgEncDict['cross20'], convs20))
        deconvs1_comb = torch.cat(deconvs1_list, 1)

        deconvs2_list = [deconv2(deconvs1_comb) for deconv2 in
                         self.linked_deconvs2]
        deconvs2_list.append(convs10)
        deconvs2_list.append(self.attention(imgEncDict['conv10'], convs10))
        deconvs2_list.append(self.attention(imgEncDict['cross10'], convs10))
        deconvs2_comb = torch.cat(deconvs2_list, 1)

        deconvs3_list = [deconv3(deconvs2_comb) for deconv3 in
                         self.linked_deconvs3]
        deconvs3_list.append(convs00)
        deconvs3_list.append(self.attention(imgEncDict['conv00'], convs00))
        deconvs3_list.append(self.attention(imgEncDict['cross00'], convs00))
        deconvs3_comb = torch.cat(deconvs3_list, 1)

        output_list = [output(deconvs3_comb) for output in
                       self.linked_output]

        intrinsic_dict['reflectance'] = output_list[0]
        intrinsic_dict['shading'] = output_list[1]

        return intrinsic_dict

class DecScaleClampedIllumEdgeGuidedNetworkBatchNorm(nn.Module):

    def __init__(self):
        super(DecScaleClampedIllumEdgeGuidedNetworkBatchNorm, self).__init__()
        self.imgEncoder = VGGEncoderBatchNorm()
        self.edgeDecoder = VGGScaleClampEdgeDecoderBatchNorm()
        self.unrefinedDecoder = VGGUnrefinedDecoderBatchNorm()
        self.refinerNet = VGGDecRefinerBatchNorm()

    def forward(self, x):
        imgCCREncFeatures = self.imgEncoder(x)
        edgePrediction = self.edgeDecoder(imgCCREncFeatures)
        unrefinedIntrinsics = self.unrefinedDecoder(imgCCREncFeatures,
                                                        edgePrediction)
        x = [edgePrediction['reflect_edge_output'],
              unrefinedIntrinsics['alb_output_unrefined']]
        x1 = [edgePrediction['illum_edge_output'],
              unrefinedIntrinsics['shd_output_unrefined']]

        intrinsics = self.refinerNet(x, x1, imgCCREncFeatures)
        output_dict = {}

        output_dict['reflec_edge_64'] = edgePrediction['reflec_edge_side_output1']
        output_dict['reflec_edge_128'] = edgePrediction['reflec_edge_side_output2']
        output_dict['reflec_edge'] = edgePrediction['reflect_edge_output']

        output_dict['illum_edge_64'] = edgePrediction['illum_edge_side_output1']
        output_dict['illum_edge_128'] = edgePrediction['illum_edge_side_output2']
        output_dict['illum_edge'] = edgePrediction['illum_edge_output']

        output_dict['unrefined_reflec'] = unrefinedIntrinsics['alb_output_unrefined']
        output_dict['unrefined_shd'] = unrefinedIntrinsics['shd_output_unrefined']

        output_dict['reflectance'] = intrinsics['reflectance']
        output_dict['shading'] = intrinsics['shading']

        output_dict['recon'] = torch.mul(output_dict['reflectance'],
                                         output_dict['shading'])
        output_dict['cross_img'] = imgCCREncFeatures['cross_img']

        return output_dict
