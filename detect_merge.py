from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet, cfg_re50
from data import cfg_mnet, cfg_re50, cfg_mnetv3, cfg_gnet, cfg_re152
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface_g import RetinaFace
from utils.box_utils import decode, decode_landm, nms
import time
import torch.nn.functional as F
# from pose import utils
import math

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='Retinaface')

parser.add_argument('-m', '--trained_model', default='./res152+/Resnet1526epoch_26.pth',  # ./model1/Resnet50_2FinalQ.pth  ./weights/mobilenet0.25_Final.pth
                    type=str, help='Trained state_dict file path to open')   # mobilev3\mobilev3_FinalQ.pth weights/mobilenet0.25_Final.pth  ./mobilev3/mobilenet0.25_FinalQ.pth
parser.add_argument('--network', default='resnet152+', help='Backbone network mobile0.25 & resnet50 & ghostnet & mobilev3')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', default=True, type=bool, help='show detection results')
parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
parser.add_argument('--image_path', default=r'./curve/0.jpg', type=str, help="image's path")
parser.add_argument('--output_path', default=r'./output/152.jpg', type=str, help='predict-visual')
args = parser.parse_args()
torch.set_grad_enabled(False)

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

if __name__ == '__main__':

    torch.set_grad_enabled(False)
    # cfg = None
    if args.network == "mobile0.25":
        from models.retinaface_m import RetinaFace
        cfg = cfg_mnet
    elif args.network == "resnet50":
        from models.retinaface_q import RetinaFace
        cfg = cfg_re50
    elif args.network == "ghostnet":
        from models.retinaface_g import RetinaFace
        cfg = cfg_gnet
    elif args.network == "mobilev3":
        from models.retinaface_g import RetinaFace
        cfg = cfg_mnetv3
    elif args.network == "resnet152":
        from models.retinaface_q import RetinaFace
        cfg = cfg_re152
    elif args.network == "resnet152+":
        from models.retinaface_q import RetinaFace
        cfg = cfg_re152

    # net and model
    net = RetinaFace(cfg=cfg, phase='test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    pr = time.time()
    # print(net)
    cudnn.benchmark = True
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).to(device)

    resize = 1

    # testing begin
    for i in range(1):
        image_path = args.image_path
        # print(image_path)
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)  # 读图像，cv2.IMREAD_COLOR：默认参数，读入一副彩色图片，忽略alpha通道
        img = np.float32(img_raw)
        # print(img)
        # 测试是原始图像尺寸，不是640*640尺寸
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])  #whwh 宽高宽高
        img -= (104, 117, 123)
        # print(img)
        # 扩展 归一化；而且依旧是bgr输入，前后一致
        # img /= (57, 57, 58)

        img = img.transpose(2, 0, 1)  # chw:通道数，高，宽
        img = torch.from_numpy(img).unsqueeze(0)  # torch.from_numpy：数组转变成张量  unsqueeze：增加维度（0表示，在第一个位置增加维度）
        img = img.to(device)
        scale = scale.to(device)

        tic = time.time()
        # loc, conf, landms, yaw, pitch, roll = net(img)  # forward pass  前向传播
        loc, conf, landms = net(img)
        print('net forward time: {:.4f}'.format(time.time() - tic))

        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        # decode 就相当于 匹配了！！！将anchor与预测框之间进行匹配  将bbox从基于anchors的情况下解码到在原图中的位置
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        # print("boxes:", boxes, "        11111111")
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        # 处理classification预测结果
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        # 将landmarks从基于anchors的情况下解码到在原图中的位置
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        # wh-> xy
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores   直接阈值过滤
        inds = np.where(scores > args.confidence_threshold)[0]  # confidence_threshold=0.02
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # yaw = yaw.squeeze(0)[inds]
        # pitch = pitch.squeeze(0)[inds]
        # roll = roll.squeeze(0)[inds]

        # keep top-K before NMS  需要进行排序，获取每个预测框的score 按照从大到小排序，应该是每一类！
        order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        # print(boxes)
        landms = landms[order]
        scores = scores[order]

        # yaw = yaw[order.tolist()]
        # pitch = pitch[order.tolist()]
        # roll = roll[order.tolist()]

        # do NMS 非极大值抑制
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        # print(dets)
        keep = py_cpu_nms(dets, args.nms_threshold)
        # keep = nms(dets, args.nms_threshold, force_cpu=args.cpu)
        # keep = nms(dets, args.nms_threshold, args.keep_top_k)
        dets = dets[keep, :]
        landms = landms[keep]

        # yaw = yaw[keep]
        # pitch = pitch[keep]
        # roll = roll[keep]

        # keep top-K faster NMS
        dets = dets[:args.keep_top_k, :]
        landms = landms[:args.keep_top_k, :]

        # yaw = yaw[:args.keep_top_k]
        # pitch = pitch[:args.keep_top_k]
        # roll = roll[:args.keep_top_k]

        # yaw = F.softmax(yaw, dim=-1)
        # pitch = F.softmax(pitch, dim=-1)
        # roll = F.softmax(roll, dim=-1)
        # yaw = torch.sum(yaw * idx_tensor, -1) * 3 - 99
        # pitch = torch.sum(pitch * idx_tensor, -1) * 3 - 99
        # roll = torch.sum(roll * idx_tensor, -1) * 3 - 99

        # print(pitch)

        # yaw = yaw.unsqueeze(-1).cpu().numpy()
        # pitch = pitch.unsqueeze(-1).cpu().numpy()
        # roll = roll.unsqueeze(-1).cpu().numpy()

        # print(pitch)
        # print(dets)
        # print("!!!!!!")
        # print(landms)

        # dets = np.concatenate((dets, landms, yaw, pitch, roll), axis=1)
        dets = np.concatenate((dets, landms), axis=1)

        # print(dets)

        # show image
        number = 0
        up = 0
        down = 0
        if args.save_image:
            for b in dets:

                if b[4] < args.vis_thres:
                    continue
                text = "{:.4f}".format(b[4])
                label = 0
                # text = ("第", i, "张脸", text)
                b = list(map(int, b))
                tdy = (b[1] + b[3]) / 2
                size = abs(b[3] - b[1]) / 2
                face_y = tdy - 0.50 * size
                nose_y = b[10]
                faceHigh = b[3] - b[1]
                dy = nose_y - face_y
                rate = dy / (faceHigh * 0.6) - 1
                # print(rate)
                Xpitch = -math.asin(rate) * 180 / np.pi
                # print("Xpitch", Xpitch)
                if Xpitch > 22:
                    up = up + 1
                    label = 0
                if Xpitch < 21:
                    down = down + 1
                    label = 1
                number = number + 1
                if label == 0:
                    cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)  # 人脸框 左上，右下 红
                else:
                    cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 233, 233), 2)  # 人脸框 左上，右下 黄
                # print(b[0], b[1], b[2], b[3], "!!!!!!!!!!!!!11")
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(img_raw, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                # landms
                cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 3)  # 左眼
                # print(b[5], b[6], "~~~~~~~~~~")
                cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 3)  # 右眼
                # print(b[7], b[8], "~~~~~~~~~~")
                cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 3)   # 鼻子
                # print(b[9], b[10], "~~~~~~~~~~")
                cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 3)  # 左嘴角
                # print(b[11], b[12], "~~~~~~~~~~")
                cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 3)  # 右嘴角
                # print(b[13], b[14], "~~~~~~~~~~")

                # print(b[15], b[16], b[17])

                # pose  yaw, pitch, roll,  b[15], b[16], b[17],
                # utils.draw_axis(img_raw, b[15], b[16], b[17], tdx=(b[0] + b[2]) / 2, tdy=(b[1] + b[3]) / 2, size=abs(b[3]-b[1]) / 2)
                # utils.plot_pose_cube(img_raw, b[15], b[16], b[17], tdx=(b[0] + b[2]) / 2, tdy=(b[1] + b[3]) / 2, size=abs(b[3]-b[1]) / 2)

            # print("一共", i, "张脸")
            print(args.output_path)
            print('学生数：', number)
            print('抬头率：{:.3f}'.format(up / number * 100), "%", " up:", up)
            print('低头率：{:.3f}'.format(down / number * 100), "%", " down:", down, )


            # save image
            name = args.output_path
            cv2.imwrite(name, img_raw)
            print('run time: {:.4f}'.format(time.time() - pr))
            # print("  yaw, pitch, roll", yaw, pitch, roll)

