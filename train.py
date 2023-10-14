import clip
import torch
import configargparse
import cv2
import clip.model
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import tqdm
import pandas as pd
import torchvision


plt.rcParams['axes.grid'] = False
plt.rcParams['figure.dpi'] = 600
plt.rcParams['figure.figsize'] = (8, 4)
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['savefig.format'] = 'pdf'
plt.rcParams['savefig.transparent'] = True
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0


def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='D:/work2/LLM_medicine/CLIP-main/data/image/', help='the path containing data')
    parser.add_argument('--image_type', type=str, default='.jpg')
    parser.add_argument('--expname', type=str, default='exp1')
    parser.add_argument('--txt_path', type=str, default='D:/work2/LLM_medicine/CLIP_MRI/data/label.txt',
                        help='the path to training label and image path text')
    parser.add_argument('--save_dir', type=str, default='ckpt/',
                        help='the path to save model and test output')
    parser.add_argument('--test_txt', type=str,
                        default='D:/work2/LLM_medicine/CLIP_MRI/data/test.txt', help='the path to test label text')

    # for vision
    parser.add_argument('--resolution', type=int,
                        default=512, help='image resolution')
    parser.add_argument('--visual_model', type=str, default='Resnet_50',
                        help='visual model name, you can choose ViT, Resnet_50 or Resnet_101')
    # these parameters are for VIT
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--width', type=int, default=768)
    parser.add_argument('--layers', type=int, default=12)
    parser.add_argument('--heads', type=int, default=12)

    # for transformer
    parser.add_argument('--context_length', type=int, default=100)
    parser.add_argument('--embed_dim', type=int, default=512)
    parser.add_argument('--transformer_name', type=str, default='BAAI/bge-large-zh-v1.5',
                        help='name of transformers and its tokenizer')
    parser.add_argument('--transformer_width',type=int,default=1024)

    # training options
    parser.add_argument('--use_pretrained', type=bool, default=True)
    parser.add_argument('--model_name', type=str,
                        default='ViT-B/32', help='pre-trained model name')
    parser.add_argument('--lr', type=float, default=5e-3,
                        help='learning rate')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', default=torch.device('cuda'))
    parser.add_argument('--epochs', type=int, default=40)

    parser.add_argument('--i_save', type=int, default=8,
                        help='frequency of model testing and saving')
    return parser


class dataset(Dataset):
    def __init__(self, args):
        super(dataset, self).__init__()
        self.txt_path = args.txt_path
        self.image_type = args.image_type
        self.data_dir = args.data_dir
        self.resolution = args.resolution
        self.device = args.device
        self.data = []
        with open(self.txt_path, 'r',encoding='utf-8') as f:
            self.data = f.readlines()

    def __getitem__(self, index):
        image_path, label = self.data[index].split('\t')
        image = cv2.imread(self.data_dir+image_path).astype('float32')
        image = cv2.resize(image, (self.resolution, self.resolution))
        image = image/255
        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image).to(self.device)
        return image, label

    def __len__(self):
        return len(self.data)

def calc_similarity(args,txt_path,model):
    data = []
    with open(txt_path, 'r',encoding='utf-8') as f:
        data = f.readlines()

    def preprocess(image_path: str):
        image = cv2.imread(args.data_dir+image_path).astype('float32')
        image = cv2.resize(image, (args.resolution, args.resolution))
        image = image/255
        image = np.transpose(image, (2, 0, 1))
        return image
    images = []
    text = []
    for line in data:
        info = line.split('\t')
        images.append(preprocess(info[0]))
        text.append(info[1])
    images = np.array(images)
    images = torch.tensor(images).to(args.device)
    with torch.no_grad():
        image_features = model.encode_image(images)
        text_features = model.encode_text(text)

        # normalized features
        image_features = image_features / \
            image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        similarity=torch.cosine_similarity(image_features,text_features)
        return float(similarity.mean())

def test(args, model):
    data = []
    with open(args.test_txt, 'r',encoding='utf-8') as f:
        data = f.readlines()

    def preprocess(image_path: str):
        image = cv2.imread(args.data_dir+image_path).astype('float32')
        image = cv2.resize(image, (args.resolution, args.resolution))
        image = image/255
        image = np.transpose(image, (2, 0, 1))
        return image
    images = []
    text = []
    for line in data:
        info = line.split('\t')
        images.append(preprocess(info[0]))
        text.append(info[1])
    origional_images = images.copy()
    images = np.array(images)
    images = torch.tensor(images).to(args.device)
    with torch.no_grad():
        _, logits_per_image = model(images, text)
    for i in range(len(text)):
        text[i] = text[i][:30]+'...'
    return logits_per_image.detach().cpu().numpy(), origional_images, text


def train():
    parser = config_parser()
    args = parser.parse_args()
    for i in vars(args):
        print(i+' : '+str(vars(args)[i]))
    model=clip.model.build_model(args)
    model.to(args.device)
    model.freeze_model()
    model.to(torch.float32)
    dataloader = DataLoader(
        dataset(args), batch_size=args.batch_size, shuffle=True)
    optimizer = optim.SGD(params=model.visual.parameters(), lr=args.lr)
    loss_x = torch.nn.CrossEntropyLoss()
    loss_y = torch.nn.CrossEntropyLoss()

    loss_all = []
    similarity_test=[]
    similarity_train=[]
    similarity_test.append(calc_similarity(args,args.test_txt,model))
    similarity_train.append(calc_similarity(args,args.txt_path,model))
    for epoch in tqdm.tqdm(range(args.epochs)):
        for index, (image, text) in enumerate(dataloader):
            optimizer.zero_grad()
            logits_per_image, logits_per_text = model(image, text)
            label = torch.arange(len(text)).to(args.device)
            loss_i = loss_x(logits_per_image, label)
            loss_t = loss_y(logits_per_text, label)
            loss = (loss_i+loss_t)/2
            loss.backward()
            optimizer.step()
            loss_all.append(float(loss.detach().cpu().numpy())
                            )
        # test and save model
        if (epoch+1) % args.i_save == 0:
            path = args.save_dir+'test'+str(epoch)+'/'
            if not os.path.exists(path):
                os.makedirs(path)
                os.makedirs(path+'output/')
            torch.save(model, os.path.join(path, 'model.pt'))
            similarity, original_images, text = test(args, model)
            plt.figure(figsize=(15, 7))
            plt.imshow(similarity)
            count = similarity.shape[0]
            for i, image in enumerate(original_images):
                plt.imshow(np.transpose(image, (1, 2, 0)), extent=(
                    i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")
            for x in range(similarity.shape[1]):
                for y in range(similarity.shape[0]):
                    plt.text(
                        x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=10)
            for side in ["left", "top", "right", "bottom"]:
                plt.gca().spines[side].set_visible(False)
            plt.xlim([-0.5, count - 0.5])
            plt.ylim([count + 0.5, -2])
            plt.yticks(range(similarity.shape[1]), text, fontsize=8)
            plt.xticks([])
            plt.savefig(path+'output/'+'output.pdf')
            similarity_test.append(calc_similarity(args,args.test_txt,model))
            similarity_train.append(calc_similarity(args,args.txt_path,model))
            pd.DataFrame({'loss': loss_all}).to_excel(path+'output/'+'loss.xlsx')
            pd.DataFrame({'similarity_test':similarity_test,'similarity_train':similarity_train}).to_excel(path+'output/'+'similarity.xlsx')
            with open(path+'output/'+'parameters.txt', 'w') as f:
                for i in vars(args):
                    f.write(i+':'+str(vars(args)[i])+'\n')



if __name__ == '__main__':
    train()
