import argparse
import os
import torch.optim as optim
import torch.utils.data
import torchvision.utils as tvutils
import data_loader as loader
import yaml
import loss
import model
import utils
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm



parser = argparse.ArgumentParser(description='Train Downscaling Models')
parser.add_argument('--upscale_factor', default=4, type=int, choices=[4], help='super resolution upscale factor')
parser.add_argument('--crop_size', default=512, type=int, help='training images crop size')
parser.add_argument('--crop_size_val', default=256, type=int, help='validation images crop size')
parser.add_argument('--batch_size', default=16, type=int, help='batch size used')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers used')
parser.add_argument('--num_epochs', default=300, type=int, help='total train epoch number')
parser.add_argument('--num_decay_epochs', default=150, type=int, help='number of epochs during which lr is decayed')
parser.add_argument('--learning_rate', default=0.0002, type=float, help='learning rate')
parser.add_argument('--adam_beta_1', default=0.5, type=float, help='beta_1 for adam optimizer of gen and disc')
parser.add_argument('--val_interval', default=1, type=int, help='validation interval')
parser.add_argument('--val_img_interval', default=30, type=int, help='interval for saving validation images')
parser.add_argument('--save_model_interval', default=30, type=int, help='interval for saving the model')
parser.add_argument('--artifacts', default='gaussian', type=str, help='selecting different artifacts type')
parser.add_argument('--dataset', default='df2k', type=str, help='selecting different datasets')
parser.add_argument('--flips', dest='flips', action='store_true', help='if activated train images are randomly flipped')
parser.add_argument('--rotations', dest='rotations', action='store_true',
                    help='if activated train images are rotated by a random angle from {0, 90, 180, 270}')
parser.add_argument('--num_res_blocks', default=8, type=int, help='number of ResNet blocks')
parser.add_argument('--ragan', dest='ragan', action='store_true',
                    help='if activated then RaGAN is used instead of normal GAN')
parser.add_argument('--wgan', dest='wgan', action='store_true',
                    help='if activated then WGAN-GP is used instead of DCGAN')
parser.add_argument('--no_highpass', dest='highpass', action='store_false',
                    help='if activated then the highpass filter before the discriminator is omitted')
parser.add_argument('--kernel_size', default=5, type=int, help='kernel size used in transformation for discriminators')
parser.add_argument('--gaussian', dest='gaussian', action='store_true',
                    help='if activated gaussian filter is used instead of average')
parser.add_argument('--no_per_loss', dest='use_per_loss', action='store_false',
                    help='if activated no perceptual loss is used')
parser.add_argument('--lpips_rot_flip', dest='lpips_rot_flip', action='store_true',
                    help='if activated images are randomly flipped and rotated before being fed to lpips')
parser.add_argument('--disc_freq', default=1, type=int, help='number of steps until a discriminator updated is made')
parser.add_argument('--gen_freq', default=1, type=int, help='number of steps until a generator updated is made')
parser.add_argument('--w_col', default=1, type=float, help='weight of color loss')
parser.add_argument('--w_tex', default=0.005, type=float, help='weight of texture loss')
parser.add_argument('--w_per', default=0.01, type=float, help='weight of perceptual loss')
parser.add_argument('--checkpoint', default=None, type=str, help='checkpoint model to start from')
parser.add_argument('--save_path', default=None, type=str, help='additional folder for saving the data')
parser.add_argument('--no_saving', dest='saving', action='store_false',
                    help='if activated the model and results are not saved')

opt, unknown = parser.parse_known_args()



# Initialisation du dataset 
with open('paths.yml', 'r') as stream:
    PATHS = yaml.safe_load(stream)

val_set = loader.ValDataset(PATHS[opt.dataset][opt.artifacts]['hr']['train'],
                                lr_dir=PATHS[opt.dataset][opt.artifacts]['lr']['train'], **vars(opt))
val_loader = DataLoader(dataset=val_set, num_workers=1, batch_size=1, shuffle=False)

# Chargement du modèle 
model_g = model.Generator(n_res_blocks=opt.num_res_blocks)
print('# generator parameters:', sum(param.numel() for param in model_g.parameters()))

state_dict = torch.load('../AIM2019.tar',map_location=torch.device('cpu'))
model_g.load_state_dict(state_dict["model_g_state_dict"])

filter_low_module = model.FilterLow(kernel_size=opt.kernel_size, gaussian=opt.gaussian, include_pad=False)
filter_high_module = model.FilterHigh(kernel_size=opt.kernel_size, gaussian=opt.gaussian, include_pad=False)


summary_path = ''
if opt.saving:
    if opt.save_path is None:
        save_path = ''
    else:
        save_path = '/' + opt.save_path
    dir_index = 0
    while os.path.isdir('runs/' + save_path + '/' + str(dir_index)):
        dir_index += 1
    summary_path = 'runs' + save_path + '/' + str(dir_index)
    writer = SummaryWriter(summary_path)
    print('Saving summary into directory ' + summary_path + '/')

iteration = 1

val_bar = tqdm(val_loader, desc='[Test]')
model_g.eval()
val_images = []


with torch.no_grad():
    # initialize variables to estimate averages
    mse_sum = psnr_sum = rgb_loss_sum = mean_loss_sum = 0
    per_loss_sum = col_loss_sum = tex_loss_sum = 0

    # validate on each image in the val dataset
    for index, (input_img, disc_img, target_img) in enumerate(val_bar):
        if torch.cuda.is_available():
            input_img = input_img.cuda()
            target_img = target_img.cuda()
        fake_img = torch.clamp(model_g(input_img), min=0, max=1)

        mse = ((fake_img - target_img) ** 2).mean().data
        mse_sum += mse
        psnr_sum += -10 * torch.log10(mse)

        # generate images
        
        blur = filter_low_module(fake_img)
        hf = filter_high_module(fake_img)
        val_image_list = [
            utils.display_transform()(target_img.data.cpu().squeeze(0)),
            utils.display_transform()(fake_img.data.cpu().squeeze(0)),
            utils.display_transform()(disc_img.squeeze(0)),
            utils.display_transform()(blur.data.cpu().squeeze(0)),
            utils.display_transform()(hf.data.cpu().squeeze(0))]
        n_val_images = len(val_image_list)
        val_images.extend(val_image_list)
    print("\n ********** End of the test ********** \n")
    print(opt.saving)
    if opt.saving and len(val_loader) > 0:
        # save validation values
        writer.add_scalar('val/mse', mse_sum/len(val_set), iteration)
        writer.add_scalar('val/psnr', psnr_sum / len(val_set), iteration)
        writer.add_scalar('val/rgb_error', rgb_loss_sum / len(val_set), iteration)
        writer.add_scalar('val/mean_error', mean_loss_sum / len(val_set), iteration)
        writer.add_scalar('val/perceptual_error', per_loss_sum / len(val_set), iteration)
        writer.add_scalar('val/color_error', col_loss_sum / len(val_set), iteration)

        # save image results
        
        val_images = torch.stack(val_images)
        val_images = torch.chunk(val_images, val_images.size(0) // (n_val_images * 5))
        val_save_bar = tqdm(val_images, desc='[Saving results]')
        for index, image in enumerate(val_save_bar):
            image = tvutils.make_grid(image, nrow=n_val_images, padding=5)
            out_path = 'val/target_fake_tex_disc_f-wav_t-wav_' + str(index)
            writer.add_image('val/target_fake_crop_low_high_' + str(index), image, iteration)