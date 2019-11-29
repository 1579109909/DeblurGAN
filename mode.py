import os
import tensorflow as tf
from PIL import Image
import numpy as np
import time
import util


def train(args, model, sess, saver):
    if args.fine_tuning:
        saver.restore(sess, args.pre_trained_model)
        print("saved model is loaded for fine-tuning!")
        print("model path is %s" % args.pre_trained_model)

    num_imgs = len(os.listdir(args.train_Sharp_path))

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./logs', sess.graph)
    if args.test_with_train:
        f = open("valid_logs.txt", 'w')

    epoch = 0
    step = num_imgs // args.batch_size

    blur_imgs = util.image_loader(args.train_Blur_path, args.load_X, args.load_Y)
    sharp_imgs = util.image_loader(args.train_Sharp_path, args.load_X, args.load_Y)

    while epoch < args.max_epoch:
        random_index = np.random.permutation(len(blur_imgs))
        for k in range(step):
            s_time = time.time()
            blur_batch, sharp_batch = util.batch_gen(blur_imgs, sharp_imgs, args.patch_size,
                                                     args.batch_size, random_index, k)

            for t in range(args.critic_updates):
                _, D_loss = sess.run([model.D_train, model.D_loss],
                                     feed_dict={model.blur: blur_batch, model.sharp: sharp_batch, model.epoch: epoch})

            _, G_loss = sess.run([model.G_train, model.G_loss],
                                 feed_dict={model.blur: blur_batch, model.sharp: sharp_batch, model.epoch: epoch})

            e_time = time.time()

        if epoch % args.log_freq == 0:
            summary = sess.run(merged, feed_dict={model.blur: blur_batch, model.sharp: sharp_batch})
            train_writer.add_summary(summary, epoch)
            if args.test_with_train:
                test(args, model, sess, saver, f, epoch, loading=False)
            print("%d training epoch completed" % epoch)
            print("D_loss : {}, \t G_loss : {}".format(D_loss, G_loss))
            print("Elpased time : %0.4f" % (e_time - s_time))
            # print("D_loss : %0.4f, \t G_loss : %0.4f" % (D_loss, G_loss))
            # print("Elpased time : %0.4f" % (e_time - s_time))
        if (epoch) % args.model_save_freq == 0:
            saver.save(sess, './model/DeblurrGAN', global_step=epoch, write_meta_graph=False)

        epoch += 1

    saver.save(sess, './model/DeblurrGAN_last', write_meta_graph=False)

    if args.test_with_train:
        f.close()


def test(args, model, sess, saver, file, step=-1, loading=False):
    if loading:

        import re
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(args.pre_trained_model)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(args.pre_trained_model, ckpt_name))
            print(" [*] Success to read {}".format(ckpt_name))
        else:
            print(" [*] Failed to find a checkpoint")

    blur_img_name = sorted(os.listdir(args.test_Blur_path))
    sharp_img_name = sorted(os.listdir(args.test_Sharp_path))

    PSNR_list = []
    ssim_list = []

    blur_imgs = util.image_loader(args.test_Blur_path, args.load_X, args.load_Y, is_train=False)
    sharp_imgs = util.image_loader(args.test_Sharp_path, args.load_X, args.load_Y, is_train=False)

    if not os.path.exists('./result/'):
        os.makedirs('./result/')

    for i, ele in enumerate(blur_imgs):
        blur = np.expand_dims(ele, axis=0)
        sharp = np.expand_dims(sharp_imgs[i], axis=0)
        output, psnr, ssim = sess.run([model.output, model.PSNR, model.ssim],
                                      feed_dict={model.blur: blur, model.sharp: sharp})
        if args.save_test_result:
            output = Image.fromarray(output[0])
            split_name = blur_img_name[i].split('.')
            output.save(os.path.join(args.result_path, '%s_sharp.png' % (''.join(map(str, split_name[:-1])))))

        PSNR_list.append(psnr)
        ssim_list.append(ssim)

    length = len(PSNR_list)

    mean_PSNR = sum(PSNR_list) / length
    mean_ssim = sum(ssim_list) / length

    if step == -1:
        file.write('PSNR : {} SSIM : {}'.format(mean_PSNR, mean_ssim))
        file.close()

    else:
        file.write("{}d-epoch step PSNR : {} SSIM : {} \n".format(step, mean_PSNR, mean_ssim))

