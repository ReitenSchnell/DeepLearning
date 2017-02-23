import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def build_gif(imgs):
    img_array = np.asarray(imgs)
    h, w, *c = imgs[0].shape
    interval = 0.1
    dpi = 72
    fig, ax = plt.subplots(figsize=(np.round(w / dpi), np.round(h / dpi)))
    fig.subplots_adjust(bottom=0)
    fig.subplots_adjust(top=1)
    fig.subplots_adjust(right=1)
    fig.subplots_adjust(left=0)
    ax.set_axis_off()
    axs = list(map(lambda x: [ax.imshow(x)], img_array))
    im_ani = animation.ArtistAnimation(fig, axs, interval=interval * 1000, repeat_delay=0, blit=False)
    save_to = 'animation.mp4'
    ff_writer = animation.FFMpegWriter()
    im_ani.save(save_to, writer=ff_writer, dpi=dpi)
    plt.show()