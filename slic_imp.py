from PIL import Image, ImageDraw
import numpy as np
import slic

# This function obtain the percentage of pixels in a superpixel
# that are of the background class. Larger = More background.
# Smaller = more foreground. A very large or very small prediction means
# it is consistent with the superpixel.
# It assumes:
# 127 - pixel value in all channels for the background class
# 0 - pixel value in all channels for the superpixel boundaries
# 255 - pixel value in all chnnales for the foreground class
# It will return:
# ratio: The percentage of pixels in the superpixel nearest to the pos value
# that are of the background class. So, 100%, only background, 0% only foreground.
# supermask: A new arrangement of superpixels without any foreground on the
# analysed superpixel, so that one can look only for areas with foreground
# to calculate further ratios.
def obtain_percentage(supermask, superonly, pos):
    # Load as an image to use PIL
    # supermask = Superpixels overimposed on masks
    # superonly = superpixel arrangement only
    supermask = Image.fromarray(supermask, mode='RGB')
    superonly = Image.fromarray(superonly, mode='RGB')
    # Use the flood fill algorithm to identify a mask contoured by the
    # superpixels.
    pos = (int(pos[0]), int(pos[1])) # Fixing a strange colab error
    ImageDraw.floodfill(supermask, pos, (2, 127, 127), thresh=1)
    ImageDraw.floodfill(superonly, pos, (2, 255, 255), thresh=1)
    # Count the amount of pixels that are masked with and without
    # the prediction.
    supermask = np.array(supermask) # np.asarray will give a permission error
    superpixels2 = supermask[:, :, 0]
    superpixels2 = np.count_nonzero(superpixels2==2)
    superonly = np.asarray(superonly)
    superonly = np.count_nonzero(superonly[:, :, 0] == 2)
    # Turn the counted areas back to background so we can iterate later
    supermask[supermask == 2] = 127
    # The actual percentage. Temp = background only. Temp is always equal or
    # larger than superpixels.
    ratio = (superonly - superpixels2) / superonly
    return ratio, supermask

# analyze_image checks every prediction cluster in a mask (area where there is a prediction
# that is not separated by background or superpixel boundary) and returns a
# percentage of clusters that is consistent with the superpixels.
def analyze_image(superpixels, misk, thresh):
    # Transform tensor into numpy arrays
    superpixels = superpixels.numpy()
    misk = misk.numpy() # I make a copy here since there are permission errors otherwise
    # Remove batch dimensions
    superpixels = np.squeeze(superpixels, axis=0)
    mask = np.squeeze(misk, axis=0)
    mask = np.squeeze(mask, axis=2)
    # Transform into uint8
    mask[mask >= 0.5] = 1
    mask[mask < 0.5] = 0
    mask = np.uint8(mask*255)
    mask = np.repeat(mask[:,:,np.newaxis],3,axis=2)
    superpixels = np.uint8(superpixels*255)
    img = Image.fromarray(superpixels)
    # Create image for superpixel generator
    img.save("test.png")
    superpixels = slic.re_run()
    # Turn background class into 127
    superpixels[superpixels == 255] = 127
    # Temp = only superpixels and background
    temp = superpixels.copy()
    # Turn foreground class into 255
    superpixels[mask == 255] = 255
    # Turn boundaries into 0
    superpixels[temp == 0] = 0
    # Initialize loop counters
    checked = 0
    correct = 0
    for pixel in superpixels[:,:,0]:
        # We can just check areas where there is a prediction instead
        # of checking every pixel.
        place = np.where(superpixels[:, :, 0] == 255)
        # If we ever not find any predictions, then we finalize early.

        try:
            pos = (place[1][0],place[0][0])
        except IndexError:
            if checked != 0:
                return correct/checked
            else:
                return float(0)


        # Check what the ratio background-foreground is.
        [ratio, superpixels] = obtain_percentage(superpixels, temp, pos)
        # If the ratio is very small or very large, it's a good prediction.
        if ratio <= thresh or ratio >= (1-thresh):
            correct += 1

        # Keep track of how many prediction clusters we've checked.
        checked += 1
    return correct/checked


#superpixels = np.asarray(Image.open("SLICmasks/ISBI_000.png")) #superpixel grid
#mask = np.asarray(Image.open("data/label/ISBI_000.png")) #mask
#result = analyze_image(superpixels, mask, 0.1)
# Reulst: 86.7% of clusters are consistent with a 90% superpixel occupancy.
#print(result)