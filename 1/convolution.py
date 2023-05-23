#credit: Juan Carlos Niebles and Ranjay Krishna

import numpy as np

def conv_naive(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    
    hk = Hk//2
    wk = Wk//2
    
    img = np.zeros(image.shape)
    
    for i in range(hk, Hi - hk):
        for j in range(wk, Wi - wk):
            sum = 0
            
            for m in range(Hk):
                for n in range(Wk):
                    sum += kernel[m][n] * image[i-hk+m][j-wk+n]
                    
            img[i][j] = sum
            
    out = img
    ######################################
    #        END OF YOUR CODE            #
    ######################################

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Example: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W)
        pad_width: width of the zero padding (left and right padding)
        pad_height: height of the zero padding (bottom and top padding)

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width)
    """

    H, W = image.shape
    out = None

    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    img = np.zeros((H+2*pad_height, W+2*pad_width))
    img[pad_height: H+pad_height, pad_width: W+pad_width] = image
    out = img
    ######################################
    #        END OF YOUR CODE            #
    ######################################
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    hk=Hk//2
    wk=Wk//2
    padded_img = zero_pad(image, hk, wk)
    
    
    Xk_flip=np.flip(kernel,1)
    Yk_flip=np.flip(Xk_flip,0)
    
    
    for x in range(Hi):
        for y in range(Wi):
                out[x,y]=np.sum(padded_img[x: x+Hk, y: y+Wk] * Yk_flip)
    ######################################
    #        END OF YOUR CODE            #
    ######################################

    return out

