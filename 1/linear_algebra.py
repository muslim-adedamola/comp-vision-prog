import numpy as np


def dot_product(vector1, vector2):
    """ Implement dot product of the two vectors.
    Args:
        vector1: numpy array of shape (n, 1)
        vector2: numpy array of shape (n, 1)

    Returns:
        out: scalar value
    """
    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    v1 = vector1.copy()
    v2 = vector2.copy()
    
    out = np.dot(v1.T,v2)
    ######################################
    #        END OF YOUR CODE            #
    ######################################
    
    return out

def matrix_mult(M, vector1, vector2,vector3):
    """ Implement (vector2.T * vector3) * (M * vector1.T)
    Args:
        M: numpy matrix of shape (m, n)
        vector1: numpy array of shape (1, n)
        vector2: numpy array of shape (n, 1)
        vector3: numpy array of shape (n, 1)

    Returns:
        out: numpy matrix of shape (m, 1)
    """
    
    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    v3 = vector3.copy()
    v2 = vector2.copy()    
    v1 = vector1.copy()
    M = M.T
    
    #/////////////
    
    n1 = dot_product(v2, v3)
    n2 = dot_product(v1, M)
    
    out = n2*n1
    out = out.T
    ######################################
    #        END OF YOUR CODE            #
    ######################################

    return out

def svd(matrix):
    """ Implement Singular Value Decomposition
    Args:
        matrix: numpy matrix of shape (m, n)

    Returns:
        u: numpy array of shape (m, m)
        s: numpy array of shape (k)
        v: numpy array of shape (n, n)
    """
    
    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    u, s, v = np.linalg.svd(matrix)
    ######################################
    #        END OF YOUR CODE            #
    ######################################

    return u, s, v

def get_singular_values(matrix, n):
    """ Return top n singular values of matrix
    Args:
        matrix: numpy matrix of shape (m, w)
        n: number of singular values to output
        
    Returns:
        singular_values: array of shape (n)
    """
    singular_values = None
    u, s, v = svd(matrix)
    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    singular_values = s[:n]
    ######################################
    #        END OF YOUR CODE            #
    ######################################
    return singular_values

