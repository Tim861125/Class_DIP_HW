import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

def Boundary( value ):  # 讓值在 0 ~ 255 
    ret_value = value.copy()
    if value < 0:
        ret_value = 0
    elif value > 255:
        ret_value = 255
    return ret_value

def Sobel( ori_img ):
    ret_sobel = np.zeros( ( ori_img.shape[ 0 ], ori_img.shape[ 1 ], 3 ), np.float64 )
    sobel_x = np.array( [ [ -1, -2, -1 ], [ 0, 0, 0 ], [ 1, 2, 1 ] ] )
    sobel_y = np.array( [ [ -1, 0, 1 ], [ -2, 0, 2 ], [ -1, 0, 1 ] ] )
    for i in range( 1, ori_img.shape[ 0 ] - 1 ):
        for j in range( 1, ori_img.shape[ 1 ] - 1 ):
            part_img = ori_img[i-1:i+2, j-1:j+2, :]
            for c in range( 3 ):
                value = abs( np.sum( part_img[ :, :, c] * sobel_x ) ) + abs( np.sum( part_img[ :, :, c] * sobel_y ) )
                value = Boundary( value )
                ret_sobel[ i, j, c ] = value
    return  ret_sobel

def convolution( ori_img, mask ):
    ret_img = np.zeros( ( ori_img.shape[ 0 ], ori_img.shape[ 1 ], 3 ), np.float64 )
    for i in range( 1, ori_img.shape[ 0 ] - 1 ):
        for j in range( 1, ori_img.shape[ 1 ] - 1 ):
            part_img = ori_img[i-1:i+2, j-1:j+2, :]
            for c in range( 3 ):
                value = np.sum( part_img[ :, :, c] * mask )
                value = Boundary( value )
                ret_img[ i, j, c ] = value
    return ret_img

def main():
    test_name = 'test.jpg'
    ori_img = cv2.imread( test_name )
    ori_img = cv2.copyMakeBorder( ori_img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value = 0 ) # Padding
    gray_img = cv2.cvtColor( ori_img, cv2.COLOR_BGR2GRAY )
    l_mask = np.array([ [ -1, -1, -1 ], [ -1, 8, -1 ], [ -1, -1, -1 ] ] )
    m_mask = np.array([ [ 1, 1, 1 ], [ 1, 1, 1 ], [ 1, 1, 1 ] ] )
    m_mask = m_mask / 9
    cv2.imshow( 'Ori', ori_img )

    sobel_img = Sobel( ori_img )                            # 一階微分
    s_img = convolution( sobel_img, m_mask )                # 模糊 (去雜訊)
    La_img = convolution( ori_img, l_mask )                 # 二階微分
    n_img = s_img / ( np.amax( s_img ) - np.amin( s_img ) ) # 正規化 0 ~ 1 之間
    multi_img = np.multiply( n_img, La_img )                # 一階微分模糊完乘上二階微分
    final_img = ori_img + multi_img                         # 最後再加上原圖

    cv2.imshow( 'Final', final_img.astype( np.uint8 ) )
    cv2.imwrite( 'Sobel.jpg', sobel_img )
    cv2.imwrite( 'Smooth.jpg', s_img )
    cv2.imwrite( 'Laplacian.jpg', La_img )
    cv2.imwrite( 'Multi.jpg', multi_img )
    cv2.imwrite( 'Final.jpg', final_img )
    cv2.waitKey()
    cv2.destroyAllWindows()

main()