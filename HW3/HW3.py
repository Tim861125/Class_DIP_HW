import numpy as np
import random
import math
import cv2

def Adaptive_Median_filter( img ):
    ret_img = img.copy()
    ret_img = cv2.copyMakeBorder( ret_img, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value = 0 )
    window_size = [ 3, 5, 7 ]                               # 3*3 5*5 7*7 的 Window Size
    for i in range( 3, img.shape[ 0 ] - 3  ):               
        for j in range( 3 , img.shape[ 1 ] - 3 ):               
            for w in window_size:                           # 在 Window 中跑
                ret_img[ i, j ] = img[ i, j ]
                f = math.floor( w / 2 )
                c = math.ceil( w / 2 )
                window_img = img[ i-f:i+c, j-f:j+c ]        # Window 範圍
                window_img = window_img.reshape( 1, -1 )
                s = np.sort( window_img[ 0 ] )              # 排序
                m = math.floor( ( w * w ) / 2 )
                min = s[ 0 ]                                # 最小
                med = s[ m ]                                # 中間
                max = s[ -1 ]                               # 最大
                if med <= min or med >= max:                
                    continue                                # 若不是 min < med < max 則去下一個 Window
                else:
                    if img[ i, j ] <= min or img[ i, j ] >= max:
                        ret_img[ i, j ] = med
                    break
    return ret_img[ 3:img.shape[ 0 ] - 3, 3:img.shape[ 1 ] - 3]

def Median_filter( img ):
    ret_img = img.copy()
    ret_img = cv2.copyMakeBorder( ret_img, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value = 0 )
    for i in range( 3, img.shape[ 0 ] - 3 ):
        for j in range( 3, img.shape[ 1 ] - 3 ):
            window_img = img[ i-1:i+2, j-1:j+2 ]
            window_img = window_img.reshape( 1, -1 )
            s = np.sort( window_img[ 0 ] )              # 排序
            ret_img[ i, j ] = s[ 4 ]                    # 取中間值
    return ret_img[ 3:img.shape[ 0 ] - 3, 3:img.shape[ 1 ] - 3]

def main():
    test_name = 'test.jpg'
    gray_img = cv2.imread( test_name, cv2.IMREAD_GRAYSCALE )
    noise_img = gray_img.copy()
    print( gray_img.shape )
    h, w = gray_img.shape
    noise_num = int( 0.5 * h * w )
    for index in range( noise_num ):
        x = random.randint( 0, h - 1 )
        y = random.randint( 0, w - 1 )
        if random.randint( 0, 1 ) == 0:
            noise_img[ x, y ] = 0
        else:
            noise_img[ x, y ] = 255

    a_img = noise_img.copy()
    m_img = noise_img.copy()

    final_M_img = Median_filter( m_img )            # 中值濾波器
    final_A_img = Adaptive_Median_filter( a_img )   # 適應性中值濾波器

    cv2.imshow( 'Gray', gray_img )
    cv2.imshow( 'Noise', noise_img )
    cv2.imshow( 'Median', final_M_img )
    cv2.imshow( 'Adaptive', final_A_img )
    cv2.imwrite( 'Gray.jpg', gray_img )
    cv2.imwrite( 'Noise.jpg', noise_img )
    cv2.imwrite( 'Median.jpg', final_M_img )
    cv2.imwrite( 'Adaptiven.jpg', final_A_img )

    cv2.waitKey() 
    cv2.destroyAllWindows()

main()