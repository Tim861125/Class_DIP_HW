import matplotlib.pyplot as plt
import numpy as np
import sys
import cv2

def Otsu( img ):
    ret_img = img.copy()
    row, col = img.shape
    for i in range( row ):
        for j in range( col ):
            if img[ i, j ] < 127:
                ret_img[ i, j ] = 1
            else:
                ret_img[ i, j ] = 0
    return ret_img

def Dfs( ret_img, label, r, c ):
    count = 0
    row, col = ret_img.shape
    ret_img[ r, c ] = label
    for next_r in range( r - 1, r + 2 ):
        for next_c in range( c - 1, c + 2 ):
            if ret_img[ next_r, next_c ] == 1:
                count += 1
                count += Dfs( ret_img, label, next_r, next_c )
    return count

def Count_pixel( img ):
    ret_img = img.copy()
    row, col = ret_img.shape
    label = 255
    count_list = []
    for i in range( 1, row - 1 ):
        for j in range( 1, col - 1 ):
            if ret_img[ i, j ] == 1:
                count = Dfs( ret_img, label, i, j )
                label -= 1
                count_list.append( count + 1 )
                ret_img[ i, j ] = 0
    return ret_img[ 1:row-1, 1:col-1 ], count_list

def main():
    
    test_name = 'test.png'
    ori_img = cv2.imread( test_name, cv2.IMREAD_GRAYSCALE )
    ori_img = cv2.copyMakeBorder( ori_img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value = 0 )
    print( ori_img.shape )
    cv2.imshow( 'Ori', ori_img )
    sys.setrecursionlimit( ori_img.shape[ 0 ] * ori_img.shape[ 1 ] )

    binary_img = Otsu( ori_img )                    # 二值化
    final_img, count = Count_pixel( binary_img )    # 跑遞迴，計算幾個連通域以及分別像素

    print( 'Image has', len( count ), 'components' )
    print( 'Each pixel is' )
    for i in count:
        print( i )
    cv2.imshow( 'Final', final_img )
    cv2.waitKey()
    cv2.destroyAllWindows()

main()

