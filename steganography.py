import cv2 as cv
import numpy as np
import random

"""
T = input
X = digitalized input
Y = encrypted info
K = Key matrix

Encription
KX = Y
Decryption
X = K^-1 Y
"""


def modified_gram_schmidt(A):
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for i in range(n):
        v = A[:, i]
        for j in range(i):
            R[j, i] = np.dot(Q[:, j], A[:, i])
            v = v - R[j, i] * Q[:, j]
        R[i, i] = np.linalg.norm(v)
        Q[:, i] = v / R[i, i]

    return Q, R


def is_invertible(A):
    Q, R = modified_gram_schmidt(A)
    return np.linalg.matrix_rank(A) == np.linalg.matrix_rank(R)


def SOR(A, b, omega, tol, max_iter):
    x = np.zeros_like(b)
    n = len(b)
    for i in range(max_iter):
        x_new = np.zeros(n)
        for j in range(n):
            s1 = np.dot(A[j, :j], x_new[:j])
            s2 = np.dot(A[j, j + 1:], x[j + 1:])
            x_new[j] = (b[j] - s1 - s2) / A[j, j]
            x_new[j] = omega * (x_new[j] - x[j]) + x[j]
        if np.linalg.norm(x - x_new) < tol:
            return x_new
        x = x_new
    return x


def richardson_preconditioned(A, b, M, tol, max_iter):
    x = np.zeros_like(b)
    for i in range(max_iter):
        r = b - np.dot(A, x)
        z = np.linalg.solve(M, r)
        alpha = np.dot(r, z) / np.dot(np.dot(A, z), z)
        x = x + alpha * z
        if np.linalg.norm(r) < tol:
            return x
    return x


def generate_key(n):
    encryption_key = np.random.randint(2, size=(n, n))
    while not is_invertible(encryption_key):
        encryption_key = np.random.randint(2, size=(n, n))
    return encryption_key


"""
    Unimplemented Idea:
        use above mentioned methods to encrypt current data, by multiplying it on generated key
        and then decrypt it. (for text we can use their orders to create matrix, then encrypt and save it into text
        again, as for image, you can encrypt every significant bits and then decrypt)
        
        Use gram-Schimdt to validate key's invertibility, use richardson or SOR to decrypt using iterative method.
"""


def convert_binary(text):
    if type(text) == str:
        return ''.join([format(ord(i), "08b") for i in text])
    elif type(text) == bytes or type(text) == np.ndarray:
        return [format(i, "08b") for i in text]
    elif type(text) == int or type(text) == np.uint8:
        return format(text, "08b")
    else:
        raise TypeError("Input type not supported")


def encrypt_text(img, text):
    max = img.shape[0] * img.shape[1] * 3 // 8
    # maximum bytes

    if len(text) > max:
        raise ValueError("Error encountered insufficient bytes, need bigger image or less data!!")

    text += '#####'
    index = 0

    bin_secret_msg = convert_binary(text)

    size = len(bin_secret_msg)
    for values in img:
        for pixels in values:
            r, g, b = convert_binary(pixels)
            if index < size:
                pixels[0] = int(r[:-1] + bin_secret_msg[index], 2)
                index += 1
            if index < size:
                pixels[1] = int(g[:-1] + bin_secret_msg[index], 2)
                index += 1
            if index < size:
                pixels[2] = int(b[:-1] + bin_secret_msg[index], 2)
                index += 1
            if index < size:
                break

    return img


def encrypt_img(img, toEncrypt):
    img[img.shape[0] - 1][img.shape[1] - 1][0] = toEncrypt.shape[0] / 255
    img[img.shape[0] - 1][img.shape[1] - 1][1] = toEncrypt.shape[1] / 255

    for i in range(toEncrypt.shape[0]):
        for j in range(toEncrypt.shape[1]):
            for l in range(3):
                v1 = format(img[i][j][l], '08b')
                v2 = format(toEncrypt[i][j][l], '08b')

                # Taking 4 MSBs of each image
                v3 = v1[:5] + v2[:3]

                img[i][j][l] = int(v3, 2)

    return img


def decrypt_text(img):
    # retrieve data
    data = ""
    for values in img:
        for pixels in values:
            r, g, b = convert_binary(pixels)
            data += r[-1]
            data += g[-1]
            data += b[-1]

    allBytes = [data[i: i + 8] for i in range(0, len(data), 8)]

    # converting from bits to characters
    decodedData = ""
    for bytes in allBytes:

        decodedData += chr(int(bytes, 2))

        if decodedData[-5:] == "#####":
            break

    return decodedData[:-5]


def decrypt_img(img):
    width = img.shape[0]
    height = img.shape[1]

    subImageWidth = img[img.shape[0] - 1][img.shape[1] - 1][0] * 255
    subImageHeight = img[img.shape[0] - 1][img.shape[1] - 1][1] * 255

    # img1 and img2 are two blank images
    img1 = np.zeros((width, height, 3), np.uint8)
    img2 = np.zeros((subImageWidth, subImageHeight, 3), np.uint8)

    for i in range(width):
        for j in range(height):
            for l in range(3):
                v1 = format(img[i][j][l], '08b')
                if i < subImageWidth and j < subImageHeight:
                    v2 = v1[:5] + chr(random.randint(0, 1) + 48) * 3
                    v3 = v1[5:] + chr(random.randint(0, 1) + 48) * 5

                    # Appending data to img1 and img2
                    img1[i][j][l] = int(v2, 2)
                    img2[i][j][l] = int(v3, 2)
                else:
                    img1[i][j][l] = int(v1, 2)

    return img1, img2
