from os import listdir
import sys

import numpy as np

from matplotlib import pyplot as plt

from scipy import ndimage as ndi

from math import ceil, isclose, radians

from skimage import io
from skimage.exposure import equalize_hist, rescale_intensity, histogram, adjust_gamma
from skimage import feature
from skimage.transform import hough_circle, hough_circle_peaks, warp
from skimage import color
from skimage.draw import circle_perimeter
from skimage import segmentation
from skimage.morphology import watershed, disk, binary_erosion, binary_dilation, closing, opening
from skimage.filters import rank
from skimage.measure import grid_points_in_poly
from skimage import exposure
from skimage.measure import label, regionprops

from sklearn import cluster

def complementary(inImage):
    M = np.shape(inImage)[0]
    N = np.shape(inImage)[1]
    outImage = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            if (inImage[i][j] == 1):
                outImage[i][j] = 0
            else:
               outImage[i][j] = 1
    return outImage

def intersection(inImage1, inImage2):
    M = np.shape(inImage1)[0]
    N = np.shape(inImage1)[1]
    outImage = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            if ((inImage1[i][j] == 1) and (inImage2[i][j] == 1)):
                outImage[i][j] = 1
    return outImage

def histogram_of_zone(inImage, mask):
    M = np.shape(inImage)[0]
    N = np.shape(inImage)[1]
    for i in range(M):
        for j in range(N):
            inImage[i][j] = ceil(inImage[i][j] * 255)
    M = np.shape(inImage)[0]
    N = np.shape(inImage)[1]
    hist = np.zeros((256, ))
    for i in range(M):
        for j in range(N):
            if (mask[i][j] == 1):
                hist[int(inImage[i][j])] += 1
    return hist, range(256)

def prepare_for_kmeans(inImage, mask):
    M = np.shape(inImage)[0]
    N = np.shape(inImage)[1]
    outImage = np.zeros_like(inImage)
    for i in range(M):
        for j in range(N):
            if (mask[i][j] == 1):
                outImage[i][j] = inImage[i][j]
    return outImage

def _linear_polar_mapping(output_coords, k_angle, k_radius, center):
    """Inverse mapping function to convert from cartesion to polar coordinates
    Parameters
    ----------
    output_coords : ndarray
        `(M, 2)` array of `(col, row)` coordinates in the output image
    k_angle : float
        Scaling factor that relates the intended number of rows in the output
        image to angle: ``k_angle = nrows / (2 * np.pi)``
    k_radius : float
        Scaling factor that relates the radius of the circle bounding the
        area to be transformed to the intended number of columns in the output
        image: ``k_radius = ncols / radius``
    center : tuple (row, col)
        Coordinates that represent the center of the circle that bounds the
        area to be transformed in an input image.
    Returns
    -------
    coords : ndarray
        `(M, 2)` array of `(col, row)` coordinates in the input image that
        correspond to the `output_coords` given as input.
    """
    angle = output_coords[:, 1] / k_angle
    rr = ((output_coords[:, 0] / k_radius) * np.sin(angle)) + center[0]
    cc = ((output_coords[:, 0] / k_radius) * np.cos(angle)) + center[1]
    coords = np.column_stack((cc, rr))
    return coords


def _log_polar_mapping(output_coords, k_angle, k_radius, center):
    """Inverse mapping function to convert from cartesion to polar coordinates
    Parameters
    ----------
    output_coords : ndarray
        `(M, 2)` array of `(col, row)` coordinates in the output image
    k_angle : float
        Scaling factor that relates the intended number of rows in the output
        image to angle: ``k_angle = nrows / (2 * np.pi)``
    k_radius : float
        Scaling factor that relates the radius of the circle bounding the
        area to be transformed to the intended number of columns in the output
        image: ``k_radius = width / np.log(radius)``
    center : tuple (row, col)
        Coordinates that represent the center of the circle that bounds the
        area to be transformed in an input image.
    Returns
    -------
    coords : ndarray
        `(M, 2)` array of `(col, row)` coordinates in the input image that
        correspond to the `output_coords` given as input.
    """
    angle = output_coords[:, 1] / k_angle
    rr = ((np.exp(output_coords[:, 0] / k_radius)) * np.sin(angle)) + center[0]
    cc = ((np.exp(output_coords[:, 0] / k_radius)) * np.cos(angle)) + center[1]
    coords = np.column_stack((cc, rr))
    return coords

def warp_polar(image, center=None, *, radius=None, output_shape=None,
               scaling='linear', multichannel=False, **kwargs):
    """Remap image to polor or log-polar coordinates space.
    Parameters
    ----------
    image : ndarray
        Input image. Only 2-D arrays are accepted by default. If
        `multichannel=True`, 3-D arrays are accepted and the last axis is
        interpreted as multiple channels.
    center : tuple (row, col), optional
        Point in image that represents the center of the transformation (i.e.,
        the origin in cartesian space). Values can be of type `float`.
        If no value is given, the center is assumed to be the center point
        of the image.
    radius : float, optional
        Radius of the circle that bounds the area to be transformed.
    output_shape : tuple (row, col), optional
    scaling : {'linear', 'log'}, optional
        Specify whether the image warp is polar or log-polar. Defaults to
        'linear'.
    multichannel : bool, optional
        Whether the image is a 3-D array in which the third axis is to be
        interpreted as multiple channels. If set to `False` (default), only 2-D
        arrays are accepted.
    **kwargs : keyword arguments
        Passed to `transform.warp`.
    Returns
    -------
    warped : ndarray
        The polar or log-polar warped image.
    Examples
    --------
    Perform a basic polar warp on a grayscale image:
    >>> from skimage import data
    >>> from skimage.transform import warp_polar
    >>> image = data.checkerboard()
    >>> warped = warp_polar(image)
    Perform a log-polar warp on a grayscale image:
    >>> warped = warp_polar(image, scaling='log')
    Perform a log-polar warp on a grayscale image while specifying center,
    radius, and output shape:
    >>> warped = warp_polar(image, (100,100), radius=100,
    ...                     output_shape=image.shape, scaling='log')
    Perform a log-polar warp on a color image:
    >>> image = data.astronaut()
    >>> warped = warp_polar(image, scaling='log', multichannel=True)
    """
    if image.ndim != 2 and not multichannel:
        raise ValueError("Input array must be 2 dimensions "
                         "when `multichannel=False`,"
                         " got {}".format(image.ndim))

    if image.ndim != 3 and multichannel:
        raise ValueError("Input array must be 3 dimensions "
                         "when `multichannel=True`,"
                         " got {}".format(image.ndim))

    if center is None:
        center = (np.array(image.shape)[:2] / 2) - 0.5

    if radius is None:
        w, h = np.array(image.shape)[:2] / 2
        radius = np.sqrt(w ** 2 + h ** 2)

    if output_shape is None:
        height = 360
        width = int(np.ceil(radius))
        output_shape = (height, width)
    else:
        #output_shape = safe_as_int(output_shape)
        height = output_shape[0]
        width = output_shape[1]

    if scaling == 'linear':
        k_radius = width / radius
        map_func = _linear_polar_mapping
    elif scaling == 'log':
        k_radius = width / np.log(radius)
        map_func = _log_polar_mapping
    else:
        raise ValueError("Scaling value must be in {'linear', 'log'}")

    k_angle = height / (2 * np.pi)
    warp_args = {'k_angle': k_angle, 'k_radius': k_radius, 'center': center}

    warped = warp(image, map_func, map_args=warp_args,
                  output_shape=output_shape, **kwargs)

    return warped

def pol2cart(rho, phi, center):
    x = rho * np.cos(radians(360 + 90 - phi)) + center[0]
    y = rho * np.sin(radians(360 + 90 - phi)) + center[1]
    return(int(x), int(y))

def distance(coord1, coord2):
    return np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)

def drawCircleFrom3Points(coord1, coord2, coord3):
    print(coord1, coord2, coord3)
    x, y, z = complex(coord1[1], coord1[0]), complex(coord2[1], coord2[0]), complex(coord3[1], coord3[0])
    w = z - x
    w /= y - x
    c = (x - y) * (w - abs(w)**2) / 2j / w.imag - x
    circy, circx = circle_perimeter(round(-c.real), round(-c.imag), round(abs(c + x)))
    newCircy = []
    newCircx = []
    for i in range(len(circy)):
        if not ((circy[i] < 0) or (circy[i] >= N) or (circx[i] < 0) or (circx[i] >= M)):
            newCircy.append(circy[i])
            newCircx.append(circx[i])
    plt.plot(np.asarray(newCircy), np.asarray(newCircx), color='red', marker='.', linestyle='None', markersize=1)

    return np.asarray(newCircy), np.asarray(newCircx)

def ransac_polyfit(x, y, order=3, n=20, k=100, t=0.1, d=100, f=0.8):
  # Thanks https://en.wikipedia.org/wiki/Random_sample_consensus
  
  # n – minimum number of data points required to fit the model
  # k – maximum number of iterations allowed in the algorithm
  # t – threshold value to determine when a data point fits a model
  # d – number of close data points required to assert that a model fits well to data
  # f – fraction of close data points required
  
  besterr = np.inf
  bestfit = None
  for kk in range(k):
    maybeinliers = np.random.randint(len(x), size=n)
    maybemodel = np.polyfit(x[maybeinliers], y[maybeinliers], order)
    alsoinliers = np.abs(np.polyval(maybemodel, x)-y) < t
    if sum(alsoinliers) > d and sum(alsoinliers) > len(x)*f:
      bettermodel = np.polyfit(x[alsoinliers], y[alsoinliers], order)
      thiserr = np.sum(np.abs(np.polyval(bettermodel, x[alsoinliers])-y[alsoinliers]))
      if thiserr < besterr:
        bestfit = bettermodel
        besterr = thiserr
  return bestfit

pathEyePictures = "ojos/"

#eyePictures = [f for f in listdir(pathEyePictures)]

#inImage = io.imread(pathEyePictures + eyePictures[0], as_gray=True)

inImage = io.imread(pathEyePictures + sys.argv[1], as_gray=True)

plt.figure(1)
plt.imshow(inImage, cmap='gray')

M = np.shape(inImage)[0]
N = np.shape(inImage)[1]

#<Transformada de Hough para el iris>
canny = feature.canny(inImage, sigma=1.0)

houghRadiuses = np.arange(40, 70, 1)
houghCircle = hough_circle(canny, houghRadiuses)

accums, cx, cy, radiuses = hough_circle_peaks(houghCircle, houghRadiuses, total_num_peaks=1)
irisDiameter = radiuses[0] * 2

polarOrigin2 = (cy, cx) #Almacena las coordenadas del centro del iris para usar como dentro del espacio polar para la detección real del iris

outImage = color.gray2rgb(inImage)
irisMask = np.zeros((M, N), dtype=int)
circy, circx = circle_perimeter(cy[0], cx[0], radiuses[0])
irisMinRow = circy.min()
irisMinColumn = circx.min()
outImage[circy, circx] = (220, 20, 20)
irisMask[circy, circx] = 1
#</Transformada de Hough para el iris>

#<Transformada de Hough para la pupila>
canny = feature.canny(inImage, sigma=1.0)

houghRadiuses = np.arange(20, 40, 1)
houghCircle = hough_circle(canny, houghRadiuses)

accums, cx, cy, radiuses = hough_circle_peaks(houghCircle, houghRadiuses, total_num_peaks=5)

#De las tres pupilas candidatas nos quedamos con la más votada cuyo centro esté en la zona central del iris
circle_index = -1
for center_y, center_x, radius in zip(cy, cx, radiuses):
    circle_index += 1
    if (center_y > (irisMinRow + (1 / 3) * irisDiameter) and center_y < (irisMinRow + (2 / 3) * irisDiameter) and center_x > (irisMinColumn + (1 / 3) * irisDiameter) and center_x < (irisMinColumn + (2 / 3) * irisDiameter)):
        break

polarOrigin = (cy[circle_index], cx[circle_index]) #Almacena las coordenadas del centro de la pupila para usar como dentro del espacio polar para la detección de la esclerótica más tarde

pupilMask = np.zeros((M, N), dtype=int)
circy, circx = circle_perimeter(cy[circle_index], cx[circle_index], radiuses[circle_index])
outImage[circy, circx] = (220, 20, 20)
pupilMask[circy, circx] = 1
#</Transformada de Hough para la pupila>
    
pupilMask = ndi.binary_fill_holes(pupilMask)
irisMask = ndi.binary_fill_holes(irisMask)

irisMask = intersection(irisMask, complementary(pupilMask))

#<Parte de KMeans>
pupilForKMeans2d = prepare_for_kmeans(inImage, pupilMask)
pupilForKMeans = color.gray2rgb(pupilForKMeans2d)

plt.figure(2)
plt.imshow(pupilForKMeans)

x, y, z = pupilForKMeans.shape
pupilForKMeans2d = pupilForKMeans.reshape(x*y, z)

kMeansCluster = cluster.KMeans(n_clusters=3)
kMeansCluster.fit(pupilForKMeans2d)
clusterCenters = kMeansCluster.cluster_centers_
clusterLabels = kMeansCluster.labels_

plt.figure(3)
pupilInClusters = clusterCenters[clusterLabels].reshape(x, y, z)
pupilInClusters = color.rgb2gray(pupilInClusters)
plt.imshow(pupilInClusters, cmap='gray')
#</Parte de KMeans>

#<Hacer que el contorno del brillo aparezca en la imagen original>
clusterCentersMax = clusterCenters.max()
shinePositions = np.asarray([[i, j] for (i, j), intensity in np.ndenumerate(pupilInClusters) if isclose(intensity, clusterCentersMax, rel_tol=1e-5)])
print(clusterCentersMax)
shineMask = np.zeros_like(inImage)
for i in range(len(shinePositions)):
    shineMask[shinePositions[i, 0]][shinePositions[i, 1]] = 1
plt.figure(4)
plt.imshow(shineMask, cmap='gray')
plt.figure(5)
plt.imshow(segmentation.mark_boundaries(outImage, shineMask.astype(np.int8)), cmap='gray')
#</Hacer que el contorno del brillo aparezca en la imagen original>



#<Segmentar el iris sin el párpado>
finalIrisMask = irisMask
finalIris = np.zeros_like(inImage)

irisFromInImage = np.zeros_like(inImage)
for i in range(M):
    for j in range(N):
        if (irisMask[i, j] == 1):
            irisFromInImage[i, j] = inImage[i, j]

plt.figure(6)
plt.imshow(irisFromInImage, cmap='gray')

irisFromInImageClosed = closing(irisFromInImage, disk(5))

plt.figure(7)
plt.imshow(irisFromInImageClosed, cmap='gray')

c = feature.canny(irisFromInImageClosed, low_threshold=0.1, high_threshold=0.2)

dilatePupilMask = binary_dilation(pupilMask, disk(2))
for i in range(M):
    for j in range(N):
        if ((c[i, j] == 1) and (dilatePupilMask[i, j] == 1)):
            c[i, j] = 0

polar_image = warp_polar(c, polarOrigin2)

supContour = []
supContour.append([0, 0])
for angle in range(215, 325):
    for r in range(int(irisDiameter / 2) - 2):
        x, y = pol2cart(r, angle, polarOrigin2)
        if (c[x, y] == 1):
            supContour.append([x, y])
            break
    

plt.figure(8)
plt.imshow(c, cmap='gray')
supContourArr = np.asarray(supContour)
plt.plot(supContourArr[:, 1], supContourArr[:, 0], color='cyan', marker='o', linestyle='None', markersize=2)

model = None
if (len(supContourArr[:, 0]) > 30):
    model = ransac_polyfit(supContourArr[:, 1], supContourArr[:, 0], order = 2, n=4, d=0.5*len(supContourArr[:, 0]), t=2., f=0.5, k=300)

if model is not None:
    p = np.poly1d(model)

    line_x = np.arange(0, N)
    line_y = p(line_x)

    plt.plot(line_x, line_y, '-b')
    plt.xlim(0,  N - 1)
    plt.ylim(M - 1, 0)
    
    verts = np.concatenate((np.asarray([[M - 1, 0]]), np.column_stack((line_y, line_x))))
    verts = np.concatenate((verts, np.asarray([[M - 1, N - 1]])))
    supModelMask = grid_points_in_poly((M, N), verts)
    finalIrisMask = intersection(finalIrisMask, supModelMask)
    
    
infContour = []
infContour.append([0, 0])
for angle in range(45, 135):
    firstLocated = False
    for r in range(int(irisDiameter / 2) - 2):
        x, y = pol2cart(r, angle, polarOrigin2)
        if (c[x, y] == 1):
            infContour.append([x, y])
            break
    
plt.figure(9)
plt.imshow(irisFromInImage, cmap='gray')
infContourArr = np.asarray(infContour)
plt.plot(infContourArr[:, 1], infContourArr[:, 0], color='yellow', marker='o', linestyle='None', markersize=2)

model = None
if (len(infContourArr[:, 0]) > 30):
    model = ransac_polyfit(infContourArr[:, 1], infContourArr[:, 0], order = 2, n=4, d=0.5*len(infContourArr[:, 0]), t=2., f=0.5, k=300)

if model is not None:
    p = np.poly1d(model)

    line_x = np.arange(0, N)
    line_y = p(line_x)

    plt.plot(line_x, line_y, '-b')
    plt.xlim(0,  N - 1)
    plt.ylim(M - 1, 0)
    
    verts = np.concatenate((np.asarray([[0, 0]]), np.column_stack((line_y, line_x))))
    verts = np.concatenate((verts, np.asarray([[0, N - 1]])))
    infModelMask = grid_points_in_poly((M, N), verts)
    finalIrisMask = intersection(finalIrisMask, infModelMask)
    
for i in range(M):
    for j in range(N):
        if (finalIrisMask[i, j] == 1):
            finalIris[i, j] = inImage[i, j]
            
plt.imshow(segmentation.mark_boundaries(irisFromInImage, finalIrisMask.astype(np.int8)), cmap='gray')
#</Segmentar el iris sin el párpado>

"""#<Segmentar pestañas en el iris final>
finalIrisForKMeans = color.gray2rgb(finalIris)

x, y, z = finalIrisForKMeans.shape
finalIrisForKMeans2d = finalIrisForKMeans.reshape(x*y, z)

kMeansCluster = cluster.KMeans(n_clusters=4)
kMeansCluster.fit(finalIrisForKMeans2d)
clusterCenters = kMeansCluster.cluster_centers_
clusterLabels = kMeansCluster.labels_

plt.figure(3)
finalIrisInClusters = clusterCenters[clusterLabels].reshape(x, y, z)
finalIrisInClusters = color.rgb2gray(finalIrisInClusters)
plt.imshow(finalIrisInClusters, cmap='gray')

#clusterCentersSecondMax = np.sort(clusterCenters[:, 0])[1]
clusterCenters.max()
eyelashPositions = np.asarray([[i, j] for (i, j), intensity in np.ndenumerate(finalIrisInClusters) if isclose(intensity, clusterCentersMax, rel_tol=1e-5)])
#print(clusterCentersMax)
eyelashMask = np.zeros_like(inImage)
for i in range(len(eyelashPositions)):
    eyelashMask[eyelashPositions[i, 0]][eyelashPositions[i, 1]] = 1
plt.figure(4)
plt.imshow(eyelashMask, cmap='gray')
plt.figure(5)
plt.imshow(segmentation.mark_boundaries(finalIris, eyelashMask.astype(np.int8)), cmap='gray')
#</Segmentar pestañas en el iris final>"""




#<Hallar el modelo que ajusta el contorno inferior del ojo>

# Adaptive Equalization
img_adapteq = exposure.equalize_adapthist(inImage, clip_limit=0.03)

plt.figure(10)
plt.imshow(img_adapteq, cmap='gray')
    
c = feature.canny(img_adapteq, sigma=1, low_threshold=0.05, high_threshold=0.15)

plt.figure(11)
plt.imshow(c, cmap='gray')

labeled_c = label(c)
props = regionprops(labeled_c)

for prop in props:
    if ((prop.area < 50) or (prop.coords[:, 1].max() - prop.coords[:, 1].min() < 30)):
        labeled_c = np.where(labeled_c==prop.label, 0, labeled_c)
        for coord in prop.coords:
            c[coord[0], coord[1]] = 0

plt.figure(12)
plt.imshow(c, cmap='gray')

#Se suprimen bordes verticales
erode1_c = binary_erosion(c, np.asarray([[1], [1], [1]]))
erode2_c = binary_erosion(c, np.asarray([[0], [0], [1], [1], [1]]))
erode3_c = binary_erosion(c, np.asarray([[1], [1], [1], [0], [0]]))
c = intersection(c, complementary(erode1_c))
c = intersection(c, complementary(erode2_c))
c = intersection(c, complementary(erode3_c))

plt.figure(13)
plt.imshow(c, cmap='gray')

polar_image = warp_polar(c, polarOrigin)

plt.figure(14)
plt.imshow(polar_image, cmap='gray')

#Se buscan dobles lineas lanzando rayos desde el centro de la pupila (nos quedamos con el punto del medio) 
middlePoints = []
firsts = []
seconds = []
for angle in range(20, 160, 1):
    if ((angle > 70) and (angle < 100)):
        continue
    nextAngle = False
    for r in range(int(irisDiameter / 2) + 1, 100):
        x, y = pol2cart(r, angle, polarOrigin)
        if ((x > (M - 1)) or (y > (N - 1))):
            break
        if (c[x, y] == 1):
            first = [x, y]
            for r2 in range(r + 2, r + 8):
                x2, y2 = pol2cart(r2, angle, polarOrigin)
                if ((x2 > (M - 1)) or (y2 > (N - 1))):
                    nextAngle = True
                    break
                if (c[x2, y2] == 1):
                    second = [x2, y2]
                    firsts.append(first)
                    seconds.append(second)
                    fPlusS = [first[0] + second[0], first[1] + second[1]]
                    middlePoints.append([int(fPlusS[0] / 2), int(fPlusS[1] / 2)])
                    nextAngle = True
                    break
            if (nextAngle == True):
                break

plt.figure(15)
plt.imshow(c, cmap='gray')

middlePointsArr = np.asarray(middlePoints)
firstsArr = np.asarray(firsts)
secondsArr = np.asarray(seconds)

plt.plot(firstsArr[:, 1], firstsArr[:, 0], color='red', marker='o', linestyle='None', markersize=3)
plt.plot(secondsArr[:, 1], secondsArr[:, 0], color='red', marker='o', linestyle='None', markersize=3)
plt.plot(middlePointsArr[:, 1], middlePointsArr[:, 0], color='cyan', marker='o', linestyle='None', markersize=3)

plt.figure(16)
plt.imshow(inImage, cmap='gray')       
plt.plot(middlePointsArr[:, 1], middlePointsArr[:, 0], color='cyan', marker='o', linestyle='None', markersize=3) 

#<Quedarse con los 3 mejores puntos y hallar el círculo que pasa por ellos (parte de abajo)>
leftPoint = middlePoints[0]
for i in range(len(middlePoints)):
    if (middlePoints[i][1] < leftPoint[1]):
        leftPoint = middlePoints[i]

rightPoint = middlePoints[0]
for i in range(len(middlePoints)):
    if (middlePoints[i][1] > rightPoint[1]):
        rightPoint = middlePoints[i]
        
middlePointLR = [(leftPoint[0] + rightPoint[0]) / 2, (leftPoint[1] + rightPoint[1]) / 2]

middlePoint = middlePoints[0]
minDistance = 1000.0
for i in range(len(middlePoints)):
    actualDistance = distance(middlePoints[i], middlePointLR)
    if (actualDistance < minDistance):
        middlePoint = middlePoints[i]
        minDistance = actualDistance
        
plt.plot(leftPoint[1], leftPoint[0], color='blue', marker='+', linestyle='None', markersize=3)
plt.plot(rightPoint[1], rightPoint[0], color='blue', marker='+', linestyle='None', markersize=3)
plt.plot(middlePointLR[1], middlePointLR[0], color='blue', marker='+', linestyle='None', markersize=3)
plt.plot(middlePoint[1], middlePoint[0], color='blue', marker='+', linestyle='None', markersize=3)
circ_y, circ_x = drawCircleFrom3Points(leftPoint, middlePoint, rightPoint)
#</Quedarse con los 3 mejores puntos y hallar el círculo que pasa por ellos  (parte de abajo)>

#<Aproximar los puntos con ransac (parte de abajo)>
plt.figure(17)
plt.imshow(inImage, cmap='gray')       
plt.plot(middlePointsArr[:, 1], middlePointsArr[:, 0], color='cyan', marker='o', linestyle='None', markersize=3)

model = ransac_polyfit(middlePointsArr[:, 1], middlePointsArr[:, 0], order = 2, n=10, d=0.6*len(middlePointsArr[:, 0]), t=4., f=0.6, k=300)

infModelMask = np.zeros_like(inImage)
if model is not None:
    p = np.poly1d(model)
    
    line_x = np.arange(0, N)
    line_y = p(line_x)
    
    plt.plot(line_x, line_y, '-b')
    plt.xlim(0,  N - 1)
    plt.ylim(M - 1, 0)
    
    #infModelMask tiene 1s en las partes de la imagen que quedan por debajo del modelo
    verts = np.concatenate((np.asarray([[0, 0]]), np.column_stack((line_y, line_x))))
    verts = np.concatenate((verts, np.asarray([[0, N - 1]])))
    infModelMask = grid_points_in_poly((M, N), verts)
    
    print("\ncontorno inferior del ojo aproximado con ransac\n")
else:
    """#infModelMask tiene 1s en las partes de la imagen que quedan por debajo del modelo
    verts = np.concatenate((np.asarray([[0, 0]]), np.column_stack((circ_y, circ_x))))
    verts = np.concatenate((verts, np.asarray([[0, N - 1]])))
    infModelMask = grid_points_in_poly((M, N), verts)"""
    
    supModelMask[0, :] = 1
    supModelMask[circ_x, circ_y] = 1
    supModelMask = ndi.binary_fill_holes(supModelMask)
    
    print("\ncontorno inferior del ojo aproximado con el círculo que pasa por tres puntos\n")
#</Aproximar los puntos con ransac (parte de abajo)>

#</Hallar el modelo que ajusta el contorno inferior del ojo>




#<Hallar el modelo que ajusta el contorno superior del ojo>

c = opening(inImage, disk(4))

plt.figure(18)
plt.imshow(c, cmap='gray')

c = feature.canny(c, sigma=1, low_threshold=0.05, high_threshold=0.15)

plt.figure(19)
plt.imshow(c, cmap='gray')

labeled_c = label(c)
props = regionprops(labeled_c)

for prop in props:
    if ((prop.area < 50) or (prop.coords[:, 1].max() - prop.coords[:, 1].min() < 30)):
        labeled_c = np.where(labeled_c==prop.label, 0, labeled_c)
        for coord in prop.coords:
            c[coord[0], coord[1]] = 0

plt.figure(20)
plt.imshow(c, cmap='gray')

#Se suprimen bordes verticales
erode1_c = binary_erosion(c, np.asarray([[1], [1], [1]]))
erode2_c = binary_erosion(c, np.asarray([[0], [0], [1], [1], [1]]))
erode3_c = binary_erosion(c, np.asarray([[1], [1], [1], [0], [0]]))
#c = np.subtract(c, erode1_c, dtype=np.float32)
c = intersection(c, complementary(erode1_c))
c = intersection(c, complementary(erode2_c))
c = intersection(c, complementary(erode3_c))

plt.figure(21)
plt.imshow(c, cmap='gray')

#Se lanzan rayos hacia arriba y se queda con las primeras intersecciones
points = []
for angle in range(180, 360):
    if ((angle > 240) and (angle < 300)):
        continue
    for r in range(int(irisDiameter / 2) + 1, 100):
        x, y = pol2cart(r, angle, polarOrigin)
        if (c[x, y] == 1):
            points.append([x, y])
            break

pointsArr = np.asarray(points)

plt.figure(22)
plt.imshow(inImage, cmap='gray')       
plt.plot(pointsArr[:, 1], pointsArr[:, 0], color='yellow', marker='o', linestyle='None', markersize=3) 

#<Quedarse con los 3 mejores puntos y hallar el círculo que pasa por ellos (parte de arriba)>
leftPoint = points[0]
for i in range(len(points)):
    if (points[i][1] < leftPoint[1]):
        leftPoint = points[i]

rightPoint = points[0]
for i in range(len(points)):
    if (points[i][1] > rightPoint[1]):
        rightPoint = points[i]
        
pointLR = [(leftPoint[0] + rightPoint[0]) / 2, (leftPoint[1] + rightPoint[1]) / 2]

point = points[0]
minDistance = 1000.0
for i in range(len(points)):
    actualDistance = distance(points[i], pointLR)
    if (actualDistance < minDistance):
        point = points[i]
        minDistance = actualDistance
        
plt.plot(leftPoint[1], leftPoint[0], color='blue', marker='+', linestyle='None', markersize=3)
plt.plot(rightPoint[1], rightPoint[0], color='blue', marker='+', linestyle='None', markersize=3)
plt.plot(pointLR[1], pointLR[0], color='blue', marker='+', linestyle='None', markersize=3)
plt.plot(point[1], point[0], color='blue', marker='+', linestyle='None', markersize=3)
circ_y, circ_x = drawCircleFrom3Points(leftPoint, point, rightPoint)
#</Quedarse con los 3 mejores puntos y hallar el círculo que pasa por ellos  (parte de arriba)>

#<Aproximar los puntos con ransac (parte de arriba)>
plt.figure(23)
plt.imshow(inImage, cmap='gray')       
plt.plot(pointsArr[:, 1], pointsArr[:, 0], color='yellow', marker='o', linestyle='None', markersize=3)

model = ransac_polyfit(pointsArr[:, 1], pointsArr[:, 0], order = 2, n=10, d=0.6*len(pointsArr[:, 0]), t=3., f=0.6, k=300)

supModelMask = np.zeros_like(inImage)
if model is not None:
    p = np.poly1d(model)
    
    line_x = np.arange(0, N)
    line_y = p(line_x)
    
    plt.plot(line_x, line_y, '-b')
    plt.xlim(0,  N - 1)
    plt.ylim(M - 1, 0)
    
    #supModelMask tiene 1s en las partes de la imagen que quedan por encima del modelo
    verts = np.concatenate((np.asarray([[M - 1, 0]]), np.column_stack((line_y, line_x))))
    verts = np.concatenate((verts, np.asarray([[M - 1, N - 1]])))
    supModelMask = grid_points_in_poly((M, N), verts)

    print("\ncontorno superior del ojo aproximado con ransac\n")
else:
    """#infModelMask tiene 1s en las partes de la imagen que quedan por debajo del modelo
    verts = np.concatenate((np.asarray([[M - 1, 0]]), np.column_stack((circ_y, circ_x))))
    verts = np.concatenate((verts, np.asarray([[M - 1, N - 1]])))
    supModelMask = grid_points_in_poly((M, N), verts)"""
    
    supModelMask[M - 1, :] = 1
    supModelMask[circ_x, circ_y] = 1
    supModelMask = ndi.binary_fill_holes(supModelMask)
    
    print("\ncontorno superior del ojo aproximado con el círculo que pasa por tres puntos\n")
#</Aproximar los puntos con ransac (parte de arriba)>

#<Obtención de la máscara de la esclerótica>
eyeMask = intersection(infModelMask, supModelMask)
scleraMask = intersection(eyeMask, complementary(irisMask))
scleraMask = intersection(scleraMask, complementary(pupilMask))

plt.figure(24)
plt.imshow(segmentation.mark_boundaries(inImage, scleraMask.astype(np.int8)), cmap='gray')

scleraFromInImage = np.zeros_like(inImage)
for i in range(M):
    for j in range(N):
        if (scleraMask[i, j] == 1):
            scleraFromInImage[i, j] = inImage[i, j]
            
plt.figure(25)
plt.imshow(scleraFromInImage, cmap='gray')
#</Obtención de la máscara de la esclerótica>

plt.show()