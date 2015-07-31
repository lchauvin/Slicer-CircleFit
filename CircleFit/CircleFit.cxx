/*=========================================================================

  Program:      Circle Fit
  Language:     C++
  Contributors: Laurent Chauvin, Junichi Tokuda

  Copyright (c) Brigham and Women's Hospital. All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notices for more information.

  =========================================================================*/

#include <cmath>
#include <complex>

#include "itkListSample.h"
#include "itkCovarianceSampleFilter.h"
#include "itkSymmetricEigenAnalysis.h"
#include "itkAffineTransform.h"
#include "itkFixedCenterOfRotationAffineTransform.h"
#include "itkTransformFileWriter.h"
#include "itkPluginUtilities.h"
#include "CircleFitCLP.h"

typedef itk::Vector< double, 3 > VectorType;
typedef itk::Point< double, 3 > PointType;
typedef itk::Matrix< double, 3, 3 > MatrixType;
typedef itk::FixedArray< double, 3 > ArrayType;
typedef itk::AffineTransform< double, 3 > TransformType;
typedef itk::FixedCenterOfRotationAffineTransform< double, 3 > FixedCenterOfRotationTransformType;
typedef itk::TransformFileWriter TransformWriterType;
typedef std::vector< VectorType > PointListType;
typedef PointListType::iterator PointListIteratorType;
typedef itk::Statistics::ListSample< VectorType > ListSampleType;
typedef itk::Statistics::CovarianceSampleFilter< ListSampleType > CovarianceAlgorithmType;
typedef itk::SymmetricEigenAnalysis< CovarianceAlgorithmType::MatrixType, ArrayType, MatrixType > SymmetricEigenAnalysisType;


int    CalcIntersectionOfPerpendicularBisectors2D(VectorType& p1, VectorType& p2, VectorType& p3, VectorType& intersec, double radius);

void   FindCircleFromPoints(PointListType& movingPointList, TransformType::Pointer transform, double& radius);

void   FindCenter(PointListType& points, MatrixType& originalToPlaneMatrix, VectorType& center, double radius);

int    FindAndRemoveOutliers(PointListType& fixedPointList, MatrixType& rotationMatrix, VectorType& center, double radius);

int   RemoveBiggestOutlierFromList(PointListType& pointList, double outlyingScores[], unsigned int size);

double FindRotationAngle(MatrixType& originalToPlaneMatrix,
                         PointListType& movingPointList,
                         PointListType& fixedPointList,
			 VectorType principalVector, VectorType center,
			 double tuningStep, double tuningRange,
			 double& minAvgMinSqDist);

double EstimateRotationAngle(PointListType& inPlanePoints, PointListType& fixedPointList,
                             VectorType principalVector, VectorType center);

double FineTuneRotationAngle(PointListType& inPlanePoints, PointListType& fixedPointList,
                             VectorType principalVector, VectorType center, double estimatedAngle,
                             double tuningStep,  double tuningRange,
                             double& minAvgMinSqDist);

double AngleBetweenPoints(VectorType p1, VectorType p2, VectorType origin, VectorType principalVector);

void   TransformPoints(PointListType& inputPoints, PointListType& outputPoints,
		       TransformType::Pointer transform);

void   RotatePoints(PointListType& inputPoints,
                    VectorType principalVector, VectorType center, double angle,
                    PointListType& outputPoints);

double AverageMinimumSquareDistance(PointListType& set1, PointListType& set2);

int main( int argc, char * argv[] )
{
  
  PARSE_ARGS;
  
  //----------------------------------------
  // Convert points into PointListType

  PointListType movingPointList;
  size_t numberOfMovingPoints = movingPoints.size();
  for (size_t mp = 0; mp < numberOfMovingPoints; ++mp)
    {
    VectorType tmpMp;
    tmpMp[0] = movingPoints[mp][0];
    tmpMp[1] = movingPoints[mp][1];
    tmpMp[2] = movingPoints[mp][2];
    movingPointList.push_back(tmpMp);
    }

  PointListType fixedPointList;
  size_t numberOfFixedPoints = fixedPoints.size();
  for (size_t fp = 0; fp < numberOfFixedPoints; ++fp)
    {
    VectorType tmpFp;
    tmpFp[0] = fixedPoints[fp][0];
    tmpFp[1] = fixedPoints[fp][1];
    tmpFp[2] = fixedPoints[fp][2];
    fixedPointList.push_back(tmpFp);
    }

  //--------------------------------------------------------------------------------
  //
  // The registration process consists of the following steps:
  //
  //  1. Find the transform from the moving points to the origin (the circle is on the X-Y plane).
  //  2. Transform the moving points to the origin. The resulted point list is named 'originPoints'.
  //  3. Find the plane that fit the fixed points and estimate the center of the circle.
  //  4. Remove outliers, and recalculate PCA.
  //  5. Find the best fitting rotation angle between both sets of points.
  //  6. Compute the transform from the moving points to to the fixed points.
  //

  //----------------------------------------
  // 1. Find the transform from the moving points to the origin (the circle is on the X-Y plane).

  // Find transform from original model (a circle on the X-Y plane with the center at the origin) 
  double srcRadius;
  TransformType::Pointer originToMovingTransform = TransformType::New();
  FindCircleFromPoints(movingPointList, originToMovingTransform, srcRadius);

  std::cout << "originToMoving = " << originToMovingTransform << std::endl;
  std::cout << "srcRadius = " << srcRadius << std::endl;
  
  //----------------------------------------
  // 2. Transform the moving points to the origin. The resulted point list is named 'originPoints'.

  TransformType::Pointer movingToOriginTransform = TransformType::New();
  originToMovingTransform->GetInverse(movingToOriginTransform);

  PointListType originPointList;
  for (size_t i = 0; i < movingPointList.size(); ++i)
    {
    PointType p = movingToOriginTransform->TransformPoint(movingPointList[i]);
    originPointList.push_back(p.GetVectorFromOrigin());
    }

  //----------------------------------------
  // 3. Find the plane that fit the fixed points and estimate the center of the circle.

  int outlierFound = 0;
  MatrixType originToPlaneMatrix;
  VectorType axisVector;
  VectorType center;
  VectorType nx, ny, nz;

  do
    {
    // Convert vector points to ITK List
    ListSampleType::Pointer sampleList = ListSampleType::New();
    for (size_t i = 0; i < fixedPointList.size(); ++i)
      {
      sampleList->PushBack(fixedPointList[i]);
      }

    // Perform PCA
    CovarianceAlgorithmType::Pointer covarianceAlgorithm =
      CovarianceAlgorithmType::New();
    covarianceAlgorithm->SetInput( sampleList );
    covarianceAlgorithm->Update();
    
    // Perform Symmetric Eigen Analysis
    SymmetricEigenAnalysisType analysis ( 3 );
    ArrayType eigenValues;
    MatrixType eigenMatrix;
    analysis.SetOrderEigenMagnitudes( true );
    analysis.ComputeEigenValuesAndVectors( covarianceAlgorithm->GetCovarianceMatrix(),
					   eigenValues, eigenMatrix );

    // Extract the normal vector
    // The first eigen vector (with the minimal eigenvalue) is
    // the normal vector of the fitted plane.

    // NOTE: The third vector is generated by computing the cross product of
    // the first two vectors to ensure the right-hand coordinate system
    nx =  eigenMatrix[1];
    ny =  eigenMatrix[2];
    nz =  itk::CrossProduct(nx, ny);

    // Calculate matrix from original coordinate system to plane coordinate system
    for (int i = 0; i < 3; ++i)
      {
      originToPlaneMatrix[i][0] = nx[i];
      originToPlaneMatrix[i][1] = ny[i];
      originToPlaneMatrix[i][2] = nz[i];
      }
    axisVector = nz;

    // Estimate the center
    FindCenter(fixedPointList, originToPlaneMatrix, center, srcRadius);
    std::cout << "originToPlaneMatrix = " << originToPlaneMatrix << std::endl;

    //----------------------------------------
    // 4. Remove outliers.

    // After this line, bigger outliers will be removed from fixedPointList, eventually reducing number
    // of points in the list.
    // Then PCA is recalculated to get a more accurate center and plane orientation.

    outlierFound = FindAndRemoveOutliers(fixedPointList, originToPlaneMatrix, center, srcRadius);

    } while( outlierFound );

  //----------------------------------------
  // 5. Find the best fitting rotation angle between both sets of points.

  double minAvgMinSqDist = -1.0;
  double bestAngle = FindRotationAngle(originToPlaneMatrix,
                                       originPointList,
                                       fixedPointList,
                                       axisVector,
                                       center,
                                       M_PI*0.1/180.0, M_PI*2.0/180.0,
                                       minAvgMinSqDist);

  // Flip the fitted plane (about ny) and calculate the best rotation angle
  MatrixType flippedOriginToPlaneMatrix;
  for (int i = 0; i < 3; ++i)
    {
    flippedOriginToPlaneMatrix[i][0] = -nx[i];
    flippedOriginToPlaneMatrix[i][1] = ny[i];
    flippedOriginToPlaneMatrix[i][2] = -nz[i];
    }
  VectorType flippedAxisVector = -nz;

  // No need to estimate the center because it remains at the same point when the plane is flipped
  double flippedMinAvgMinSqDist = -1.0;
  double flippedBestAngle = FindRotationAngle(flippedOriginToPlaneMatrix,
                                              originPointList,
                                              fixedPointList,
                                              flippedAxisVector,
                                              center,
                                              M_PI*0.1/180.0, M_PI*2.0/180.0,
                                              flippedMinAvgMinSqDist);


  //----------------------------------------
  // 6. Compute the transform from the moving points to to the fixed points.

  TransformType::Pointer originToPlaneTransform = TransformType::New();
  if (minAvgMinSqDist < flippedMinAvgMinSqDist)
    {
    originToPlaneTransform->SetMatrix(originToPlaneMatrix);
    }
  else
    {
    originToPlaneTransform->SetMatrix(flippedOriginToPlaneMatrix);
    }
  originToPlaneTransform->SetOffset(center);

  TransformType::Pointer registrationTransform = TransformType::New();
  registrationTransform->SetCenter(center);

  if (minAvgMinSqDist < flippedMinAvgMinSqDist)
    {
    registrationTransform->Rotate3D(axisVector, bestAngle);
    }
  else
    {
    registrationTransform->Rotate3D(flippedAxisVector, flippedBestAngle);
    }

  registrationTransform->Compose(originToPlaneTransform, true);
  registrationTransform->Compose(movingToOriginTransform, true);

  // Output
  TransformWriterType::Pointer registrationTransformWriter = TransformWriterType::New();
  registrationTransformWriter->SetInput(registrationTransform->GetInverseTransform());
  registrationTransformWriter->SetFileName(registration);
  try
    {
    registrationTransformWriter->Update();
    }
  catch (itk::ExceptionObject &err)
    {
    std::cerr << err << std::endl;
    return EXIT_FAILURE;
    }

  return EXIT_SUCCESS;
}

//--------------------------------------------------------------------------------
// Return 0, if the data set may not give a good estimate.

int CalcIntersectionOfPerpendicularBisectors2D(VectorType& p1, VectorType& p2, VectorType& p3, VectorType& intersect, double radius)
{

  const double posErr = 5.0;

  // Compute the bisecting points between p1 and p2 (m1), and p2 and p3 (m2)
  VectorType m1 = (p1+p2)/2.0;
  VectorType m2 = (p2+p3)/2.0;

  // Compute the normal vectors along the perpendicular bisectors
  VectorType v12 = (p2-p1);
  VectorType v23 = (p3-p2);

  if (v12.GetNorm() < posErr || v23.GetNorm() < posErr)
    {
    return 0;
    }

  v12.Normalize();
  v23.Normalize();

  VectorType n1;
  n1[0] = -v12[1];
  n1[1] = v12[0];
  n1[2] = v12[2];

  VectorType n2;
  n2[0] = -v23[1];
  n2[1] = v23[0];
  n2[2] = v23[2];
  
  // Compute the projection of m2 onto the perpendicular bisector of p1p2
  VectorType h = m1 + ((m2-m1)*n1)*n1;

  // The intersecting point of the two perpendicular bisectors (= estimated
  // center of fitted circle) is 'c' can be written as:
  //
  //    <c> = <m2> + a * <n2>
  //
  // where 'a' is a scalar value. Projection of 'c' on the m2h is 'h'
  //
  //    a * <n2> * (<h> - <m2>)/(|<h> - <m2>|) = |<h> - <m2>|
  //    a = |<h> - <m2>|^2 / {<n2> * (<h> - <m2>)}
  //

  VectorType m2h = h-m2;
  VectorType::RealValueType a = m2h.GetSquaredNorm() / (n2 * m2h);
  
  intersect = m2 + a * n2;

  // Validate by radius, if radius is greater than 0
  if (radius > 0)
    {
    VectorType d1 = p1-intersect;
    VectorType d2 = p2-intersect;
    VectorType d3 = p3-intersect;
    
    if (fabs(d1.GetNorm()-radius) > posErr ||
        fabs(d2.GetNorm()-radius) > posErr ||
        fabs(d3.GetNorm()-radius) > posErr)
      {
      return 0;
      }
    }

  return 1;
}


//--------------------------------------------------------------------------------
// Estimate the center of the circle from the points on the arc.
// If 'radius' is given, the function will validate the center by comparing
// the distance between the center and each point and the given radius.

void FindCenter(PointListType& points, MatrixType& rotationMatrix, VectorType& center, double radius=0)
{

  //----------------------------------------
  // Transform the center point to the original coordinate system
  VectorType nx;
  VectorType ny;
  VectorType nz;

  for (int i = 0; i < 3; ++i) // TODO: check if this is correct
    {
    nx[i] = rotationMatrix[i][0];
    ny[i] = rotationMatrix[i][1];
    nz[i] = rotationMatrix[i][2];
    }

  //----------------------------------------
  // Calculate the average position of the all points.
  // This is used as the origin of the new coordinate system on the
  // fitted plane.
  VectorType meanPoint;
  for (size_t i = 0; i < points.size(); ++i)
    {
    meanPoint += points[i];
    }
  meanPoint /= (double)points.size();

  //----------------------------------------
  // Project all the points to the fitted plane.

  PointListType projectedPoints;
  for (size_t i = 0; i < points.size(); ++i)
    {
    VectorType p1;
    VectorType p2;
    p1 = points[i] - meanPoint;
    p2[0] = p1*nx;
    p2[1] = p1*ny;
    p2[2] = 0.0;
    projectedPoints.push_back(p2);
    }

  //----------------------------------------
  // Pick up every combination of three points from the list and calculate
  // the intersection of the perpendicular bisectors of the two chords connecting
  // the three points.

  VectorType meanIntersect;
  meanIntersect[0] = 0.0;
  meanIntersect[1] = 0.0;
  meanIntersect[2] = 0.0;

  int nPoints = 0;
  int nPointsUsed = 0;

  for (size_t i = 0; i < projectedPoints.size(); ++i)
    {
    for (size_t j = i+1; j < projectedPoints.size(); ++j)
      {
      for (size_t k = j+1; k < projectedPoints.size(); ++k)
        {
        VectorType p1 = projectedPoints[i];
        VectorType p2 = projectedPoints[j];
        VectorType p3 = projectedPoints[k];
        VectorType c;

        nPoints ++;
        if (CalcIntersectionOfPerpendicularBisectors2D(p1, p2, p3, c, radius) > 0)
          {
          meanIntersect = meanIntersect + c;
          nPointsUsed ++;
          }
        }
      }
    }

  meanIntersect /= (double)nPointsUsed;

  center = meanIntersect[0] * nx + meanIntersect[1] * ny + meanIntersect[2] * nz + meanPoint;

  std::cout << "Center = " << center << std::endl;  
  std::cout << std::endl;
}

//----------------------------------------
// Return 1 if outlier found and removed, 0 otherwise.

int FindAndRemoveOutliers(PointListType& fixedPointList, MatrixType& rotationMatrix, VectorType& center, double radius)
{
  // Get plane vectors
  VectorType nx;
  VectorType ny;
  VectorType nz;

  for (int i = 0; i < 3; ++i) // TODO: check if this is correct
    {
    nx[i] = rotationMatrix[i][0];
    ny[i] = rotationMatrix[i][1];
    nz[i] = rotationMatrix[i][2];
    }

  // Project points on the estimated plane
  PointListType projectedPoints;
  for (size_t i = 0; i < fixedPointList.size(); ++i)
    {
    VectorType p1;
    VectorType p2;
    p1 = fixedPointList[i] - center;
    p2[0] = p1*nx;
    p2[1] = p1*ny;
    p2[2] = 0.0;

    projectedPoints.push_back(p2);
    }

  // Define margin error, based on circle radius
  double outOfPlaneMargin = 2.0;
  double radiusMargin = .2;
  double radiusMax = radius*(1+radiusMargin);
  double radiusMin = radius*(1-radiusMargin);

  //----------------------------------------
  // First Pass
  // Find and remove points further than the radius from the estimated center

  std::cout << "----------------------------------------" << std::endl
	    << "Outlier Detection: 1st Pass" << std::endl
	    << "Find and remove points further than the radius from the estimated center" << std::endl
	    << std::endl;

  double scoreFirstPass[fixedPointList.size()];
  for (size_t i = 0; i < fixedPointList.size(); ++i)
    {
    scoreFirstPass[i] = 0.0;
    }

  // TODO: Should we use Norm or SquaredNorm to score points ?
  for (size_t i = 0; i < fixedPointList.size(); ++i)
    {
    double distanceToEstimatedCenter = (fixedPointList[i] - center).GetNorm();
    if ( distanceToEstimatedCenter > radiusMax )
      {
      scoreFirstPass[i] += distanceToEstimatedCenter;
      }
    }

  std::cout << "First Pass Results: " << std::endl;
  if ( RemoveBiggestOutlierFromList(fixedPointList, scoreFirstPass, fixedPointList.size()) )
    {
    std::cout << "[First Pass]: Outlier found and removed." << std::endl;
    return 1;
    }
  std::cout << "[First Pass]: No outlier found. Moving to second pass." << std::endl << std::endl;

  //----------------------------------------
  // Second Pass
  // Find and remove points out of the estimated plane

  std::cout << "----------------------------------------" << std::endl
	    << "Outlier Detection: 2nd Pass" << std::endl
	    << "Find and remove points out of the estimated plane" << std::endl
	    << std::endl;

  double scoreSecondPass[fixedPointList.size()];
  for (size_t i = 0; i < fixedPointList.size(); ++i)
    {
    scoreSecondPass[i] = 0.0;
    }

  // Calculate distance from original points to projected points.
  // If original point is in the estimated plan, distance should be close to 0.
  // Score points with the rounded value of their distance to their projection
  // on the estimated plane.
  for (size_t i = 0; i < fixedPointList.size(); ++i)
    {
    // Get projected points in x,y,z coordinate system
    VectorType unprojected;
    unprojected = fixedPointList[i] - center;

    VectorType projected;
    projected[0] = projectedPoints[i][0]*nx[0] + projectedPoints[i][1]*ny[0];
    projected[1] = projectedPoints[i][0]*nx[1] + projectedPoints[i][1]*ny[1];
    projected[2] = projectedPoints[i][0]*nx[2] + projectedPoints[i][1]*ny[2];

    // TODO: Should we use Norm or SquaredNorm to score points ?
    double distanceToProjection = (unprojected - projected).GetNorm();

    // Do not increase point score if distance is less than a millimeter off the plane
    // because fiducial detection could have some variability.
    // If detected point has some error in nz direction, it could be considered as outlier
    // if all other points are in the plane, which is untrue. Allowing a margin error
    // allow us to overcome this issue.
    if ( distanceToProjection > outOfPlaneMargin )
      {
      scoreSecondPass[i] += distanceToProjection;
      }
    }

  std::cout << "Second Pass Results: " << std::endl;
  if ( RemoveBiggestOutlierFromList(fixedPointList, scoreSecondPass, fixedPointList.size()) )
    {
    std::cout << "[Second Pass]: Outlier found and removed." << std::endl;
    return 1;
    }
  std::cout << "[Second Pass]: No outlier found. Moving to third pass." << std::endl << std::endl;

  //----------------------------------------
  // Third Pass
  // Find and remove points not fitting the circle

  std::cout << "----------------------------------------" << std::endl
	    << "Outlier Detection: 3rd Pass" << std::endl
	    << "Find and remove points not fitting the circle" << std::endl
	    << std::endl;

  // Calculate points' outlying score
  double scoreThirdPass[projectedPoints.size()];
  for (size_t i = 0; i < projectedPoints.size(); ++i)
    {
    scoreThirdPass[i] = 0.0;
    }

  for (size_t i = 0; i < projectedPoints.size(); ++i)
    {
    for (size_t j = i+1; j < projectedPoints.size(); ++j)
      {
      for (size_t k = j+1; k < projectedPoints.size(); ++k)
	{
	VectorType intersection;
	CalcIntersectionOfPerpendicularBisectors2D(projectedPoints[i],
						   projectedPoints[j],
						   projectedPoints[k],
						   intersection,
						   0);

	double firstPointDistance = (projectedPoints[i] - intersection).GetNorm();
	double secondPointDistance = (projectedPoints[j] - intersection).GetNorm();
	double thirdPointDistance = (projectedPoints[k] - intersection).GetNorm();
	if ( (firstPointDistance > radiusMax || firstPointDistance < radiusMin) ||
	     (secondPointDistance > radiusMax || secondPointDistance < radiusMin) ||
	     (thirdPointDistance > radiusMax || thirdPointDistance < radiusMin) )
	  {
	  // Invalid combination. Go to next.
	  continue;
	  }

	// Calculate distance from intersection to all other points
	for (size_t l = 0; l < projectedPoints.size(); ++l)
	  {
	  if (l != i && l != j && l != k)
	    {
	    // TODO: Should we use Norm or SquaredNorm to score points ?
	    double distanceToIntersection = (projectedPoints[l] - intersection).GetNorm();
	    if (distanceToIntersection > radiusMax || distanceToIntersection < radiusMin)
	      {
	      scoreThirdPass[l] += distanceToIntersection;
	      }
	    }
	  }
	}
      }
    }

  std::cout << "Third Pass Results: " << std::endl;
  if ( RemoveBiggestOutlierFromList(fixedPointList, scoreThirdPass, projectedPoints.size()) )
    {
    std::cout << "[Third Pass]: Outlier found and removed." << std::endl;
    return 1;
    }
  std::cout << "[Third Pass]: No outlier found. Moving to third pass." << std::endl << std::endl;

  return 0;
}

//----------------------------------------
// Find biggest outlier from the list and remove it.
// To overcome some possible equality in the scores when using integers,
// we score points using doubles.
// It is much more unlikely to have two double numbers strictly equal,
// than it is with integers.
// Return 1 if outlier has been removed, 0 otherwise

int RemoveBiggestOutlierFromList(PointListType& pointList, double outlyingScores[], unsigned int size)
{
  if (pointList.size() != size)
    {
    return 0;
    }

  // Find index of the higher outlying score
  size_t higherIndex = 0;
  for (size_t i = 0; i < size; ++i)
    {
    std::cout << "Score[" << i << "]: " << outlyingScores[i] << std::endl;
    if (outlyingScores[i] > outlyingScores[higherIndex])
      {
      higherIndex = i;
      }
    }

  if (higherIndex < pointList.size())
    {
    if (outlyingScores[higherIndex] > 0)
      {
      // Outlier found. Remove it and return 1.
      PointListType::iterator iter = pointList.begin() + higherIndex;
      pointList.erase(iter);
      return 1;
      }
    }

  return 0;
}

//----------------------------------------
// Calculate the normal vector, radius, and center point of the model circle
// based on the movingPointList.
// Assume that the fixed points were generated from the model
// and does not contain error.

void FindCircleFromPoints(PointListType& movingPointList, TransformType::Pointer transform, double& radius)
{
  if (movingPointList.size() < 1)
    {
    return;
    }

  VectorType v1 = movingPointList[1]-movingPointList[0];
  VectorType v2 = movingPointList[2]-movingPointList[0];
  VectorType srcNormal = itk::CrossProduct(v1, v2);
  srcNormal.Normalize();
  
  // Define arbtrary vector that is perpendicular to the srcNormal
  // Use the right-hand coordinate system
  VectorType srcInplane1 = v1;
  srcInplane1.Normalize();
  VectorType srcInplane2 = itk::CrossProduct(srcNormal, srcInplane1);
  srcInplane2.Normalize();

  // Estimate the center
  MatrixType rotationMatrix;
  for (int i = 0; i < 3; ++i)
    {
    rotationMatrix[i][0] = srcInplane1[i];
    rotationMatrix[i][1] = srcInplane2[i];
    rotationMatrix[i][2] = srcNormal[i];
    }

  VectorType center;
  FindCenter(movingPointList, rotationMatrix, center);

  double sumRadius = 0.0;
  for (size_t i = 0; i < movingPointList.size(); ++i)
    {
    VectorType v = movingPointList[i] - center;
    sumRadius += v.GetNorm();
    }

  radius = sumRadius / (double)movingPointList.size();

  transform->SetIdentity();
  transform->SetMatrix(rotationMatrix);
  transform->SetOffset(center);
}


//--------------------------------------------------------------------------------
// First step consists of calculating a first estimation of the angle by matching
// a selected point from one set to all points of the other set and calculate
// average minimum square distance for each rotation. Use the angle with minimum average minimum
// square distance as the angle estimation.
// Second step consists of rotating first set around estimated angle +/- 2 degrees with
// a fine step (0.1 degree).
// Return best fitting angle (in degrees) between 2 pointsets.

double FindRotationAngle(MatrixType& originToPlaneMatrix,
                         PointListType& movingPointList,
                         PointListType& fixedPointList,
			 VectorType axisVector, VectorType center,
			 double tuningStep, double tuningRange,
			 double& minAvgMinSqDist)
{

  //----------------------------------------
  // Rotate points from original position to in-plane position
  
  PointListType inPlanePoints;
  for (size_t i = 0; i < movingPointList.size(); ++i)
    {
      VectorType pp = originToPlaneMatrix*movingPointList[i] + center;
      inPlanePoints.push_back(pp);
    }
  
  //----------------------------------------
  // Rotate point around axis vector and compute average minimum square distance

  double estimatedAngle = EstimateRotationAngle(inPlanePoints, fixedPointList,
                                                axisVector, center);
  double fineTunedAngle = FineTuneRotationAngle(inPlanePoints, fixedPointList,
						axisVector, center, estimatedAngle,
						tuningStep, tuningRange, minAvgMinSqDist);
  return fineTunedAngle;
}


//--------------------------------------------------------------------------------
// Estimate best fitting angle by computing angle between 2 given points (and center),
// and compute average minimum square distance for this rotation.
// Process is repeated for all 3-points combination.
// Return estimated angle (in radians) between 2 pointsets.

double EstimateRotationAngle(PointListType& inPlanePoints, PointListType& fixedPointList,
			     VectorType axisVector, VectorType center)
{
  if (fixedPointList.size() < 1)
    {
    return 0;
    }

  double estimatedAngle = -1.0;
  double minAverageMinSqDist = -1.0;
  PointListType rotatedPoints;

  // Select first point
  VectorType selectedPoint = fixedPointList[0];

  for (size_t i = 0; i < inPlanePoints.size(); ++i)
    {
    rotatedPoints.clear();

    double currentAngle = AngleBetweenPoints(inPlanePoints[i], selectedPoint, center, axisVector);
    RotatePoints(inPlanePoints,
		 axisVector, center, currentAngle,
		 rotatedPoints);
    double averageMinSqDist = AverageMinimumSquareDistance(rotatedPoints, fixedPointList);

    if (minAverageMinSqDist < 0 || averageMinSqDist < minAverageMinSqDist)
      {
      minAverageMinSqDist = averageMinSqDist;
      estimatedAngle = currentAngle;
      }
    }

  return estimatedAngle;
}

//--------------------------------------------------------------------------------
// Rotate pointset around axis vector, with the circle center as rotation center,
// by 'estimatedAngle +/- 2' degrees with a fine angle step (0.1 degree), and calculate
// average minimum square distance for each. Keep the angle with the minimum average
// minimum square distance. Return fine tuned angle between both pointsets.

double FineTuneRotationAngle(PointListType& inPlanePoints, PointListType& fixedPointList,
                             VectorType axisVector, VectorType center, double estimatedAngle,
                             double tuningStep,  double tuningRange,
                             double& minAvgMinSqDist)
{
  double fineTunedAngle = -1.0;
  PointListType rotatedPoints;

  // We compute average minimum distance for angle estimatedAngle +/-tuningRange (radian)
  // with a step of 'tuningStep'
  for (double angle = estimatedAngle-tuningRange; angle < estimatedAngle+tuningRange; angle += tuningStep)
    {
    rotatedPoints.clear();

    RotatePoints(inPlanePoints,
		 axisVector, center, angle,
		 rotatedPoints);
    double averageMinSqDist = AverageMinimumSquareDistance(rotatedPoints, fixedPointList);

    if (minAvgMinSqDist < 0 || averageMinSqDist < minAvgMinSqDist)
      {
      minAvgMinSqDist = averageMinSqDist;
      fineTunedAngle = angle;
      }
    }

  return fineTunedAngle;
}

//--------------------------------------------------------------------------------
// Return the angle (in degrees) between 3 points.

double AngleBetweenPoints(VectorType p1, VectorType p2, VectorType origin, VectorType axisVector)
{
  // Calculate the angle between v1 and v2
  VectorType v1 = p1 - origin;
  VectorType v2 = p2 - origin;

  double dotProduct = v1*v2;
  double v1Norm = v1.GetNorm();
  double v2Norm = v2.GetNorm();
  double theta = std::acos( dotProduct / (v1Norm*v2Norm) );

  VectorType rotAxis = itk::CrossProduct(v1, v2);
  double rotAxisDotProduct = rotAxis * axisVector;
  double sign = (rotAxisDotProduct > 0.0) ? 1.0: -1.0;
  
  return sign * theta;
}


//--------------------------------------------------------------------------------
// Transform all points in the list

void TransformPoints(PointListType& inputPoints, PointListType& outputPoints,
                     TransformType::Pointer transform)
{
  outputPoints.clear();

  for (size_t i = 0; i < inputPoints.size(); ++i)
    {
    PointType rp = transform->TransformPoint(inputPoints[i]);
    outputPoints.push_back(rp.GetVectorFromOrigin());
    }
}

//--------------------------------------------------------------------------------
// Rotate a pointset around the axis vector by a given angle, with the circle 
// center as rotation center and output new rotated pointset

void RotatePoints(PointListType& inputPoints,
		  VectorType axisVector, VectorType center, double angle,
		  PointListType& outputPoints)
{
  outputPoints.clear();

  TransformType::Pointer rotationTransform = TransformType::New();
  rotationTransform->SetCenter(center);
  rotationTransform->Rotate3D(axisVector, angle);

  TransformPoints(inputPoints, outputPoints, rotationTransform);
}

//--------------------------------------------------------------------------------
// Return the average minimum square distance between 2 pointsets

double AverageMinimumSquareDistance(PointListType& set1, PointListType& set2)
{
  // If set1 and set2 have the same size, order doesn't matter.
  // We then defined smallest = set1, biggest = set2
  PointListType smallestSet = (set1.size() <= set2.size() ? set1 : set2);
  PointListType biggestSet = (set1.size() > set2.size() ? set1 : set2);

  double sumMinSquareDistance = 0.0;
  for (size_t i = 0; i < smallestSet.size(); ++i)
    {
    double minSquareDistance = -1.0;
    VectorType p1 = smallestSet[i];

    for (size_t j = 0; j < biggestSet.size(); ++j)
      {
      VectorType p2 = biggestSet[j];
      VectorType d = p2-p1;

      double squareDistance = d.GetSquaredNorm();
      if (minSquareDistance < 0 || squareDistance < minSquareDistance)
	{
        minSquareDistance = squareDistance;
	}
      }
    sumMinSquareDistance += minSquareDistance;
    }

  return sumMinSquareDistance / (double)smallestSet.size();
}
