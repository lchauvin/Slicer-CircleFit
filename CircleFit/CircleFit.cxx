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
typedef itk::Statistics::ListSample< VectorType > PointListType;
typedef PointListType::Iterator PointListIteratorType;
typedef itk::Statistics::CovarianceSampleFilter< PointListType > CovarianceAlgorithmType;
typedef itk::SymmetricEigenAnalysis< CovarianceAlgorithmType::MatrixType, ArrayType, MatrixType > SymmetricEigenAnalysisType;


int    CalcIntersectionOfPerpendicularBisectors2D(VectorType& p1, VectorType& p2, VectorType& p3, VectorType& intersec, double radius);

double FindRotationAngle(PointListType::Pointer inPlanePoints, PointListType::Pointer dstPoints,
			 VectorType principalVector, VectorType center,
			 double tuningStep,
			 double& minAvgMinSqDist);

double FindEstimatedAngle(PointListType::Pointer inPlanePoints, PointListType::Pointer dstPoints,
			  VectorType principalVector, VectorType center);
double AngleBetweenPoints(VectorType p1, VectorType p2, VectorType p3, VectorType principalVector);
void RotatePoints(PointListType::Pointer inputPoints,
		  VectorType principalVector, VectorType center, double angle,
		  PointListType::Pointer outputPoints);
double FineTuneAngle(PointListType::Pointer inPlanePoints, PointListType::Pointer dstPoints,
		     VectorType principalVector, VectorType center, double estimatedAngle,
		     double tuningStep,
		     double& minAvgMinSqDist);

double AverageMinimumSquareDistance(PointListType::Pointer set1, PointListType::Pointer set2);


int main( int argc, char * argv[] )
{
  PARSE_ARGS;

  //----------------------------------------
  // Convert points into PointListType

  PointListType::Pointer srcPoints, dstPoints;
  srcPoints = PointListType::New();
  dstPoints = PointListType::New();

  size_t numberOfMovingPoints = movingPoints.size();
  for (size_t mp = 0; mp < numberOfMovingPoints; ++mp)
    {
    VectorType tmpMp;
    tmpMp[0] = movingPoints[mp][0];
    tmpMp[1] = movingPoints[mp][1];
    tmpMp[2] = movingPoints[mp][2];
    srcPoints->PushBack(tmpMp);
    }

  size_t numberOfFixedPoints = fixedPoints.size();
  for (size_t fp = 0; fp < numberOfFixedPoints; ++fp)
    {
    VectorType tmpFp;
    tmpFp[0] = fixedPoints[fp][0];
    tmpFp[1] = fixedPoints[fp][1];
    tmpFp[2] = fixedPoints[fp][2];
    dstPoints->PushBack(tmpFp);
    }

  //----------------------------------------
  // Perform PCA
  
  CovarianceAlgorithmType::Pointer covarianceAlgorithm = 
    CovarianceAlgorithmType::New();
  covarianceAlgorithm->SetInput( dstPoints );
  covarianceAlgorithm->Update();
    
  // Perform Symmetric Eigen Analysis
  SymmetricEigenAnalysisType analysis ( 3 );
  ArrayType eigenValues;
  MatrixType eigenMatrix;
  analysis.SetOrderEigenMagnitudes( true );
  analysis.ComputeEigenValuesAndVectors( covarianceAlgorithm->GetCovarianceMatrix(),
                                         eigenValues, eigenMatrix );    

  //----------------------------------------
  // Extract the normal vector 
  // The first eigen vector (with the minimal eigenvalue) is
  // the normal vector of the fitted plane.

  VectorType nx =  eigenMatrix[2];
  VectorType ny =  eigenMatrix[1];
  VectorType nz =  eigenMatrix[0];
  VectorType principalVector = nz;

  //----------------------------------------
  // Calculate the average position of the all points.
  // This is used as the origin of the new coordinate system on the
  // fitted plane.

  CovarianceAlgorithmType::MeasurementVectorType meanPoint;
  meanPoint = covarianceAlgorithm->GetMean();
  
  //----------------------------------------
  // Project all the points to the fitted plane.

  PointListType::Pointer projectedPoints = PointListType::New();
  for (PointListIteratorType iter = dstPoints->Begin(); iter != dstPoints->End(); ++iter)
    {
    VectorType p1;
    VectorType p2;
    p1 = iter.GetMeasurementVector() - meanPoint;
    p2[0] = p1*nx;
    p2[1] = p1*ny;
    p2[2] = 0.0;
    projectedPoints->PushBack(p2);
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

  for (PointListIteratorType iter1 = projectedPoints->Begin(); iter1 != projectedPoints->End(); ++iter1)
    {
    PointListIteratorType iter2 = iter1;
    for (++iter2; iter2 != projectedPoints->End(); ++iter2)
      {
      PointListIteratorType iter3 = iter2;
      for (++iter3; iter3 != projectedPoints->End(); ++iter3)
        {
        VectorType p1 = iter1.GetMeasurementVector();
        VectorType p2 = iter2.GetMeasurementVector();
        VectorType p3 = iter3.GetMeasurementVector();
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
  meanIntersect = meanIntersect / nPointsUsed;

  //----------------------------------------
  // Transform the center point to the original coordinate system

  VectorType center = meanIntersect[0] * nx + meanIntersect[1] * ny + meanIntersect[2] * nz + meanPoint;

  std::cout << "Number of estimated center points: " << nPoints << std::endl;
  std::cout << "Number of estimated center points used: " << nPointsUsed << std::endl;
  std::cout << "Center = " << center << std::endl;  
  std::cout << std::endl;

  //----------------------------------------
  // Calculate matrix from original coordinate system to plane coordinate system

  MatrixType originalToPlaneMatrix;
  for (int i = 0; i < 3; ++i)
    {
    originalToPlaneMatrix[i][0] = nx[i];
    originalToPlaneMatrix[i][1] = ny[i];
    originalToPlaneMatrix[i][2] = nz[i];
    }
    
  //----------------------------------------
  // Rotate points from original position to in-plane position
  
  PointListType::Pointer inPlanePoints = PointListType::New();
  for (PointListIteratorType iter = srcPoints->Begin(); iter != srcPoints->End(); ++iter)
    {
    VectorType pp = originalToPlaneMatrix*iter.GetMeasurementVector() + center;
    inPlanePoints->PushBack(pp);
    }

  //----------------------------------------
  // Rotate point around principal vector and compute average minimum squiare distance

  double minAvgMinSqDist = -1.0;
  double bestAngle = FindRotationAngle(inPlanePoints, dstPoints,
				       principalVector, center,
				       0.1,
				       minAvgMinSqDist);

  //----------------------------------------
  // Rotate circle around one of the other axis, nx or ny, and recalculate average minimum squiare distance
  // to also find the global minimum (including symmetry)

  TransformType::Pointer flippingTransform = TransformType::New();
  flippingTransform->SetCenter(center);
  flippingTransform->Rotate3D(nx, M_PI);

  PointListType::Pointer flippedInPlanePoints = PointListType::New();
  for (PointListIteratorType iter = inPlanePoints->Begin(); iter != inPlanePoints->End(); ++iter)
    {
    PointType pp = flippingTransform->TransformPoint(iter.GetMeasurementVector());
    VectorType vp;
    vp[0] = pp[0];
    vp[1] = pp[1];
    vp[2] = pp[2];
    flippedInPlanePoints->PushBack(vp);
    }

  double flippedMinAvgMinSqDist = -1.0;
  double flippedBestAngle = FindRotationAngle(flippedInPlanePoints, dstPoints,
					      principalVector, center,
					      0.1,
					      flippedMinAvgMinSqDist);

  //----------------------------------------
  // Build registration transform
  
  TransformType::Pointer registrationTransform = TransformType::New();
  registrationTransform->SetIdentity();
  registrationTransform->SetMatrix(originalToPlaneMatrix);

  std::cout << "Fitting Angle: ";
  if (flippedMinAvgMinSqDist < minAvgMinSqDist)
    {
    registrationTransform->Rotate3D(nx, M_PI);
    registrationTransform->Rotate3D(principalVector, flippedBestAngle * M_PI / 180);
    std::cout << flippedBestAngle << " (flipped)" << std::endl;
    }
  else
    {
    registrationTransform->Rotate3D(principalVector, bestAngle * M_PI / 180);
    std::cout << bestAngle << std::endl;
    }
  registrationTransform->SetOffset(center);

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
// First step consists of calculating a first estimation of the angle by matching
// a selected point from one set to all points of the other set and calculate
// average minimum square distance for each rotation. Use the angle with minimum average minimum
// square distance as the angle estimation.
// Second step consists of rotating first set around estimated angle +/- 2 degrees with
// a fine step (0.1 degree).
// Return best fitting angle (in degrees) between 2 pointsets.

double FindRotationAngle(PointListType::Pointer inPlanePoints, PointListType::Pointer dstPoints,
			 VectorType principalVector, VectorType center,
			 double tuningStep,
			 double& minAvgMinSqDist)
{
  double estimatedAngle = FindEstimatedAngle(inPlanePoints, dstPoints,
					     principalVector, center);
  double fineTunedAngle = FineTuneAngle(inPlanePoints, dstPoints,
					principalVector, center, estimatedAngle,
					tuningStep,
					minAvgMinSqDist);
  return fineTunedAngle;
}

//--------------------------------------------------------------------------------
// Estimate best fitting angle by computing angle between 2 given points (and center),
// and compute average minimum square distance for this rotation.
// Process is repeated for all 3-points combination.
// Return estimated angle (in degrees) between 2 pointsets.

double FindEstimatedAngle(PointListType::Pointer inPlanePoints, PointListType::Pointer dstPoints,
			  VectorType principalVector, VectorType center)
{
  double estimatedAngle = -1.0;
  double minAverageMinSqDist = -1.0;
  PointListType::Pointer rotatedPoints = PointListType::New();

  // Select first point
  VectorType selectedPoint = inPlanePoints->Begin().GetMeasurementVector();

  for (PointListIteratorType iter = dstPoints->Begin(); iter != dstPoints->End(); ++iter)
    {
    rotatedPoints->Clear();

    double currentAngle = AngleBetweenPoints(selectedPoint, iter.GetMeasurementVector(), center, principalVector);
    RotatePoints(inPlanePoints,
		 principalVector, center, currentAngle,
		 rotatedPoints);
    double averageMinSqDist = AverageMinimumSquareDistance(rotatedPoints, dstPoints);

    if (minAverageMinSqDist < 0 || averageMinSqDist < minAverageMinSqDist)
      {
      minAverageMinSqDist = averageMinSqDist;
      estimatedAngle = currentAngle;
      }
    }

  return estimatedAngle;
}

//--------------------------------------------------------------------------------
// Return the angle (in degrees) between 3 points.

double AngleBetweenPoints(VectorType p1, VectorType p2, VectorType p3, VectorType principalVector)
{
  // Calculate angle value
  VectorType v1 = p1 - p3;
  VectorType v2 = p2 - p3;
  double dotProduct = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];
  double v1Norm = std::sqrt(std::pow(v1[0],2) + std::pow(v1[1],2) + std::pow(v1[2],2));
  double v2Norm = std::sqrt(std::pow(v2[0],2) + std::pow(v2[1],2) + std::pow(v2[2],2));
  double theta = std::acos( dotProduct / (v1Norm*v2Norm) );

  // Calculate angle between v1,v2 perpendicular vector and principal vector
  // If angle ~180 degrees, sign of the rotation angle should be inverted
  double rightHandAxis[3] = { v1[1]*v2[2] - v1[2]*v2[1],
			      v1[2]*v2[0] - v1[0]*v2[2],
			      v1[0]*v2[1] - v1[1]*v2[0] };

  double rotationAxisDotProduct =
    rightHandAxis[0]*principalVector[0] +
    rightHandAxis[1]*principalVector[1] +
    rightHandAxis[2]*principalVector[2];

  double crossProduct[3] = { rightHandAxis[1]*principalVector[2] - rightHandAxis[2]*principalVector[1],
			     rightHandAxis[2]*principalVector[0] - rightHandAxis[0]*principalVector[2],
			     rightHandAxis[0]*principalVector[1] - rightHandAxis[1]*principalVector[0] };

  double crossProductNorm = std::sqrt(std::pow(crossProduct[0],2) +
				      std::pow(crossProduct[1],2) +
				      std::pow(crossProduct[2],2));

  double axisAngle = std::atan2(crossProductNorm, rotationAxisDotProduct) * 180 / M_PI;
  int rotationSign = ((axisAngle < 90 && axisAngle > -90) ? 1 : -1);

  return rotationSign * theta * 180 / M_PI;
}

//--------------------------------------------------------------------------------
// Rotate a pointset around the principal vector by a given angle, with the circle 
// center as rotation center and output new rotated pointset

void RotatePoints(PointListType::Pointer inputPoints,
		  VectorType principalVector, VectorType center, double angle,
		  PointListType::Pointer outputPoints)
{
  outputPoints->Clear();

  TransformType::Pointer rotationTransform = TransformType::New();
  rotationTransform->SetCenter(center);
  rotationTransform->Rotate3D(principalVector, angle * M_PI / 180);

  for (PointListIteratorType iter = inputPoints->Begin(); iter != inputPoints->End(); ++iter)
    {
    PointType rp = rotationTransform->TransformPoint(iter.GetMeasurementVector());
    VectorType rotatedPoint;
    rotatedPoint[0] = rp[0];
    rotatedPoint[1] = rp[1];
    rotatedPoint[2] = rp[2];
    outputPoints->PushBack(rotatedPoint);
    }
}


//--------------------------------------------------------------------------------
// Return the average minimum square distance between 2 pointsets

double AverageMinimumSquareDistance(PointListType::Pointer set1, PointListType::Pointer set2)
{
  // TODO: What if set1 and set2 have different number of points ?

  double total = 0.0;
  int numberOfPoints = 0;

  for (PointListIteratorType iter1 = set1->Begin(); iter1 != set1->End(); ++iter1)
    {
    double minSquareDistance = -1.0;
    VectorType p1 = iter1.GetMeasurementVector();

    for (PointListIteratorType iter2 = set2->Begin(); iter2 != set2->End(); ++iter2)
      {
      VectorType p2 = iter2.GetMeasurementVector();

      double squareDistance = std::pow(p2[0]-p1[0],2) + std::pow(p2[1]-p1[1],2) + std::pow(p2[2]-p1[2],2);

      if (minSquareDistance < 0 || squareDistance < minSquareDistance)
	{
        minSquareDistance = squareDistance;
	}
      }
    total += minSquareDistance;
    numberOfPoints++;
    }

  return total / (double) numberOfPoints;

}

//--------------------------------------------------------------------------------
// Rotate pointset around principal vector, with the circle center as rotation center,
// by 'estimatedAngle +/- 2' degrees with a fine angle step (0.1 degree), and calculate
// average minimum square distance for each. Keep the angle with the minimum average
// minimum square distance. Return fine tuned angle between both pointsets.

double FineTuneAngle(PointListType::Pointer inPlanePoints, PointListType::Pointer dstPoints,
		     VectorType principalVector, VectorType center, double estimatedAngle,
		     double tuningStep,
		     double& minAvgMinSqDist)
{
    double fineTunedAngle = -1.0;
    PointListType::Pointer rotatedPoints = PointListType::New();

  // We compute average minimum distance for angle estimatedAngle +/- 2, with a step of 'tuningStep'
  for (double angle = estimatedAngle-2; angle < estimatedAngle+2; angle += tuningStep)
    {
    rotatedPoints->Clear();

    RotatePoints(inPlanePoints,
		 principalVector, center, angle,
		 rotatedPoints);
    double averageMinSqDist = AverageMinimumSquareDistance(rotatedPoints, dstPoints);

    if (minAvgMinSqDist < 0 || averageMinSqDist < minAvgMinSqDist)
      {
      minAvgMinSqDist = averageMinSqDist;
      fineTunedAngle = angle;
      }
    }

  return fineTunedAngle;
}
