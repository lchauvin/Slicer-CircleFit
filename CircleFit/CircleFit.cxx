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

void   FindCircleFromPoints(PointListType::Pointer movingPointList, TransformType::Pointer transform, double& radius);
void   FindCenter(PointListType::Pointer points, MatrixType& originalToPlaneMatrix, VectorType& center, double radius);
double FindTransformByRotation(MatrixType& originalToPlaneMatrix, PointListType::Pointer movingPointList, PointListType::Pointer fixedPointList,
                               VectorType axisVector, VectorType center, double tuningStep, double tuningRange, double minAvgMinSqDist);


double FindRotationAngle(MatrixType& originalToPlaneMatrix,
                         PointListType::Pointer movingPointList,
                         PointListType::Pointer fixedPointList,
			 VectorType principalVector, VectorType center,
			 double tuningStep, double tuningRange,
			 double& minAvgMinSqDist);
double EstimateRotationAngle(PointListType::Pointer inPlanePoints, PointListType::Pointer fixedPointList,
			  VectorType principalVector, VectorType center);
double AngleBetweenPoints(VectorType p1, VectorType p2, VectorType origin, VectorType principalVector);
void   TransformPoints(PointListType::Pointer inputPoints, PointListType::Pointer outputPoints,
                     TransformType::Pointer transform);
void   RotatePoints(PointListType::Pointer inputPoints,
                    VectorType principalVector, VectorType center, double angle,
                    PointListType::Pointer outputPoints);
double FineTuneRotationAngle(PointListType::Pointer inPlanePoints, PointListType::Pointer fixedPointList,
		     VectorType principalVector, VectorType center, double estimatedAngle,
		     double tuningStep,  double tuningRange,
		     double& minAvgMinSqDist);

double AverageMinimumSquareDistance(PointListType::Pointer set1, PointListType::Pointer set2);



int main( int argc, char * argv[] )
{
  
  PARSE_ARGS;
  
  //----------------------------------------
  // Convert points into PointListType

  PointListType::Pointer movingPointList = PointListType::New();
  size_t numberOfMovingPoints = movingPoints.size();
  for (size_t mp = 0; mp < numberOfMovingPoints; ++mp)
    {
    VectorType tmpMp;
    tmpMp[0] = movingPoints[mp][0];
    tmpMp[1] = movingPoints[mp][1];
    tmpMp[2] = movingPoints[mp][2];
    movingPointList->PushBack(tmpMp);
    }

  PointListType::Pointer fixedPointList = PointListType::New();
  size_t numberOfFixedPoints = fixedPoints.size();
  for (size_t fp = 0; fp < numberOfFixedPoints; ++fp)
    {
    VectorType tmpFp;
    tmpFp[0] = fixedPoints[fp][0];
    tmpFp[1] = fixedPoints[fp][1];
    tmpFp[2] = fixedPoints[fp][2];
    fixedPointList->PushBack(tmpFp);
    }

  //--------------------------------------------------------------------------------
  //
  // The registration process consists of the following steps:
  //
  //  1. Find the transform from the moving points to the origin (the circle is on the X-Y plane).
  //  2. Transform the moving points to the origin. The resulted point list is named 'originPoints'.
  //  3. Find the transform from the origin to the fixed points.
  //  4. Compute the transform from the moving points to to the fixed points.
  //

  //----------------------------------------
  // 1. Find the transform from the moving points to the origin (the circle is on the X-Y plane).

  // Find transform from original model (a circle on the X-Y plane with the center at the origin) 
  double srcRadius;
  TransformType::Pointer originToMovingTransform = TransformType::New();
  FindCircleFromPoints(movingPointList, originToMovingTransform, srcRadius);

  //std::cout << "srcTransformMatrix = " << srcTransformMatrix << std::endl;
  std::cout << "originToMoving = " << originToMovingTransform << std::endl;
  std::cout << "srcRadius = " << srcRadius << std::endl;
  
  //----------------------------------------
  // 2. Transform the moving points to the origin. The resulted point list is named 'originPoints'.

  TransformType::Pointer movingToOriginTransform = TransformType::New();
  originToMovingTransform->GetInverse(movingToOriginTransform);

  PointListType::Pointer originPointList = PointListType::New();

  for (PointListIteratorType iter = movingPointList->Begin(); iter != movingPointList->End(); ++iter)
    {
    PointType p = movingToOriginTransform->TransformPoint(iter.GetMeasurementVector());
    originPointList->PushBack(p.GetVectorFromOrigin());
    }


  //----------------------------------------
  // 3. Find the transform from the origin to the fixed points.

  // Perform PCA
  CovarianceAlgorithmType::Pointer covarianceAlgorithm = 
    CovarianceAlgorithmType::New();
  covarianceAlgorithm->SetInput( fixedPointList );
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
  VectorType nx =  eigenMatrix[1];
  VectorType ny =  eigenMatrix[2];
  VectorType nz =  itk::CrossProduct(nx, ny);


  // Calculate matrix from original coordinate system to plane coordinate system

  MatrixType originToPlaneMatrix;
  for (int i = 0; i < 3; ++i)
    { 
    originToPlaneMatrix[i][0] = nx[i];
    originToPlaneMatrix[i][1] = ny[i];
    originToPlaneMatrix[i][2] = nz[i];
    }
  VectorType axisVector = nz;
    
  // Estimate the center
  VectorType center;
  FindCenter(fixedPointList, originToPlaneMatrix, center, srcRadius);
  std::cout << "originToPlaneMatrix = " << originToPlaneMatrix << std::endl;

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
  // 4. Compute the transform from the moving points to to the fixed points.
  
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
void FindCenter(PointListType::Pointer points, MatrixType& rotationMatrix, VectorType& center, double radius=0)
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
  for (PointListIteratorType iter = points->Begin(); iter != points->End(); ++iter)
    {
    meanPoint += iter.GetMeasurementVector();
    }
  meanPoint /= (double)points->Size();
  std::cerr << "meanPoint = " << meanPoint << std::endl;
  std::cerr << "points->Size() = " << points->Size() << std::endl;

  //----------------------------------------
  // Project all the points to the fitted plane.

  PointListType::Pointer projectedPoints = PointListType::New();
  for (PointListIteratorType iter = points->Begin(); iter != points->End(); ++iter)
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

  meanIntersect /= (double)nPointsUsed;

  center = meanIntersect[0] * nx + meanIntersect[1] * ny + meanIntersect[2] * nz + meanPoint;

  std::cout << "Number of estimated center points: " << nPoints << std::endl;
  std::cout << "Number of estimated center points used: " << nPointsUsed << std::endl;
  std::cout << "Center = " << center << std::endl;  
  std::cout << std::endl;

}


void FindCircleFromPoints(PointListType::Pointer movingPointList, TransformType::Pointer transform, double& radius)
{

  //----------------------------------------
  // Calculate the normal vector, radius, and center point of the model circle
  // based on the movingPointList.
  // Assume that the fixed points were generated from the model
  // and does not contain error.

  VectorType v1 = movingPointList->GetMeasurementVector(1)-movingPointList->GetMeasurementVector(0);
  VectorType v2 = movingPointList->GetMeasurementVector(2)-movingPointList->GetMeasurementVector(0);
  VectorType srcNormal = itk::CrossProduct(v1, v2);
  srcNormal.Normalize();
  
  // Define arbtrary vector that is perpendicular to the srcNormal
  // Use the right-hand coordinate system
  VectorType srcInplane1 = v1;
  srcInplane1.Normalize();
  VectorType srcInplane2 = itk::CrossProduct(srcNormal, srcInplane1);
  srcInplane2.Normalize();

  MatrixType rotationMatrix;
  // Estimate the center
  for (int i = 0; i < 3; ++i)
    {
    rotationMatrix[i][0] = srcInplane1[i];
    rotationMatrix[i][1] = srcInplane2[i];
    rotationMatrix[i][2] = srcNormal[i];
    }

  VectorType center;
  FindCenter(movingPointList, rotationMatrix, center);

  double sumRadius = 0.0;
  for (PointListIteratorType iter = movingPointList->Begin(); iter != movingPointList->End(); ++iter)
    {
    VectorType v = iter.GetMeasurementVector() - center;
    sumRadius += v.GetNorm();
    }

  radius = sumRadius / (double) movingPointList->Size();

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
                         PointListType::Pointer movingPointList,
                         PointListType::Pointer fixedPointList,
			 VectorType axisVector, VectorType center,
			 double tuningStep, double tuningRange,
			 double& minAvgMinSqDist)
{

  //----------------------------------------
  // Rotate points from original position to in-plane position
  
  PointListType::Pointer inPlanePoints = PointListType::New();
  for (PointListIteratorType iter = movingPointList->Begin(); iter != movingPointList->End(); ++iter)
    {
      VectorType pp = originToPlaneMatrix*iter.GetMeasurementVector() + center;
      inPlanePoints->PushBack(pp);
    }
  
  //----------------------------------------
  // Rotate point around axis vector and compute average minimum squiare distance

  double estimatedAngle = EstimateRotationAngle(inPlanePoints, fixedPointList,
                                                axisVector, center);
  double fineTunedAngle = FineTuneRotationAngle(inPlanePoints, fixedPointList,
					axisVector, center, estimatedAngle,
					tuningStep, tuningRange, minAvgMinSqDist);
  return fineTunedAngle;
}


double EstimateTransformByRotation(MatrixType& originToPlaneMatrix,
                             PointListType::Pointer movingPointList,
                             PointListType::Pointer fixedPointList,
                             VectorType axisVector,
                             VectorType center,
                             double tuningStep,
                             double tuningRange,
                             TransformType::Pointer estimatedTransform)
{
  double minAvgMinSqDist;

  TransformType::Pointer originToPlaneTransform = TransformType::New();
  originToPlaneTransform->SetMatrix(originToPlaneMatrix);
  originToPlaneTransform->SetOffset(center);

  ////----------------------------------------
  //// Rotate points from original position to in-plane position
  //PointListType::Pointer inPlanePoints = PointListType::New();
  //for (PointListIteratorType iter = movingPointList->Begin(); iter != movingPointList->End(); ++iter)
  //  {
  //    VectorType pp = originToPlaneMatrix*iter.GetMeasurementVector() + center;
  //    inPlanePoints->PushBack(pp);
  //  }
  PointListType::Pointer inPlanePoints = PointListType::New();
  TransformPoints(movingPointList, inPlanePoints, originToPlaneTransform);
  
  //----------------------------------------
  // Rotate point around axis vector and compute average minimum squiare distance

  double estimatedAngle = EstimateRotationAngle(inPlanePoints, fixedPointList,
                                                axisVector, center);
  double fineTunedAngle = FineTuneRotationAngle(inPlanePoints, fixedPointList,
                                                axisVector, center, estimatedAngle,
                                                tuningStep, tuningRange, minAvgMinSqDist);

  TransformType::Pointer rotationTransform = TransformType::New();
  rotationTransform->SetCenter(center);
  rotationTransform->Rotate3D(axisVector, fineTunedAngle);
  
  estimatedTransform->SetMatrix(originToPlaneMatrix);
  estimatedTransform->SetOffset(center);
  estimatedTransform->Compose(rotationTransform);

  return minAvgMinSqDist;
}





//--------------------------------------------------------------------------------
// Estimate best fitting angle by computing angle between 2 given points (and center),
// and compute average minimum square distance for this rotation.
// Process is repeated for all 3-points combination.
// Return estimated angle (in degrees) between 2 pointsets.

double EstimateRotationAngle(PointListType::Pointer inPlanePoints, PointListType::Pointer fixedPointList,
			  VectorType axisVector, VectorType center)
{
  double estimatedAngle = -1.0;
  double minAverageMinSqDist = -1.0;
  PointListType::Pointer rotatedPoints = PointListType::New();

  // Select first point
  VectorType selectedPoint = inPlanePoints->Begin().GetMeasurementVector();

  for (PointListIteratorType iter = fixedPointList->Begin(); iter != fixedPointList->End(); ++iter)
    {
    rotatedPoints->Clear();

    double currentAngle = AngleBetweenPoints(selectedPoint, iter.GetMeasurementVector(), center, axisVector);
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
// Trnasform all points in the list

void TransformPoints(PointListType::Pointer inputPoints, PointListType::Pointer outputPoints,
                     TransformType::Pointer transform)
{

  outputPoints->Clear();

  for (PointListIteratorType iter = inputPoints->Begin(); iter != inputPoints->End(); ++iter)
    {
    PointType rp = transform->TransformPoint(iter.GetMeasurementVector());
    outputPoints->PushBack(rp.GetVectorFromOrigin());
    }
}


//--------------------------------------------------------------------------------
// Rotate a pointset around the axis vector by a given angle, with the circle 
// center as rotation center and output new rotated pointset

void RotatePoints(PointListType::Pointer inputPoints,
		  VectorType axisVector, VectorType center, double angle,
		  PointListType::Pointer outputPoints)
{
  outputPoints->Clear();

  TransformType::Pointer rotationTransform = TransformType::New();
  rotationTransform->SetCenter(center);
  rotationTransform->Rotate3D(axisVector, angle);

  //for (PointListIteratorType iter = inputPoints->Begin(); iter != inputPoints->End(); ++iter)
  //  {
  //  PointType rp = rotationTransform->TransformPoint(iter.GetMeasurementVector());
  //  outputPoints->PushBack(rp->GetVectorFromOrigin());
  //  }
  TransformPoints(inputPoints, outputPoints, rotationTransform);
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
      VectorType d = p2-p1;

      double squareDistance = d.GetSquaredNorm();

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
// Rotate pointset around axis vector, with the circle center as rotation center,
// by 'estimatedAngle +/- 2' degrees with a fine angle step (0.1 degree), and calculate
// average minimum square distance for each. Keep the angle with the minimum average
// minimum square distance. Return fine tuned angle between both pointsets.

double FineTuneRotationAngle(PointListType::Pointer inPlanePoints, PointListType::Pointer fixedPointList,
                             VectorType axisVector, VectorType center, double estimatedAngle,
                             double tuningStep,  double tuningRange,
                             double& minAvgMinSqDist)
{
  double fineTunedAngle = -1.0;
  PointListType::Pointer rotatedPoints = PointListType::New();
  
  // We compute average minimum distance for angle estimatedAngle +/-tuningRange (radian)
  // with a step of 'tuningStep'
  for (double angle = estimatedAngle-tuningRange; angle < estimatedAngle+tuningRange; angle += tuningStep)
    {
    rotatedPoints->Clear();

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
