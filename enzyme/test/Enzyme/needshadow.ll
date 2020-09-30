; RUN: %opt < %s %loadEnzyme -enzyme -enzyme_preopt=false -S -mem2reg -instsimplify -simplifycfg | FileCheck %s
; ModuleID = 'inp.ll'
source_filename = "/mnt/Data/git/Enzyme/enzyme/test/Integration/simpleeigen-made.cpp"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }
%"class.Eigen::Matrix" = type { %"class.Eigen::PlainObjectBase" }
%"class.Eigen::PlainObjectBase" = type { %"class.Eigen::DenseStorage" }
%"class.Eigen::DenseStorage" = type { double*, i64, i64 }
%"class.Eigen::Matrix.6" = type { %"class.Eigen::PlainObjectBase.7" }
%"class.Eigen::PlainObjectBase.7" = type { %"class.Eigen::DenseStorage.14" }
%"class.Eigen::DenseStorage.14" = type { double*, i64 }
%"class.Eigen::internal::BlasVectorMapper" = type { double* }

@stderr = external dso_local local_unnamed_addr global %struct._IO_FILE*, align 8
@.str = private unnamed_addr constant [18 x i8] c"W(o=%d, i=%d)=%f\0A\00", align 1
@.str.1 = private unnamed_addr constant [12 x i8] c"M(o=%d)=%f\0A\00", align 1
@.str.2 = private unnamed_addr constant [12 x i8] c"O(i=%d)=%f\0A\00", align 1
@.str.3 = private unnamed_addr constant [68 x i8] c"Assertion Failed: fabs( [%s = %g] - [%s = %g] ) > %g at %s:%d (%s)\0A\00", align 1
@.str.4 = private unnamed_addr constant [9 x i8] c"Wp(i, o)\00", align 1
@.str.5 = private unnamed_addr constant [18 x i8] c"M(o) * Op_orig(i)\00", align 1
@.str.6 = private unnamed_addr constant [66 x i8] c"/mnt/Data/git/Enzyme/enzyme/test/Integration/simpleeigen-made.cpp\00", align 1
@__PRETTY_FUNCTION__.main = private unnamed_addr constant [23 x i8] c"int main(int, char **)\00", align 1
@.str.7 = private unnamed_addr constant [19 x i8] c"Wp(o=%d, i=%d)=%f\0A\00", align 1
@.str.8 = private unnamed_addr constant [6 x i8] c"Mp(o)\00", align 1
@.str.9 = private unnamed_addr constant [4 x i8] c"res\00", align 1
@.str.10 = private unnamed_addr constant [13 x i8] c"Mp(o=%d)=%f\0A\00", align 1
@.str.11 = private unnamed_addr constant [6 x i8] c"Op(i)\00", align 1
@.str.12 = private unnamed_addr constant [3 x i8] c"0.\00", align 1
@.str.13 = private unnamed_addr constant [13 x i8] c"Op(i=%d)=%f\0A\00", align 1
@.str.15 = private unnamed_addr constant [140 x i8] c"lhs.cols() == rhs.rows() && \22invalid matrix product\22 && \22if you wanted a coeff-wise or a dot product use the respective explicit functions\22\00", align 1
@.str.16 = private unnamed_addr constant [45 x i8] c"/usr/include/eigen3/Eigen/src/Core/Product.h\00", align 1
@__PRETTY_FUNCTION__._ZN5Eigen7ProductINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS1_IdLin1ELi1ELi0ELin1ELi1EEELi0EEC2ERKS2_RKS3_ = private unnamed_addr constant [274 x i8] c"Eigen::Product<Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, 0>::Product(const Eigen::Product::Lhs &, const Eigen::Product::Rhs &) [Lhs = Eigen::Matrix<double, -1, -1, 0, -1, -1>, Rhs = Eigen::Matrix<double, -1, 1, 0, -1, 1>, Option = 0]\00", align 1
@.str.18 = private unnamed_addr constant [399 x i8] c"(!(RowsAtCompileTime!=Dynamic) || (rows==RowsAtCompileTime)) && (!(ColsAtCompileTime!=Dynamic) || (cols==ColsAtCompileTime)) && (!(RowsAtCompileTime==Dynamic && MaxRowsAtCompileTime!=Dynamic) || (rows<=MaxRowsAtCompileTime)) && (!(ColsAtCompileTime==Dynamic && MaxColsAtCompileTime!=Dynamic) || (cols<=MaxColsAtCompileTime)) && rows>=0 && cols>=0 && \22Invalid sizes when resizing a matrix or array.\22\00", align 1
@.str.19 = private unnamed_addr constant [53 x i8] c"/usr/include/eigen3/Eigen/src/Core/PlainObjectBase.h\00", align 1
@__PRETTY_FUNCTION__._ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEEE6resizeEll = private unnamed_addr constant [156 x i8] c"void Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >::resize(Eigen::Index, Eigen::Index) [Derived = Eigen::Matrix<double, -1, 1, 0, -1, 1>]\00", align 1
@.str.21 = private unnamed_addr constant [186 x i8] c"(size<16 || (std::size_t(result)%16)==0) && \22System's malloc returned an unaligned pointer. Compile with EIGEN_MALLOC_ALREADY_ALIGNED=0 to fallback to handmade alignd memory allocator.\22\00", align 1
@.str.22 = private unnamed_addr constant [49 x i8] c"/usr/include/eigen3/Eigen/src/Core/util/Memory.h\00", align 1
@__PRETTY_FUNCTION__._ZN5Eigen8internal14aligned_mallocEm = private unnamed_addr constant [51 x i8] c"void *Eigen::internal::aligned_malloc(std::size_t)\00", align 1
@.str.23 = private unnamed_addr constant [149 x i8] c"rows >= 0 && (RowsAtCompileTime == Dynamic || RowsAtCompileTime == rows) && cols >= 0 && (ColsAtCompileTime == Dynamic || ColsAtCompileTime == cols)\00", align 1
@.str.24 = private unnamed_addr constant [52 x i8] c"/usr/include/eigen3/Eigen/src/Core/CwiseNullaryOp.h\00", align 1
@__PRETTY_FUNCTION__._ZN5Eigen14CwiseNullaryOpINS_8internal18scalar_constant_opIdEENS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEEEC2EllRKS3_ = private unnamed_addr constant [282 x i8] c"Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<double>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >::CwiseNullaryOp(Eigen::Index, Eigen::Index, const NullaryOp &) [NullaryOp = Eigen::internal::scalar_constant_op<double>, MatrixType = Eigen::Matrix<double, -1, 1, 0, -1, 1>]\00", align 1
@.str.25 = private unnamed_addr constant [14 x i8] c"v == T(Value)\00", align 1
@.str.26 = private unnamed_addr constant [52 x i8] c"/usr/include/eigen3/Eigen/src/Core/util/XprHelper.h\00", align 1
@__PRETTY_FUNCTION__._ZN5Eigen8internal19variable_if_dynamicIlLi1EEC2El = private unnamed_addr constant [92 x i8] c"Eigen::internal::variable_if_dynamic<long, 1>::variable_if_dynamic(T) [T = long, Value = 1]\00", align 1
@.str.27 = private unnamed_addr constant [47 x i8] c"dst.rows() == dstRows && dst.cols() == dstCols\00", align 1
@.str.28 = private unnamed_addr constant [53 x i8] c"/usr/include/eigen3/Eigen/src/Core/AssignEvaluator.h\00", align 1
@__PRETTY_FUNCTION__._ZN5Eigen8internal17resize_if_allowedINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EEddEEvRT_RKT0_RKNS0_9assign_opIT1_T2_EE = private unnamed_addr constant [313 x i8] c"void Eigen::internal::resize_if_allowed(DstXprType &, const SrcXprType &, const internal::assign_op<T1, T2> &) [DstXprType = Eigen::Matrix<double, -1, 1, 0, -1, 1>, SrcXprType = Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<double>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >, T1 = double, T2 = double]\00", align 1
@__PRETTY_FUNCTION__._ZN5Eigen8internal19variable_if_dynamicIlLi0EEC2El = private unnamed_addr constant [92 x i8] c"Eigen::internal::variable_if_dynamic<long, 0>::variable_if_dynamic(T) [T = long, Value = 0]\00", align 1
@__PRETTY_FUNCTION__._ZN5Eigen8internal17resize_if_allowedINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEES3_ddEEvRT_RKT0_RKNS0_9assign_opIT1_T2_EE = private unnamed_addr constant [244 x i8] c"void Eigen::internal::resize_if_allowed(DstXprType &, const SrcXprType &, const internal::assign_op<T1, T2> &) [DstXprType = Eigen::Matrix<double, -1, 1, 0, -1, 1>, SrcXprType = Eigen::Matrix<double, -1, 1, 0, -1, 1>, T1 = double, T2 = double]\00", align 1
@__PRETTY_FUNCTION__._ZN5Eigen14CwiseNullaryOpINS_8internal18scalar_constant_opIdEENS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEEC2EllRKS3_ = private unnamed_addr constant [286 x i8] c"Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<double>, Eigen::Matrix<double, -1, -1, 0, -1, -1> >::CwiseNullaryOp(Eigen::Index, Eigen::Index, const NullaryOp &) [NullaryOp = Eigen::internal::scalar_constant_op<double>, MatrixType = Eigen::Matrix<double, -1, -1, 0, -1, -1>]\00", align 1
@__PRETTY_FUNCTION__._ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEE6resizeEll = private unnamed_addr constant [160 x i8] c"void Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >::resize(Eigen::Index, Eigen::Index) [Derived = Eigen::Matrix<double, -1, -1, 0, -1, -1>]\00", align 1
@__PRETTY_FUNCTION__._ZN5Eigen8internal17resize_if_allowedINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EEddEEvRT_RKT0_RKNS0_9assign_opIT1_T2_EE = private unnamed_addr constant [317 x i8] c"void Eigen::internal::resize_if_allowed(DstXprType &, const SrcXprType &, const internal::assign_op<T1, T2> &) [DstXprType = Eigen::Matrix<double, -1, -1, 0, -1, -1>, SrcXprType = Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<double>, Eigen::Matrix<double, -1, -1, 0, -1, -1> >, T1 = double, T2 = double]\00", align 1
@.str.29 = private unnamed_addr constant [39 x i8] c"other.rows() == 1 || other.cols() == 1\00", align 1
@__PRETTY_FUNCTION__._ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEEE10resizeLikeINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEEvRKNS_9EigenBaseIT_EE = private unnamed_addr constant [289 x i8] c"void Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >::resizeLike(const EigenBase<OtherDerived> &) [Derived = Eigen::Matrix<double, -1, 1, 0, -1, 1>, OtherDerived = Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<double>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >]\00", align 1
@.str.30 = private unnamed_addr constant [53 x i8] c"row >= 0 && row < rows() && col >= 0 && col < cols()\00", align 1
@.str.31 = private unnamed_addr constant [53 x i8] c"/usr/include/eigen3/Eigen/src/Core/DenseCoeffsBase.h\00", align 1
@__PRETTY_FUNCTION__._ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll = private unnamed_addr constant [227 x i8] c"Eigen::DenseCoeffsBase<type-parameter-0-0, 1>::Scalar &Eigen::DenseCoeffsBase<Eigen::Matrix<double, -1, -1, 0, -1, -1>, 1>::operator()(Eigen::Index, Eigen::Index) [Derived = Eigen::Matrix<double, -1, -1, 0, -1, -1>, Level = 1]\00", align 1
@.str.32 = private unnamed_addr constant [29 x i8] c"index >= 0 && index < size()\00", align 1
@__PRETTY_FUNCTION__._ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl = private unnamed_addr constant [209 x i8] c"Eigen::DenseCoeffsBase<type-parameter-0-0, 1>::Scalar &Eigen::DenseCoeffsBase<Eigen::Matrix<double, -1, 1, 0, -1, 1>, 1>::operator()(Eigen::Index) [Derived = Eigen::Matrix<double, -1, 1, 0, -1, 1>, Level = 1]\00", align 1

; Function Attrs: alwaysinline norecurse nounwind uwtable
define dso_local i32 @main(i32 %argc, i8** nocapture readnone %argv) local_unnamed_addr #0 {
entry:
  %W = alloca %"class.Eigen::Matrix", align 8
  %M = alloca %"class.Eigen::Matrix.6", align 8
  %O = alloca %"class.Eigen::Matrix.6", align 8
  %Wp = alloca %"class.Eigen::Matrix", align 8
  %Mp = alloca %"class.Eigen::Matrix.6", align 8
  %Op = alloca %"class.Eigen::Matrix.6", align 8
  %0 = bitcast %"class.Eigen::Matrix"* %W to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %0) #8
  call void @llvm.memset.p0i8.i64(i8* align 8 %0, i8 0, i64 24, i1 false) #8
  %m_rows.i1 = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %W, i64 0, i32 0, i32 0, i32 1
  %1 = load i64, i64* %m_rows.i1, align 8, !tbaa !2
  %m_cols.i2 = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %W, i64 0, i32 0, i32 0, i32 2
  %2 = load i64, i64* %m_cols.i2, align 8, !tbaa !8
  %mul.i3 = mul nsw i64 %2, %1
  %cmp.i4 = icmp eq i64 %mul.i3, 4
  br i1 %cmp.i4, label %_ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EE6resizeElll.exit, label %if.then.i

if.then.i:                                        ; preds = %entry
  %3 = bitcast %"class.Eigen::Matrix"* %W to i8**
  %4 = load i8*, i8** %3, align 8, !tbaa !9
  call void @free(i8* %4) #8
  %call.i.i.i.i = call noalias i8* @malloc(i64 32) #8
  %5 = ptrtoint i8* %call.i.i.i.i to i64
  %rem.i.i.i.i = and i64 %5, 15
  %cmp1.i.i.i.i8 = icmp eq i64 %rem.i.i.i.i, 0
  br i1 %cmp1.i.i.i.i8, label %land.rhs.i.i.i.i10, label %cond.false.i.i.i.i11

land.rhs.i.i.i.i10:                               ; preds = %if.then.i
  store i8* %call.i.i.i.i, i8** %3, align 8, !tbaa !9
  br label %_ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EE6resizeElll.exit

cond.false.i.i.i.i11:                             ; preds = %if.then.i
  call void @__assert_fail(i8* getelementptr inbounds ([186 x i8], [186 x i8]* @.str.21, i64 0, i64 0), i8* getelementptr inbounds ([49 x i8], [49 x i8]* @.str.22, i64 0, i64 0), i32 161, i8* getelementptr inbounds ([51 x i8], [51 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen8internal14aligned_mallocEm, i64 0, i64 0)) #9
  unreachable

_ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EE6resizeElll.exit: ; preds = %land.rhs.i.i.i.i10, %entry
  store i64 2, i64* %m_rows.i1, align 8, !tbaa !2
  store i64 2, i64* %m_cols.i2, align 8, !tbaa !8
  %6 = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %W, i64 0, i32 0, i32 0, i32 0
  %7 = load double*, double** %6, align 8, !tbaa !9
  %8 = bitcast double* %7 to <2 x double>*
  store <2 x double> <double 3.000000e+00, double 3.000000e+00>, <2 x double>* %8, align 16, !tbaa !10
  %arrayidx.i35.1 = getelementptr inbounds double, double* %7, i64 2
  %9 = bitcast double* %arrayidx.i35.1 to <2 x double>*
  store <2 x double> <double 3.000000e+00, double 3.000000e+00>, <2 x double>* %9, align 16, !tbaa !10
  %10 = bitcast %"class.Eigen::Matrix.6"* %M to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %10) #8
  call void @llvm.memset.p0i8.i64(i8* align 8 %10, i8 0, i64 16, i1 false) #8
  %m_rows.i.i.i.i.i = getelementptr inbounds %"class.Eigen::Matrix.6", %"class.Eigen::Matrix.6"* %M, i64 0, i32 0, i32 0, i32 1
  %11 = load i64, i64* %m_rows.i.i.i.i.i, align 8, !tbaa !11
  %cmp.i1.i.i.i.i = icmp eq i64 %11, 2
  br i1 %cmp.i1.i.i.i.i, label %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEEE10resizeLikeINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEEvRKNS_9EigenBaseIT_EE.exit.i.i, label %if.then.i2.i.i.i.i

if.then.i2.i.i.i.i:                               ; preds = %_ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EE6resizeElll.exit
  %12 = bitcast %"class.Eigen::Matrix.6"* %M to i8**
  %13 = load i8*, i8** %12, align 8, !tbaa !13
  call void @free(i8* %13) #8
  %call.i.i.i.i.i.i.i.i30 = call noalias i8* @malloc(i64 16) #8
  %14 = ptrtoint i8* %call.i.i.i.i.i.i.i.i30 to i64
  %rem.i.i.i.i.i.i.i.i = and i64 %14, 15
  %cmp1.i.i.i.i.i.i.i.i = icmp eq i64 %rem.i.i.i.i.i.i.i.i, 0
  br i1 %cmp1.i.i.i.i.i.i.i.i, label %land.rhs.i.i.i.i.i.i.i.i31, label %cond.false.i.i.i.i.i.i.i.i32

land.rhs.i.i.i.i.i.i.i.i31:                       ; preds = %if.then.i2.i.i.i.i
  store i8* %call.i.i.i.i.i.i.i.i30, i8** %12, align 8, !tbaa !13
  br label %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEEE10resizeLikeINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEEvRKNS_9EigenBaseIT_EE.exit.i.i

cond.false.i.i.i.i.i.i.i.i32:                     ; preds = %if.then.i2.i.i.i.i
  call void @__assert_fail(i8* getelementptr inbounds ([186 x i8], [186 x i8]* @.str.21, i64 0, i64 0), i8* getelementptr inbounds ([49 x i8], [49 x i8]* @.str.22, i64 0, i64 0), i32 161, i8* getelementptr inbounds ([51 x i8], [51 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen8internal14aligned_mallocEm, i64 0, i64 0)) #9
  unreachable

_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEEE10resizeLikeINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEEvRKNS_9EigenBaseIT_EE.exit.i.i: ; preds = %land.rhs.i.i.i.i.i.i.i.i31, %_ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EE6resizeElll.exit
  store i64 2, i64* %m_rows.i.i.i.i.i, align 8, !tbaa !11
  %15 = getelementptr inbounds %"class.Eigen::Matrix.6", %"class.Eigen::Matrix.6"* %M, i64 0, i32 0, i32 0, i32 0
  %16 = bitcast %"class.Eigen::Matrix.6"* %M to <2 x double>**
  %17 = load <2 x double>*, <2 x double>** %16, align 8, !tbaa !13
  store <2 x double> <double 2.000000e+00, double 2.000000e+00>, <2 x double>* %17, align 16, !tbaa !10
  %18 = bitcast %"class.Eigen::Matrix.6"* %O to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %18) #8
  call void @llvm.memset.p0i8.i64(i8* align 8 %18, i8 0, i64 16, i1 false) #8
  %m_rows.i.i.i.i.i110 = getelementptr inbounds %"class.Eigen::Matrix.6", %"class.Eigen::Matrix.6"* %O, i64 0, i32 0, i32 0, i32 1
  %19 = load i64, i64* %m_rows.i.i.i.i.i110, align 8, !tbaa !11
  %cmp.i1.i.i.i.i111 = icmp eq i64 %19, 2
  br i1 %cmp.i1.i.i.i.i111, label %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEEE10resizeLikeINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEEvRKNS_9EigenBaseIT_EE.exit.i.i146, label %if.then.i2.i.i.i.i115

if.then.i2.i.i.i.i115:                            ; preds = %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEEE10resizeLikeINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEEvRKNS_9EigenBaseIT_EE.exit.i.i
  %20 = bitcast %"class.Eigen::Matrix.6"* %O to i8**
  %21 = load i8*, i8** %20, align 8, !tbaa !13
  call void @free(i8* %21) #8
  %call.i.i.i.i.i.i.i.i122 = call noalias i8* @malloc(i64 16) #8
  %22 = ptrtoint i8* %call.i.i.i.i.i.i.i.i122 to i64
  %rem.i.i.i.i.i.i.i.i125 = and i64 %22, 15
  %cmp1.i.i.i.i.i.i.i.i126 = icmp eq i64 %rem.i.i.i.i.i.i.i.i125, 0
  br i1 %cmp1.i.i.i.i.i.i.i.i126, label %land.rhs.i.i.i.i.i.i.i.i131, label %cond.false.i.i.i.i.i.i.i.i132

land.rhs.i.i.i.i.i.i.i.i131:                      ; preds = %if.then.i2.i.i.i.i115
  store i8* %call.i.i.i.i.i.i.i.i122, i8** %20, align 8, !tbaa !13
  br label %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEEE10resizeLikeINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEEvRKNS_9EigenBaseIT_EE.exit.i.i146

cond.false.i.i.i.i.i.i.i.i132:                    ; preds = %if.then.i2.i.i.i.i115
  call void @__assert_fail(i8* getelementptr inbounds ([186 x i8], [186 x i8]* @.str.21, i64 0, i64 0), i8* getelementptr inbounds ([49 x i8], [49 x i8]* @.str.22, i64 0, i64 0), i32 161, i8* getelementptr inbounds ([51 x i8], [51 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen8internal14aligned_mallocEm, i64 0, i64 0)) #9
  unreachable

_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEEE10resizeLikeINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEEvRKNS_9EigenBaseIT_EE.exit.i.i146: ; preds = %land.rhs.i.i.i.i.i.i.i.i131, %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEEE10resizeLikeINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEEvRKNS_9EigenBaseIT_EE.exit.i.i
  store i64 2, i64* %m_rows.i.i.i.i.i110, align 8, !tbaa !11
  %23 = getelementptr inbounds %"class.Eigen::Matrix.6", %"class.Eigen::Matrix.6"* %O, i64 0, i32 0, i32 0, i32 0
  %24 = bitcast %"class.Eigen::Matrix.6"* %O to <2 x double>**
  %25 = load <2 x double>*, <2 x double>** %24, align 8, !tbaa !13
  store <2 x double> zeroinitializer, <2 x double>* %25, align 16, !tbaa !10
  %26 = bitcast %"class.Eigen::Matrix"* %Wp to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %26) #8
  call void @llvm.memset.p0i8.i64(i8* align 8 %26, i8 0, i64 24, i1 false) #8
  %m_rows.i66 = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %Wp, i64 0, i32 0, i32 0, i32 1
  %27 = load i64, i64* %m_rows.i66, align 8, !tbaa !2
  %m_cols.i67 = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %Wp, i64 0, i32 0, i32 0, i32 2
  %28 = load i64, i64* %m_cols.i67, align 8, !tbaa !8
  %mul.i68 = mul nsw i64 %28, %27
  %cmp.i69 = icmp eq i64 %mul.i68, 4
  br i1 %cmp.i69, label %_ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EE6resizeElll.exit96, label %if.then.i72

if.then.i72:                                      ; preds = %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEEE10resizeLikeINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEEvRKNS_9EigenBaseIT_EE.exit.i.i146
  %29 = bitcast %"class.Eigen::Matrix"* %Wp to i8**
  %30 = load i8*, i8** %29, align 8, !tbaa !9
  call void @free(i8* %30) #8
  %call.i.i.i.i79 = call noalias i8* @malloc(i64 32) #8
  %31 = ptrtoint i8* %call.i.i.i.i79 to i64
  %rem.i.i.i.i82 = and i64 %31, 15
  %cmp1.i.i.i.i83 = icmp eq i64 %rem.i.i.i.i82, 0
  br i1 %cmp1.i.i.i.i83, label %land.rhs.i.i.i.i88, label %cond.false.i.i.i.i89

land.rhs.i.i.i.i88:                               ; preds = %if.then.i72
  store i8* %call.i.i.i.i79, i8** %29, align 8, !tbaa !9
  br label %_ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EE6resizeElll.exit96

cond.false.i.i.i.i89:                             ; preds = %if.then.i72
  call void @__assert_fail(i8* getelementptr inbounds ([186 x i8], [186 x i8]* @.str.21, i64 0, i64 0), i8* getelementptr inbounds ([49 x i8], [49 x i8]* @.str.22, i64 0, i64 0), i32 161, i8* getelementptr inbounds ([51 x i8], [51 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen8internal14aligned_mallocEm, i64 0, i64 0)) #9
  unreachable

_ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EE6resizeElll.exit96: ; preds = %land.rhs.i.i.i.i88, %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEEE10resizeLikeINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEEvRKNS_9EigenBaseIT_EE.exit.i.i146
  store i64 2, i64* %m_rows.i66, align 8, !tbaa !2
  store i64 2, i64* %m_cols.i67, align 8, !tbaa !8
  %32 = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %Wp, i64 0, i32 0, i32 0, i32 0
  %33 = load double*, double** %32, align 8, !tbaa !9
  %34 = bitcast double* %33 to <2 x double>*
  store <2 x double> zeroinitializer, <2 x double>* %34, align 16, !tbaa !10
  %arrayidx.i211.1 = getelementptr inbounds double, double* %33, i64 2
  %35 = bitcast double* %arrayidx.i211.1 to <2 x double>*
  store <2 x double> zeroinitializer, <2 x double>* %35, align 16, !tbaa !10
  %36 = bitcast %"class.Eigen::Matrix.6"* %Mp to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %36) #8
  call void @llvm.memset.p0i8.i64(i8* align 8 %36, i8 0, i64 16, i1 false) #8
  %m_rows.i.i.i.i.i350 = getelementptr inbounds %"class.Eigen::Matrix.6", %"class.Eigen::Matrix.6"* %Mp, i64 0, i32 0, i32 0, i32 1
  %37 = load i64, i64* %m_rows.i.i.i.i.i350, align 8, !tbaa !11
  %cmp.i1.i.i.i.i351 = icmp eq i64 %37, 2
  br i1 %cmp.i1.i.i.i.i351, label %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEEE10resizeLikeINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEEvRKNS_9EigenBaseIT_EE.exit.i.i386, label %if.then.i2.i.i.i.i355

if.then.i2.i.i.i.i355:                            ; preds = %_ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EE6resizeElll.exit96
  %38 = bitcast %"class.Eigen::Matrix.6"* %Mp to i8**
  %39 = load i8*, i8** %38, align 8, !tbaa !13
  call void @free(i8* %39) #8
  %call.i.i.i.i.i.i.i.i362 = call noalias i8* @malloc(i64 16) #8
  %40 = ptrtoint i8* %call.i.i.i.i.i.i.i.i362 to i64
  %rem.i.i.i.i.i.i.i.i365 = and i64 %40, 15
  %cmp1.i.i.i.i.i.i.i.i366 = icmp eq i64 %rem.i.i.i.i.i.i.i.i365, 0
  br i1 %cmp1.i.i.i.i.i.i.i.i366, label %land.rhs.i.i.i.i.i.i.i.i371, label %cond.false.i.i.i.i.i.i.i.i372

land.rhs.i.i.i.i.i.i.i.i371:                      ; preds = %if.then.i2.i.i.i.i355
  store i8* %call.i.i.i.i.i.i.i.i362, i8** %38, align 8, !tbaa !13
  br label %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEEE10resizeLikeINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEEvRKNS_9EigenBaseIT_EE.exit.i.i386

cond.false.i.i.i.i.i.i.i.i372:                    ; preds = %if.then.i2.i.i.i.i355
  call void @__assert_fail(i8* getelementptr inbounds ([186 x i8], [186 x i8]* @.str.21, i64 0, i64 0), i8* getelementptr inbounds ([49 x i8], [49 x i8]* @.str.22, i64 0, i64 0), i32 161, i8* getelementptr inbounds ([51 x i8], [51 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen8internal14aligned_mallocEm, i64 0, i64 0)) #9
  unreachable

_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEEE10resizeLikeINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEEvRKNS_9EigenBaseIT_EE.exit.i.i386: ; preds = %land.rhs.i.i.i.i.i.i.i.i371, %_ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EE6resizeElll.exit96
  store i64 2, i64* %m_rows.i.i.i.i.i350, align 8, !tbaa !11
  %41 = getelementptr inbounds %"class.Eigen::Matrix.6", %"class.Eigen::Matrix.6"* %Mp, i64 0, i32 0, i32 0, i32 0
  %42 = bitcast %"class.Eigen::Matrix.6"* %Mp to <2 x double>**
  %43 = load <2 x double>*, <2 x double>** %42, align 8, !tbaa !13
  store <2 x double> zeroinitializer, <2 x double>* %43, align 16, !tbaa !10
  %44 = bitcast %"class.Eigen::Matrix.6"* %Op to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %44) #8
  call void @llvm.memset.p0i8.i64(i8* align 8 %44, i8 0, i64 16, i1 false) #8
  %m_rows.i.i.i.i.i500 = getelementptr inbounds %"class.Eigen::Matrix.6", %"class.Eigen::Matrix.6"* %Op, i64 0, i32 0, i32 0, i32 1
  %45 = load i64, i64* %m_rows.i.i.i.i.i500, align 8, !tbaa !11
  %cmp.i1.i.i.i.i501 = icmp eq i64 %45, 2
  br i1 %cmp.i1.i.i.i.i501, label %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEEE10resizeLikeINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEEvRKNS_9EigenBaseIT_EE.exit.i.i536, label %if.then.i2.i.i.i.i505

if.then.i2.i.i.i.i505:                            ; preds = %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEEE10resizeLikeINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEEvRKNS_9EigenBaseIT_EE.exit.i.i386
  %46 = bitcast %"class.Eigen::Matrix.6"* %Op to i8**
  %47 = load i8*, i8** %46, align 8, !tbaa !13
  call void @free(i8* %47) #8
  %call.i.i.i.i.i.i.i.i512 = call noalias i8* @malloc(i64 16) #8
  %48 = ptrtoint i8* %call.i.i.i.i.i.i.i.i512 to i64
  %rem.i.i.i.i.i.i.i.i515 = and i64 %48, 15
  %cmp1.i.i.i.i.i.i.i.i516 = icmp eq i64 %rem.i.i.i.i.i.i.i.i515, 0
  br i1 %cmp1.i.i.i.i.i.i.i.i516, label %land.rhs.i.i.i.i.i.i.i.i521, label %cond.false.i.i.i.i.i.i.i.i522

land.rhs.i.i.i.i.i.i.i.i521:                      ; preds = %if.then.i2.i.i.i.i505
  store i8* %call.i.i.i.i.i.i.i.i512, i8** %46, align 8, !tbaa !13
  br label %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEEE10resizeLikeINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEEvRKNS_9EigenBaseIT_EE.exit.i.i536

cond.false.i.i.i.i.i.i.i.i522:                    ; preds = %if.then.i2.i.i.i.i505
  call void @__assert_fail(i8* getelementptr inbounds ([186 x i8], [186 x i8]* @.str.21, i64 0, i64 0), i8* getelementptr inbounds ([49 x i8], [49 x i8]* @.str.22, i64 0, i64 0), i32 161, i8* getelementptr inbounds ([51 x i8], [51 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen8internal14aligned_mallocEm, i64 0, i64 0)) #9
  unreachable

_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEEE10resizeLikeINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEEvRKNS_9EigenBaseIT_EE.exit.i.i536: ; preds = %land.rhs.i.i.i.i.i.i.i.i521, %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEEE10resizeLikeINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEEvRKNS_9EigenBaseIT_EE.exit.i.i386
  store i64 2, i64* %m_rows.i.i.i.i.i500, align 8, !tbaa !11
  %49 = getelementptr inbounds %"class.Eigen::Matrix.6", %"class.Eigen::Matrix.6"* %Op, i64 0, i32 0, i32 0, i32 0
  %50 = bitcast %"class.Eigen::Matrix.6"* %Op to <2 x double>**
  %51 = load <2 x double>*, <2 x double>** %50, align 8, !tbaa !13
  store <2 x double> <double 1.000000e+00, double 1.000000e+00>, <2 x double>* %51, align 16, !tbaa !10
  %52 = load i64, i64* %m_rows.i.i.i.i.i500, align 8, !tbaa !11
  %cmp.i.i149 = icmp eq i64 %52, 0
  br i1 %cmp.i.i149, label %_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit.i168, label %if.end.i.i151

if.end.i.i151:                                    ; preds = %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEEE10resizeLikeINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEEvRKNS_9EigenBaseIT_EE.exit.i.i536
  %mul.i.i153 = shl i64 %52, 3
  %call.i.i.i.i154 = call noalias i8* @malloc(i64 %mul.i.i153) #8
  %cmp.i.i.i.i155 = icmp ult i64 %mul.i.i153, 16
  br i1 %cmp.i.i.i.i155, label %_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit.i168, label %lor.lhs.false.i.i.i.i159

lor.lhs.false.i.i.i.i159:                         ; preds = %if.end.i.i151
  %53 = ptrtoint i8* %call.i.i.i.i154 to i64
  %rem.i.i.i.i157 = and i64 %53, 15
  %cmp1.i.i.i.i158 = icmp eq i64 %rem.i.i.i.i157, 0
  br i1 %cmp1.i.i.i.i158, label %_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit.i168, label %cond.false.i.i.i.i164

cond.false.i.i.i.i164:                            ; preds = %lor.lhs.false.i.i.i.i159
  call void @__assert_fail(i8* getelementptr inbounds ([186 x i8], [186 x i8]* @.str.21, i64 0, i64 0), i8* getelementptr inbounds ([49 x i8], [49 x i8]* @.str.22, i64 0, i64 0), i32 161, i8* getelementptr inbounds ([51 x i8], [51 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen8internal14aligned_mallocEm, i64 0, i64 0)) #9
  unreachable

_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit.i168: ; preds = %lor.lhs.false.i.i.i.i159, %if.end.i.i151, %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEEE10resizeLikeINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEEvRKNS_9EigenBaseIT_EE.exit.i.i536
  %54 = phi i8* [ null, %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEEE10resizeLikeINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEEvRKNS_9EigenBaseIT_EE.exit.i.i536 ], [ %call.i.i.i.i154, %lor.lhs.false.i.i.i.i159 ], [ %call.i.i.i.i154, %if.end.i.i151 ]
  %55 = bitcast i8* %54 to double*
  %56 = load i64, i64* %m_rows.i.i.i.i.i500, align 8, !tbaa !11
  %cmp.i.i1.i = icmp eq i64 %56, 0
  br i1 %cmp.i.i1.i, label %_ZN5Eigen12DenseStorageIdLin1ELin1ELi1ELi0EEC2ERKS1_.exit, label %if.end.i.i.i

if.end.i.i.i:                                     ; preds = %_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit.i168
  %add.ptr.i.idx = shl nuw i64 %56, 3
  %57 = bitcast %"class.Eigen::Matrix.6"* %Op to i8**
  %58 = load i8*, i8** %57, align 8, !tbaa !13
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %54, i8* align 8 %58, i64 %add.ptr.i.idx, i1 false) #8
  br label %_ZN5Eigen12DenseStorageIdLin1ELin1ELi1ELi0EEC2ERKS1_.exit

_ZN5Eigen12DenseStorageIdLin1ELin1ELi1ELi0EEC2ERKS1_.exit: ; preds = %if.end.i.i.i, %_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit.i168
  %w = bitcast %"class.Eigen::Matrix"* %W to double**
  %wp = bitcast %"class.Eigen::Matrix"* %Wp to double**
  %m = bitcast %"class.Eigen::Matrix.6"* %M to double**
  %mp = bitcast %"class.Eigen::Matrix.6"* %Mp to double**
  %o = bitcast %"class.Eigen::Matrix.6"* %O to <2 x double>**
  %op = bitcast %"class.Eigen::Matrix.6"* %Op to <2 x double>**
  %wl = load double*, double** %w
  %wpl = load double*, double** %wp
  %ml = load double*, double** %m
  %mpl = load double*, double** %mp
  %ol = load <2 x double>*, <2 x double>** %o
  %opl = load <2 x double>*, <2 x double>** %op
  %call = call double @__enzyme_autodiff(i8* bitcast (void (double*, double*, <2 x double>*)* @_ZL6matvecPKN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEEPKNS0_IdLin1ELi1ELi0ELin1ELi1EEEPS4_ to i8*), double* %wl, double* %wpl, double* %ml, double* %mpl, <2 x double>* %ol, <2 x double>* %opl)
  %59 = load i64, i64* %m_rows.i1, align 8, !tbaa !2
  %cmp2.i = icmp sgt i64 %59, 0
  %60 = load i64, i64* %m_cols.i2, align 8
  %cmp6.i = icmp sgt i64 %60, 0
  %or.cond7 = and i1 %cmp2.i, %cmp6.i
  br i1 %or.cond7, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit, label %cond.false.i

cond.false.i:                                     ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit.149, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit.1, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit, %_ZN5Eigen12DenseStorageIdLin1ELin1ELi1ELi0EEC2ERKS1_.exit
  call void @__assert_fail(i8* getelementptr inbounds ([53 x i8], [53 x i8]* @.str.30, i64 0, i64 0), i8* getelementptr inbounds ([53 x i8], [53 x i8]* @.str.31, i64 0, i64 0), i32 365, i8* getelementptr inbounds ([227 x i8], [227 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll, i64 0, i64 0)) #9
  unreachable

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit: ; preds = %_ZN5Eigen12DenseStorageIdLin1ELin1ELi1ELi0EEC2ERKS1_.exit
  %61 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !14
  %62 = load double*, double** %6, align 8, !tbaa !9
  %63 = load double, double* %62, align 8, !tbaa !15
  %call20 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %61, i8* getelementptr inbounds ([18 x i8], [18 x i8]* @.str, i64 0, i64 0), i32 0, i32 0, double %63) #10
  %64 = load i64, i64* %m_rows.i1, align 8, !tbaa !2
  %cmp2.i.1 = icmp sgt i64 %64, 1
  %65 = load i64, i64* %m_cols.i2, align 8
  %cmp6.i.1 = icmp sgt i64 %65, 0
  %or.cond7.1 = and i1 %cmp2.i.1, %cmp6.i.1
  br i1 %or.cond7.1, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit.1, label %cond.false.i

cond.false.i623:                                  ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit.1.1, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit
  call void @__assert_fail(i8* getelementptr inbounds ([29 x i8], [29 x i8]* @.str.32, i64 0, i64 0), i8* getelementptr inbounds ([53 x i8], [53 x i8]* @.str.31, i64 0, i64 0), i32 425, i8* getelementptr inbounds ([209 x i8], [209 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl, i64 0, i64 0)) #9
  unreachable

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit: ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit.1.1
  %66 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !14
  %67 = load double*, double** %15, align 8, !tbaa !13
  %68 = load double, double* %67, align 8, !tbaa !15
  %call32 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %66, i8* getelementptr inbounds ([12 x i8], [12 x i8]* @.str.1, i64 0, i64 0), i32 0, double %68) #10
  %69 = load i64, i64* %m_rows.i.i.i.i.i, align 8, !tbaa !11
  %cmp2.i621.1 = icmp sgt i64 %69, 1
  br i1 %cmp2.i621.1, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit.1, label %cond.false.i623

cond.false.i631:                                  ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit.1, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit634
  call void @__assert_fail(i8* getelementptr inbounds ([29 x i8], [29 x i8]* @.str.32, i64 0, i64 0), i8* getelementptr inbounds ([53 x i8], [53 x i8]* @.str.31, i64 0, i64 0), i32 425, i8* getelementptr inbounds ([209 x i8], [209 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl, i64 0, i64 0)) #9
  unreachable

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit634: ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit.1
  %70 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !14
  %71 = load double*, double** %23, align 8, !tbaa !13
  %72 = load double, double* %71, align 8, !tbaa !15
  %call44 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %70, i8* getelementptr inbounds ([12 x i8], [12 x i8]* @.str.2, i64 0, i64 0), i32 0, double %72) #10
  %73 = load i64, i64* %m_rows.i.i.i.i.i110, align 8, !tbaa !11
  %cmp2.i629.1 = icmp sgt i64 %73, 1
  br i1 %cmp2.i629.1, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit634.1, label %cond.false.i631

cond.false.i645:                                  ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit634.1, %if.end.142, %if.end.1, %if.end
  call void @__assert_fail(i8* getelementptr inbounds ([53 x i8], [53 x i8]* @.str.30, i64 0, i64 0), i8* getelementptr inbounds ([53 x i8], [53 x i8]* @.str.31, i64 0, i64 0), i32 365, i8* getelementptr inbounds ([227 x i8], [227 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll, i64 0, i64 0)) #9
  unreachable

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit648: ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit634.1
  %74 = load double*, double** %32, align 8, !tbaa !9
  %75 = load double, double* %74, align 8, !tbaa !15
  %76 = load i64, i64* %m_rows.i.i.i.i.i, align 8, !tbaa !11
  %cmp2.i652 = icmp sgt i64 %76, 0
  br i1 %cmp2.i652, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit657, label %cond.false.i654

cond.false.i654:                                  ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit648.1.1, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit648.133, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit648.1, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit648
  call void @__assert_fail(i8* getelementptr inbounds ([29 x i8], [29 x i8]* @.str.32, i64 0, i64 0), i8* getelementptr inbounds ([53 x i8], [53 x i8]* @.str.31, i64 0, i64 0), i32 425, i8* getelementptr inbounds ([209 x i8], [209 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl, i64 0, i64 0)) #9
  unreachable

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit657: ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit648
  %cmp2.i661 = icmp sgt i64 %56, 0
  br i1 %cmp2.i661, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit666, label %cond.false.i663

cond.false.i663:                                  ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit657.1, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit657
  call void @__assert_fail(i8* getelementptr inbounds ([29 x i8], [29 x i8]* @.str.32, i64 0, i64 0), i8* getelementptr inbounds ([53 x i8], [53 x i8]* @.str.31, i64 0, i64 0), i32 425, i8* getelementptr inbounds ([209 x i8], [209 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl, i64 0, i64 0)) #9
  unreachable

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit666: ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit657
  %77 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !14
  %78 = load double*, double** %15, align 8, !tbaa !13
  %79 = load double, double* %78, align 8, !tbaa !15
  %80 = load double, double* %55, align 8, !tbaa !15
  %mul = fmul double %79, %80
  %sub = fsub double %75, %mul
  %81 = call double @llvm.fabs.f64(double %sub)
  %cmp67 = fcmp ogt double %81, 1.000000e-10
  br i1 %cmp67, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit698, label %if.end

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit698: ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit666.1.1, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit666.140, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit666.1, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit666
  %.lcssa15 = phi %struct._IO_FILE* [ %77, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit666 ], [ %131, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit666.1 ], [ %142, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit666.140 ], [ %152, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit666.1.1 ]
  %mul.lcssa = phi double [ %mul, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit666 ], [ %mul.1, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit666.1 ], [ %mul.137, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit666.140 ], [ %mul.1.1, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit666.1.1 ]
  %.lcssa12 = phi double [ %75, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit666 ], [ %129, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit666.1 ], [ %140, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit666.140 ], [ %150, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit666.1.1 ]
  %call76 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %.lcssa15, i8* getelementptr inbounds ([68 x i8], [68 x i8]* @.str.3, i64 0, i64 0), i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str.4, i64 0, i64 0), double %.lcssa12, i8* getelementptr inbounds ([18 x i8], [18 x i8]* @.str.5, i64 0, i64 0), double %mul.lcssa, double 1.000000e-10, i8* getelementptr inbounds ([66 x i8], [66 x i8]* @.str.6, i64 0, i64 0), i32 66, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @__PRETTY_FUNCTION__.main, i64 0, i64 0)) #10
  call void @abort() #9
  unreachable

if.end:                                           ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit666
  %call80 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %77, i8* getelementptr inbounds ([19 x i8], [19 x i8]* @.str.7, i64 0, i64 0), i32 0, i32 0, double %75) #10
  %82 = load i64, i64* %m_rows.i66, align 8, !tbaa !2
  %cmp2.i638.1 = icmp sgt i64 %82, 1
  %83 = load i64, i64* %m_cols.i67, align 8
  %cmp6.i643.1 = icmp sgt i64 %83, 0
  %or.cond8.1 = and i1 %cmp2.i638.1, %cmp6.i643.1
  br i1 %or.cond8.1, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit648.1, label %cond.false.i645

cond.false.i704:                                  ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit739.1.1, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit739.1
  call void @__assert_fail(i8* getelementptr inbounds ([29 x i8], [29 x i8]* @.str.32, i64 0, i64 0), i8* getelementptr inbounds ([53 x i8], [53 x i8]* @.str.31, i64 0, i64 0), i32 425, i8* getelementptr inbounds ([209 x i8], [209 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl, i64 0, i64 0)) #9
  unreachable

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit707: ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit739.1
  %84 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !14
  %85 = load double*, double** %41, align 8, !tbaa !13
  %86 = load double, double* %85, align 8, !tbaa !15
  %sub110 = fsub double %86, %add.1
  %87 = call double @llvm.fabs.f64(double %sub110)
  %cmp111 = fcmp ogt double %87, 1.000000e-10
  br i1 %cmp111, label %if.then112, label %if.end116

cond.false.i727:                                  ; preds = %if.end.1.1, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit739.126, %if.end116, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit739
  call void @__assert_fail(i8* getelementptr inbounds ([53 x i8], [53 x i8]* @.str.30, i64 0, i64 0), i8* getelementptr inbounds ([53 x i8], [53 x i8]* @.str.31, i64 0, i64 0), i32 365, i8* getelementptr inbounds ([227 x i8], [227 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll, i64 0, i64 0)) #9
  unreachable

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit739: ; preds = %if.end.1.1
  %88 = load double*, double** %6, align 8, !tbaa !9
  %cmp2.i720.1 = icmp sgt i64 %157, 1
  br i1 %cmp2.i720.1, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit739.1, label %cond.false.i727

if.then112:                                       ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit707.1, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit707
  %.lcssa8 = phi %struct._IO_FILE* [ %84, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit707 ], [ %123, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit707.1 ]
  %.lcssa6 = phi double [ %86, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit707 ], [ %125, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit707.1 ]
  %add.lcssa.lcssa4 = phi double [ %add.1, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit707 ], [ %add.1.1, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit707.1 ]
  %call115 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %.lcssa8, i8* getelementptr inbounds ([68 x i8], [68 x i8]* @.str.3, i64 0, i64 0), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.8, i64 0, i64 0), double %.lcssa6, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.9, i64 0, i64 0), double %add.lcssa.lcssa4, double 1.000000e-10, i8* getelementptr inbounds ([66 x i8], [66 x i8]* @.str.6, i64 0, i64 0), i32 73, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @__PRETTY_FUNCTION__.main, i64 0, i64 0)) #10
  call void @abort() #9
  unreachable

if.end116:                                        ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit707
  %call119 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %84, i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.10, i64 0, i64 0), i32 0, double %86) #10
  %89 = load i64, i64* %m_rows.i1, align 8, !tbaa !2
  %cmp2.i720.117 = icmp sgt i64 %89, 0
  %90 = load i64, i64* %m_cols.i2, align 8
  %cmp6.i725.118 = icmp sgt i64 %90, 1
  %or.cond10.119 = and i1 %cmp2.i720.117, %cmp6.i725.118
  br i1 %or.cond10.119, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit739.126, label %cond.false.i727

cond.false.i751:                                  ; preds = %if.end116.1, %if.end137
  call void @__assert_fail(i8* getelementptr inbounds ([29 x i8], [29 x i8]* @.str.32, i64 0, i64 0), i8* getelementptr inbounds ([53 x i8], [53 x i8]* @.str.31, i64 0, i64 0), i32 425, i8* getelementptr inbounds ([209 x i8], [209 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl, i64 0, i64 0)) #9
  unreachable

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit754: ; preds = %if.end116.1
  %91 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !14
  %92 = load double*, double** %49, align 8, !tbaa !13
  %93 = load double, double* %92, align 8, !tbaa !15
  %94 = call double @llvm.fabs.f64(double %93)
  %cmp132 = fcmp ogt double %94, 1.000000e-10
  br i1 %cmp132, label %if.then133, label %if.end137

if.then133:                                       ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit754.1, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit754
  %.lcssa2 = phi %struct._IO_FILE* [ %91, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit754 ], [ %96, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit754.1 ]
  %.lcssa = phi double [ %93, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit754 ], [ %98, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit754.1 ]
  %call136 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %.lcssa2, i8* getelementptr inbounds ([68 x i8], [68 x i8]* @.str.3, i64 0, i64 0), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.11, i64 0, i64 0), double %.lcssa, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.12, i64 0, i64 0), double 0.000000e+00, double 1.000000e-10, i8* getelementptr inbounds ([66 x i8], [66 x i8]* @.str.6, i64 0, i64 0), i32 78, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @__PRETTY_FUNCTION__.main, i64 0, i64 0)) #10
  call void @abort() #9
  unreachable

if.end137:                                        ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit754
  %call140 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %91, i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.13, i64 0, i64 0), i32 0, double %93) #10
  %95 = load i64, i64* %m_rows.i.i.i.i.i500, align 8, !tbaa !11
  %cmp2.i749.1 = icmp sgt i64 %95, 1
  br i1 %cmp2.i749.1, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit754.1, label %cond.false.i751

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit754.1: ; preds = %if.end137
  %96 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !14
  %97 = load double*, double** %49, align 8, !tbaa !13
  %arrayidx.i13.1 = getelementptr inbounds double, double* %97, i64 1
  %98 = load double, double* %arrayidx.i13.1, align 8, !tbaa !15
  %99 = call double @llvm.fabs.f64(double %98)
  %cmp132.1 = fcmp ogt double %99, 1.000000e-10
  br i1 %cmp132.1, label %if.then133, label %if.end137.1

if.end137.1:                                      ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit754.1
  %call140.1 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %96, i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.13, i64 0, i64 0), i32 1, double %98) #10
  call void @free(i8* %54) #8
  %100 = bitcast %"class.Eigen::Matrix.6"* %Op to i8**
  %101 = load i8*, i8** %100, align 8, !tbaa !13
  call void @free(i8* %101) #8
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %44) #8
  %102 = bitcast %"class.Eigen::Matrix.6"* %Mp to i8**
  %103 = load i8*, i8** %102, align 8, !tbaa !13
  call void @free(i8* %103) #8
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %36) #8
  %104 = bitcast %"class.Eigen::Matrix"* %Wp to i8**
  %105 = load i8*, i8** %104, align 8, !tbaa !9
  call void @free(i8* %105) #8
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %26) #8
  %106 = bitcast %"class.Eigen::Matrix.6"* %O to i8**
  %107 = load i8*, i8** %106, align 8, !tbaa !13
  call void @free(i8* %107) #8
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %18) #8
  %108 = bitcast %"class.Eigen::Matrix.6"* %M to i8**
  %109 = load i8*, i8** %108, align 8, !tbaa !13
  call void @free(i8* %109) #8
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %10) #8
  %110 = bitcast %"class.Eigen::Matrix"* %W to i8**
  %111 = load i8*, i8** %110, align 8, !tbaa !9
  call void @free(i8* %111) #8
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %0) #8
  ret i32 0

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit739.1: ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit739
  %112 = load double, double* %88, align 8, !tbaa !15
  %113 = load double, double* %55, align 8, !tbaa !15
  %mul104 = fmul double %112, %113
  %add = fadd double %mul104, 0.000000e+00
  %arrayidx.i56.1 = getelementptr inbounds double, double* %88, i64 1
  %114 = load double, double* %arrayidx.i56.1, align 8, !tbaa !15
  %115 = load double, double* %134, align 8, !tbaa !15
  %mul104.1 = fmul double %114, %115
  %add.1 = fadd double %add, %mul104.1
  %116 = load i64, i64* %m_rows.i.i.i.i.i350, align 8, !tbaa !11
  %cmp2.i702 = icmp sgt i64 %116, 0
  br i1 %cmp2.i702, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit707, label %cond.false.i704

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit739.126: ; preds = %if.end116
  %117 = load double*, double** %6, align 8, !tbaa !9
  %cmp2.i720.1.1 = icmp sgt i64 %89, 1
  br i1 %cmp2.i720.1.1, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit739.1.1, label %cond.false.i727

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit739.1.1: ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit739.126
  %arrayidx.i56.123 = getelementptr inbounds double, double* %117, i64 %89
  %118 = load double, double* %arrayidx.i56.123, align 8, !tbaa !15
  %119 = load double, double* %55, align 8, !tbaa !15
  %mul104.124 = fmul double %118, %119
  %add.125 = fadd double %mul104.124, 0.000000e+00
  %add.i.1.1 = add nsw i64 %89, 1
  %arrayidx.i56.1.1 = getelementptr inbounds double, double* %117, i64 %add.i.1.1
  %120 = load double, double* %arrayidx.i56.1.1, align 8, !tbaa !15
  %121 = load double, double* %134, align 8, !tbaa !15
  %mul104.1.1 = fmul double %120, %121
  %add.1.1 = fadd double %add.125, %mul104.1.1
  %122 = load i64, i64* %m_rows.i.i.i.i.i350, align 8, !tbaa !11
  %cmp2.i702.1 = icmp sgt i64 %122, 1
  br i1 %cmp2.i702.1, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit707.1, label %cond.false.i704

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit707.1: ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit739.1.1
  %123 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !14
  %124 = load double*, double** %41, align 8, !tbaa !13
  %arrayidx.i85.1 = getelementptr inbounds double, double* %124, i64 1
  %125 = load double, double* %arrayidx.i85.1, align 8, !tbaa !15
  %sub110.1 = fsub double %125, %add.1.1
  %126 = call double @llvm.fabs.f64(double %sub110.1)
  %cmp111.1 = fcmp ogt double %126, 1.000000e-10
  br i1 %cmp111.1, label %if.then112, label %if.end116.1

if.end116.1:                                      ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit707.1
  %call119.1 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %123, i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.10, i64 0, i64 0), i32 1, double %125) #10
  %127 = load i64, i64* %m_rows.i.i.i.i.i500, align 8, !tbaa !11
  %cmp2.i749 = icmp sgt i64 %127, 0
  br i1 %cmp2.i749, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit754, label %cond.false.i751

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit648.1: ; preds = %if.end
  %128 = load double*, double** %32, align 8, !tbaa !9
  %arrayidx.i194.1 = getelementptr inbounds double, double* %128, i64 1
  %129 = load double, double* %arrayidx.i194.1, align 8, !tbaa !15
  %130 = load i64, i64* %m_rows.i.i.i.i.i, align 8, !tbaa !11
  %cmp2.i652.1 = icmp sgt i64 %130, 0
  br i1 %cmp2.i652.1, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit657.1, label %cond.false.i654

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit657.1: ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit648.1
  %cmp2.i661.1 = icmp sgt i64 %56, 1
  br i1 %cmp2.i661.1, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit666.1, label %cond.false.i663

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit666.1: ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit657.1
  %131 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !14
  %132 = load double*, double** %15, align 8, !tbaa !13
  %133 = load double, double* %132, align 8, !tbaa !15
  %arrayidx.i154.1 = getelementptr inbounds i8, i8* %54, i64 8
  %134 = bitcast i8* %arrayidx.i154.1 to double*
  %135 = load double, double* %134, align 8, !tbaa !15
  %mul.1 = fmul double %133, %135
  %sub.1 = fsub double %129, %mul.1
  %136 = call double @llvm.fabs.f64(double %sub.1)
  %cmp67.1 = fcmp ogt double %136, 1.000000e-10
  br i1 %cmp67.1, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit698, label %if.end.1

if.end.1:                                         ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit666.1
  %call80.1 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %131, i8* getelementptr inbounds ([19 x i8], [19 x i8]* @.str.7, i64 0, i64 0), i32 1, i32 0, double %129) #10
  %137 = load i64, i64* %m_rows.i66, align 8, !tbaa !2
  %cmp2.i638.127 = icmp sgt i64 %137, 0
  %138 = load i64, i64* %m_cols.i67, align 8
  %cmp6.i643.128 = icmp sgt i64 %138, 1
  %or.cond8.129 = and i1 %cmp2.i638.127, %cmp6.i643.128
  br i1 %or.cond8.129, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit648.133, label %cond.false.i645

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit648.133: ; preds = %if.end.1
  %139 = load double*, double** %32, align 8, !tbaa !9
  %arrayidx.i194.131 = getelementptr inbounds double, double* %139, i64 %137
  %140 = load double, double* %arrayidx.i194.131, align 8, !tbaa !15
  %141 = load i64, i64* %m_rows.i.i.i.i.i, align 8, !tbaa !11
  %cmp2.i652.132 = icmp sgt i64 %141, 1
  br i1 %cmp2.i652.132, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit666.140, label %cond.false.i654

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit666.140: ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit648.133
  %142 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !14
  %143 = load double*, double** %15, align 8, !tbaa !13
  %arrayidx.i170.136 = getelementptr inbounds double, double* %143, i64 1
  %144 = load double, double* %arrayidx.i170.136, align 8, !tbaa !15
  %145 = load double, double* %55, align 8, !tbaa !15
  %mul.137 = fmul double %144, %145
  %sub.138 = fsub double %140, %mul.137
  %146 = call double @llvm.fabs.f64(double %sub.138)
  %cmp67.139 = fcmp ogt double %146, 1.000000e-10
  br i1 %cmp67.139, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit698, label %if.end.142

if.end.142:                                       ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit666.140
  %call80.141 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %142, i8* getelementptr inbounds ([19 x i8], [19 x i8]* @.str.7, i64 0, i64 0), i32 0, i32 1, double %140) #10
  %147 = load i64, i64* %m_rows.i66, align 8, !tbaa !2
  %cmp2.i638.1.1 = icmp sgt i64 %147, 1
  %148 = load i64, i64* %m_cols.i67, align 8
  %cmp6.i643.1.1 = icmp sgt i64 %148, 1
  %or.cond8.1.1 = and i1 %cmp2.i638.1.1, %cmp6.i643.1.1
  br i1 %or.cond8.1.1, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit648.1.1, label %cond.false.i645

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit648.1.1: ; preds = %if.end.142
  %149 = load double*, double** %32, align 8, !tbaa !9
  %add.i193.1.1 = add nsw i64 %147, 1
  %arrayidx.i194.1.1 = getelementptr inbounds double, double* %149, i64 %add.i193.1.1
  %150 = load double, double* %arrayidx.i194.1.1, align 8, !tbaa !15
  %151 = load i64, i64* %m_rows.i.i.i.i.i, align 8, !tbaa !11
  %cmp2.i652.1.1 = icmp sgt i64 %151, 1
  br i1 %cmp2.i652.1.1, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit666.1.1, label %cond.false.i654

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit666.1.1: ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit648.1.1
  %152 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !14
  %153 = load double*, double** %15, align 8, !tbaa !13
  %arrayidx.i170.1.1 = getelementptr inbounds double, double* %153, i64 1
  %154 = load double, double* %arrayidx.i170.1.1, align 8, !tbaa !15
  %155 = load double, double* %134, align 8, !tbaa !15
  %mul.1.1 = fmul double %154, %155
  %sub.1.1 = fsub double %150, %mul.1.1
  %156 = call double @llvm.fabs.f64(double %sub.1.1)
  %cmp67.1.1 = fcmp ogt double %156, 1.000000e-10
  br i1 %cmp67.1.1, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit698, label %if.end.1.1

if.end.1.1:                                       ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit666.1.1
  %call80.1.1 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %152, i8* getelementptr inbounds ([19 x i8], [19 x i8]* @.str.7, i64 0, i64 0), i32 1, i32 1, double %150) #10
  %157 = load i64, i64* %m_rows.i1, align 8, !tbaa !2
  %cmp2.i720 = icmp sgt i64 %157, 0
  %158 = load i64, i64* %m_cols.i2, align 8
  %cmp6.i725 = icmp sgt i64 %158, 0
  %or.cond10 = and i1 %cmp2.i720, %cmp6.i725
  br i1 %or.cond10, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit739, label %cond.false.i727

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit634.1: ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit634
  %159 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !14
  %160 = load double*, double** %23, align 8, !tbaa !13
  %arrayidx.i218.1 = getelementptr inbounds double, double* %160, i64 1
  %161 = load double, double* %arrayidx.i218.1, align 8, !tbaa !15
  %call44.1 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %159, i8* getelementptr inbounds ([12 x i8], [12 x i8]* @.str.2, i64 0, i64 0), i32 1, double %161) #10
  %162 = load i64, i64* %m_rows.i66, align 8, !tbaa !2
  %cmp2.i638 = icmp sgt i64 %162, 0
  %163 = load i64, i64* %m_cols.i67, align 8
  %cmp6.i643 = icmp sgt i64 %163, 0
  %or.cond8 = and i1 %cmp2.i638, %cmp6.i643
  br i1 %or.cond8, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit648, label %cond.false.i645

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit.1: ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit
  %164 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !14
  %165 = load double*, double** %15, align 8, !tbaa !13
  %arrayidx.i230.1 = getelementptr inbounds double, double* %165, i64 1
  %166 = load double, double* %arrayidx.i230.1, align 8, !tbaa !15
  %call32.1 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %164, i8* getelementptr inbounds ([12 x i8], [12 x i8]* @.str.1, i64 0, i64 0), i32 1, double %166) #10
  %167 = load i64, i64* %m_rows.i.i.i.i.i110, align 8, !tbaa !11
  %cmp2.i629 = icmp sgt i64 %167, 0
  br i1 %cmp2.i629, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit634, label %cond.false.i631

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit.1: ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit
  %168 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !14
  %169 = load double*, double** %6, align 8, !tbaa !9
  %arrayidx.i246.1 = getelementptr inbounds double, double* %169, i64 1
  %170 = load double, double* %arrayidx.i246.1, align 8, !tbaa !15
  %call20.1 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %168, i8* getelementptr inbounds ([18 x i8], [18 x i8]* @.str, i64 0, i64 0), i32 1, i32 0, double %170) #10
  %171 = load i64, i64* %m_rows.i1, align 8, !tbaa !2
  %cmp2.i.143 = icmp sgt i64 %171, 0
  %172 = load i64, i64* %m_cols.i2, align 8
  %cmp6.i.144 = icmp sgt i64 %172, 1
  %or.cond7.145 = and i1 %cmp2.i.143, %cmp6.i.144
  br i1 %or.cond7.145, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit.149, label %cond.false.i

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit.149: ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit.1
  %173 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !14
  %174 = load double*, double** %6, align 8, !tbaa !9
  %arrayidx.i246.147 = getelementptr inbounds double, double* %174, i64 %171
  %175 = load double, double* %arrayidx.i246.147, align 8, !tbaa !15
  %call20.148 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %173, i8* getelementptr inbounds ([18 x i8], [18 x i8]* @.str, i64 0, i64 0), i32 0, i32 1, double %175) #10
  %176 = load i64, i64* %m_rows.i1, align 8, !tbaa !2
  %cmp2.i.1.1 = icmp sgt i64 %176, 1
  %177 = load i64, i64* %m_cols.i2, align 8
  %cmp6.i.1.1 = icmp sgt i64 %177, 1
  %or.cond7.1.1 = and i1 %cmp2.i.1.1, %cmp6.i.1.1
  br i1 %or.cond7.1.1, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit.1.1, label %cond.false.i

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit.1.1: ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit.149
  %178 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !14
  %179 = load double*, double** %6, align 8, !tbaa !9
  %add.i245.1.1 = add nsw i64 %176, 1
  %arrayidx.i246.1.1 = getelementptr inbounds double, double* %179, i64 %add.i245.1.1
  %180 = load double, double* %arrayidx.i246.1.1, align 8, !tbaa !15
  %call20.1.1 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %178, i8* getelementptr inbounds ([18 x i8], [18 x i8]* @.str, i64 0, i64 0), i32 1, i32 1, double %180) #10
  %181 = load i64, i64* %m_rows.i.i.i.i.i, align 8, !tbaa !11
  %cmp2.i621 = icmp sgt i64 %181, 0
  br i1 %cmp2.i621, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit, label %cond.false.i623
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

declare double @__enzyme_autodiff(i8*, double*, double*, double*, double*, <2 x double>*, <2 x double>*)

; Function Attrs: nounwind uwtable
define internal void @_ZL6matvecPKN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEEPKNS0_IdLin1ELi1ELi0ELin1ELi1EEEPS4_(double* noalias %W, double* noalias %b, <2 x double>* %output) #3 {
entry:
  ; %a3 = load double*, double** %b, align 8, !tbaa !9
  ; %a6 = load double*, double** %W, align 8, !tbaa !13
  ; %a11 = load <2 x double>*, <2 x double>** %output, align 8, !tbaa !13
  %zz = call <2 x double> @mid(double* %W, double* %b)
  store <2 x double> %zz, <2 x double>* %output, align 16, !tbaa !10
  ret void
}

; Function Attrs: alwaysinline nounwind
declare dso_local i32 @fprintf(%struct._IO_FILE* nocapture, i8* nocapture readonly, ...) local_unnamed_addr #4

; Function Attrs: nounwind readnone speculatable
declare double @llvm.fabs.f64(double) #5

; Function Attrs: alwaysinline noreturn nounwind
declare dso_local void @abort() local_unnamed_addr #6

; Function Attrs: alwaysinline noreturn nounwind
declare dso_local void @__assert_fail(i8*, i8*, i32, i8*) local_unnamed_addr #6

; Function Attrs: alwaysinline nounwind
declare dso_local void @free(i8* nocapture) local_unnamed_addr #4

; Function Attrs: alwaysinline nounwind
declare dso_local noalias i8* @malloc(i64) local_unnamed_addr #4

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local <2 x double> @mid(double* noalias %W, double* noalias %b) {
entry:
  %malloccall = tail call i8* @malloc(i64 8)
  %coerce.dive349 = bitcast i8* %malloccall to double**
  %a2 = load double, double* %b, align 8, !tbaa !15
  %vecinit.i.i = insertelement <2 x double> undef, double %a2, i32 0
  %vecinit1.i.i = insertelement <2 x double> %vecinit.i.i, double %a2, i32 1
  store double* %W, double** %coerce.dive349, align 8
  %call365 = call zeroext i1 @last(double** %coerce.dive349)
  %a6 = bitcast double** %coerce.dive349 to <2 x double>**
  %a7 = load <2 x double>*, <2 x double>** %a6, align 8
  %arrayidx.i.i763.1 = getelementptr inbounds double, double* %W, i64 2
  store double* %arrayidx.i.i763.1, double** %coerce.dive349, align 8
  br i1 %call365, label %for.body371, label %for.body388

for.body371:                                      ; preds = %entry
  %a8 = load <2 x double>, <2 x double>* %a7, align 16, !tbaa !10
  br label %for.cond.cleanup404

for.body388:                                      ; preds = %entry
  %a9 = load <2 x double>, <2 x double>* %a7, align 1, !tbaa !10
  br label %for.cond.cleanup404

for.cond.cleanup404:                              ; preds = %for.body388, %for.body371
  %a9.sink = phi <2 x double> [ %a9, %for.body388 ], [ %a8, %for.body371 ]
  %mul.i.i2 = fmul <2 x double> %a9.sink, %vecinit1.i.i
  %add.i.i = fadd <2 x double> %mul.i.i2, zeroinitializer
  %arrayidx.i769.1 = getelementptr inbounds double, double* %b, i64 1
  %a11 = load double, double* %arrayidx.i769.1, align 8, !tbaa !15
  %.cast = bitcast double* %arrayidx.i.i763.1 to <2 x double>*
  %vecinit.i.i.1 = insertelement <2 x double> undef, double %a11, i32 0
  %vecinit1.i.i.1 = insertelement <2 x double> %vecinit.i.i.1, double %a11, i32 1
  %a13 = load <2 x double>, <2 x double>* %.cast, align 16, !tbaa !10
  %mul.i.i4.1 = fmul <2 x double> %a13, %vecinit1.i.i.1
  %add.i.i3.1 = fadd <2 x double> %mul.i.i4.1, %add.i.i
  ret <2 x double> %add.i.i3.1
}

; Function Attrs: norecurse nounwind uwtable
define linkonce_odr dso_local zeroext i1 @last(double** %coerce.dive349) local_unnamed_addr #7 align 2 {
entry:
  %m0 = bitcast double** %coerce.dive349 to i64*

  %m1 = load i64, i64* %m0, align 8, !tbaa !17
  %rem = and i64 %m1, 15
  %cmp = icmp eq i64 %rem, 0
  ret i1 %cmp
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1) #1

attributes #0 = { alwaysinline norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="128" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="128" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { alwaysinline nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { nounwind readnone speculatable }
attributes #6 = { alwaysinline noreturn nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #7 = { norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #8 = { nounwind }
attributes #9 = { noreturn nounwind }
attributes #10 = { cold }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !7, i64 8}
!3 = !{!"_ZTSN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EEE", !4, i64 0, !7, i64 8, !7, i64 16}
!4 = !{!"any pointer", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}
!7 = !{!"long", !5, i64 0}
!8 = !{!3, !7, i64 16}
!9 = !{!3, !4, i64 0}
!10 = !{!5, !5, i64 0}
!11 = !{!12, !7, i64 8}
!12 = !{!"_ZTSN5Eigen12DenseStorageIdLin1ELin1ELi1ELi0EEE", !4, i64 0, !7, i64 8}
!13 = !{!12, !4, i64 0}
!14 = !{!4, !4, i64 0}
!15 = !{!16, !16, i64 0}
!16 = !{!"double", !5, i64 0}
!17 = !{!18, !4, i64 0}
!18 = !{!"_ZTSN5Eigen8internal16BlasVectorMapperIKdlEE", !4, i64 0}


; CHECK: define internal { { <2 x double>*, <2 x double>*, i1, i8*, i8*, <2 x double>, <2 x double> }, <2 x double> } @augmented_mid(double* noalias %W, double* %"W'", double* noalias %b, double* %"b'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = alloca { { <2 x double>*, <2 x double>*, i1, i8*, i8*, <2 x double>, <2 x double> }, <2 x double> }
; CHECK-NEXT:   %1 = getelementptr inbounds { { <2 x double>*, <2 x double>*, i1, i8*, i8*, <2 x double>, <2 x double> }, <2 x double> }, { { <2 x double>*, <2 x double>*, i1, i8*, i8*, <2 x double>, <2 x double> }, <2 x double> }* %0, i32 0, i32 0
; CHECK-NEXT:   %malloccall = tail call noalias nonnull dereferenceable(8) dereferenceable_or_null(8) i8* @malloc(i64 8)
; CHECK-NEXT:   %2 = getelementptr inbounds { <2 x double>*, <2 x double>*, i1, i8*, i8*, <2 x double>, <2 x double> }, { <2 x double>*, <2 x double>*, i1, i8*, i8*, <2 x double>, <2 x double> }* %1, i32 0, i32 4
; CHECK-NEXT:   store i8* %malloccall, i8** %2
; CHECK-NEXT:   %"malloccall'mi" = tail call noalias nonnull dereferenceable(8) dereferenceable_or_null(8) i8* @malloc(i64 8)
; CHECK-NEXT:   %3 = getelementptr inbounds { <2 x double>*, <2 x double>*, i1, i8*, i8*, <2 x double>, <2 x double> }, { <2 x double>*, <2 x double>*, i1, i8*, i8*, <2 x double>, <2 x double> }* %1, i32 0, i32 3
; CHECK-NEXT:   store i8* %"malloccall'mi", i8** %3
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull dereferenceable(8) dereferenceable_or_null(8) %"malloccall'mi", i8 0, i64 8, i1 false)
; CHECK-NEXT:   %"coerce.dive349'ipc" = bitcast i8* %"malloccall'mi" to double**
; CHECK-NEXT:   %coerce.dive349 = bitcast i8* %malloccall to double**
; CHECK-NEXT:   %a2 = load double, double* %b, align 8, !tbaa !15
; CHECK-NEXT:   %vecinit.i.i = insertelement <2 x double> undef, double %a2, i32 0
; CHECK-NEXT:   %vecinit1.i.i = insertelement <2 x double> %vecinit.i.i, double %a2, i32 1
; CHECK-NEXT:   store double* %"W'", double** %"coerce.dive349'ipc", align 8
; CHECK-NEXT:   store double* %W, double** %coerce.dive349, align 8
; CHECK-NEXT:   %call365 = call i1 @augmented_last(double** %coerce.dive349, double** %"coerce.dive349'ipc")
; CHECK-NEXT:   %4 = getelementptr inbounds { <2 x double>*, <2 x double>*, i1, i8*, i8*, <2 x double>, <2 x double> }, { <2 x double>*, <2 x double>*, i1, i8*, i8*, <2 x double>, <2 x double> }* %1, i32 0, i32 2
; CHECK-NEXT:   store i1 %call365, i1* %4
; CHECK-NEXT:   %"a6'ipc" = bitcast double** %"coerce.dive349'ipc" to <2 x double>**
; CHECK-NEXT:   %a6 = bitcast double** %coerce.dive349 to <2 x double>**
; CHECK-NEXT:   %"a7'ipl" = load <2 x double>*, <2 x double>** %"a6'ipc", align 8
; CHECK-NEXT:   %5 = getelementptr inbounds { <2 x double>*, <2 x double>*, i1, i8*, i8*, <2 x double>, <2 x double> }, { <2 x double>*, <2 x double>*, i1, i8*, i8*, <2 x double>, <2 x double> }* %1, i32 0, i32 0
; CHECK-NEXT:   store <2 x double>* %"a7'ipl", <2 x double>** %5
; CHECK-NEXT:   %a7 = load <2 x double>*, <2 x double>** %a6, align 8
; CHECK-NEXT:   %6 = getelementptr inbounds { <2 x double>*, <2 x double>*, i1, i8*, i8*, <2 x double>, <2 x double> }, { <2 x double>*, <2 x double>*, i1, i8*, i8*, <2 x double>, <2 x double> }* %1, i32 0, i32 1
; CHECK-NEXT:   store <2 x double>* %a7, <2 x double>** %6
; CHECK-NEXT:   %"arrayidx.i.i763.1'ipg" = getelementptr inbounds double, double* %"W'", i64 2
; CHECK-NEXT:   %arrayidx.i.i763.1 = getelementptr inbounds double, double* %W, i64 2
; CHECK-NEXT:   store double* %"arrayidx.i.i763.1'ipg", double** %"coerce.dive349'ipc", align 8
; CHECK-NEXT:   store double* %arrayidx.i.i763.1, double** %coerce.dive349, align 8
; CHECK-NEXT:   br i1 %call365, label %for.body371, label %for.body388

; CHECK: for.body371:                                      ; preds = %entry
; CHECK-NEXT:   %a8 = load <2 x double>, <2 x double>* %a7, align 16, !tbaa !10
; CHECK-NEXT:   %7 = getelementptr inbounds { <2 x double>*, <2 x double>*, i1, i8*, i8*, <2 x double>, <2 x double> }, { <2 x double>*, <2 x double>*, i1, i8*, i8*, <2 x double>, <2 x double> }* %1, i32 0, i32 5
; CHECK-NEXT:   store <2 x double> %a8, <2 x double>* %7
; CHECK-NEXT:   br label %for.cond.cleanup404

; CHECK: for.body388:                                      ; preds = %entry
; CHECK-NEXT:   %a9 = load <2 x double>, <2 x double>* %a7, align 1, !tbaa !10
; CHECK-NEXT:   %8 = getelementptr inbounds { <2 x double>*, <2 x double>*, i1, i8*, i8*, <2 x double>, <2 x double> }, { <2 x double>*, <2 x double>*, i1, i8*, i8*, <2 x double>, <2 x double> }* %1, i32 0, i32 6
; CHECK-NEXT:   store <2 x double> %a9, <2 x double>* %8
; CHECK-NEXT:   br label %for.cond.cleanup404

; CHECK: for.cond.cleanup404:                              ; preds = %for.body388, %for.body371
; CHECK-NEXT:   %a9.sink = phi <2 x double> [ %a9, %for.body388 ], [ %a8, %for.body371 ]
; CHECK-NEXT:   %mul.i.i2 = fmul <2 x double> %a9.sink, %vecinit1.i.i
; CHECK-NEXT:   %add.i.i = fadd <2 x double> %mul.i.i2, zeroinitializer
; CHECK-NEXT:   %arrayidx.i769.1 = getelementptr inbounds double, double* %b, i64 1
; CHECK-NEXT:   %a11 = load double, double* %arrayidx.i769.1, align 8, !tbaa !15
; CHECK-NEXT:   %.cast = bitcast double* %arrayidx.i.i763.1 to <2 x double>*
; CHECK-NEXT:   %vecinit.i.i.1 = insertelement <2 x double> undef, double %a11, i32 0
; CHECK-NEXT:   %vecinit1.i.i.1 = insertelement <2 x double> %vecinit.i.i.1, double %a11, i32 1
; CHECK-NEXT:   %a13 = load <2 x double>, <2 x double>* %.cast, align 16, !tbaa !10
; CHECK-NEXT:   %mul.i.i4.1 = fmul <2 x double> %a13, %vecinit1.i.i.1
; CHECK-NEXT:   %add.i.i3.1 = fadd <2 x double> %mul.i.i4.1, %add.i.i
; CHECK-NEXT:   %9 = getelementptr inbounds { { <2 x double>*, <2 x double>*, i1, i8*, i8*, <2 x double>, <2 x double> }, <2 x double> }, { { <2 x double>*, <2 x double>*, i1, i8*, i8*, <2 x double>, <2 x double> }, <2 x double> }* %0, i32 0, i32 1
; CHECK-NEXT:   store <2 x double> %add.i.i3.1, <2 x double>* %9
; CHECK-NEXT:   %10 = load { { <2 x double>*, <2 x double>*, i1, i8*, i8*, <2 x double>, <2 x double> }, <2 x double> }, { { <2 x double>*, <2 x double>*, i1, i8*, i8*, <2 x double>, <2 x double> }, <2 x double> }* %0
; CHECK-NEXT:   ret { { <2 x double>*, <2 x double>*, i1, i8*, i8*, <2 x double>, <2 x double> }, <2 x double> } %10
; CHECK-NEXT: }

; CHECK: define internal void @diffemid(double* noalias %W, double* %"W'", double* noalias %b, double* %"b'", <2 x double> %differeturn, { <2 x double>*, <2 x double>*, i1, i8*, i8*, <2 x double>, <2 x double> } %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %malloccall = extractvalue { <2 x double>*, <2 x double>*, i1, i8*, i8*, <2 x double>, <2 x double> } %tapeArg, 4
; CHECK-NEXT:   %"malloccall'mi" = extractvalue { <2 x double>*, <2 x double>*, i1, i8*, i8*, <2 x double>, <2 x double> } %tapeArg, 3
; CHECK-NEXT:   %"coerce.dive349'ipc" = bitcast i8* %"malloccall'mi" to double**
; CHECK-NEXT:   %coerce.dive349 = bitcast i8* %malloccall to double**
; CHECK-NEXT:   %a2 = load double, double* %b, align 8, !tbaa !15
; CHECK-NEXT:   %vecinit.i.i = insertelement <2 x double> undef, double %a2, i32 0
; CHECK-NEXT:   %vecinit1.i.i = insertelement <2 x double> %vecinit.i.i, double %a2, i32 1
; CHECK-NEXT:   %call365 = extractvalue { <2 x double>*, <2 x double>*, i1, i8*, i8*, <2 x double>, <2 x double> } %tapeArg, 2
; CHECK-NEXT:   %"a7'il_phi" = extractvalue { <2 x double>*, <2 x double>*, i1, i8*, i8*, <2 x double>, <2 x double> } %tapeArg, 0
; CHECK-NEXT:   %"arrayidx.i.i763.1'ipg" = getelementptr inbounds double, double* %"W'", i64 2
; CHECK-NEXT:   %arrayidx.i.i763.1 = getelementptr inbounds double, double* %W, i64 2
; CHECK-NEXT:   %a9 = extractvalue { <2 x double>*, <2 x double>*, i1, i8*, i8*, <2 x double>, <2 x double> } %tapeArg, 6
; CHECK-NEXT:   %a8 = extractvalue { <2 x double>*, <2 x double>*, i1, i8*, i8*, <2 x double>, <2 x double> } %tapeArg, 5
; CHECK-NEXT:   %a9.sink = select i1 %call365, <2 x double> %a8, <2 x double> %a9
; CHECK-NEXT:   %"arrayidx.i769.1'ipg" = getelementptr inbounds double, double* %"b'", i64 1
; CHECK-NEXT:   %arrayidx.i769.1 = getelementptr inbounds double, double* %b, i64 1
; CHECK-NEXT:   %a11 = load double, double* %arrayidx.i769.1, align 8, !tbaa !15
; CHECK-NEXT:   %".cast'ipc" = bitcast double* %"arrayidx.i.i763.1'ipg" to <2 x double>*
; CHECK-NEXT:   %.cast = bitcast double* %arrayidx.i.i763.1 to <2 x double>*
; CHECK-NEXT:   %vecinit.i.i.1 = insertelement <2 x double> undef, double %a11, i32 0
; CHECK-NEXT:   %vecinit1.i.i.1 = insertelement <2 x double> %vecinit.i.i.1, double %a11, i32 1
; CHECK-NEXT:   %a13 = load <2 x double>, <2 x double>* %.cast, align 16, !tbaa !10
; CHECK-NEXT:   %m0diffea13 = fmul fast <2 x double> %differeturn, %vecinit1.i.i.1
; CHECK-NEXT:   %m1diffevecinit1.i.i.1 = fmul fast <2 x double> %differeturn, %a13
; CHECK-NEXT:   %0 = load <2 x double>, <2 x double>* %".cast'ipc", align 16
; CHECK-NEXT:   %1 = fadd fast <2 x double> %0, %m0diffea13
; CHECK-NEXT:   store <2 x double> %1, <2 x double>* %".cast'ipc", align 16
; CHECK-NEXT:   %2 = insertelement <2 x double> %m1diffevecinit1.i.i.1, double 0.000000e+00, i32 1
; CHECK-NEXT:   %3 = extractelement <2 x double> %m1diffevecinit1.i.i.1, i32 1
; CHECK-NEXT:   %4 = extractelement <2 x double> %2, i32 0
; CHECK-NEXT:   %5 = fadd fast double %3, %4
; CHECK-NEXT:   %6 = load double, double* %"arrayidx.i769.1'ipg", align 8
; CHECK-NEXT:   %7 = fadd fast double %6, %5
; CHECK-NEXT:   store double %7, double* %"arrayidx.i769.1'ipg", align 8
; CHECK-NEXT:   %m0diffea9.sink = fmul fast <2 x double> %differeturn, %vecinit1.i.i
; CHECK-NEXT:   %m1diffevecinit1.i.i = fmul fast <2 x double> %differeturn, %a9.sink
; CHECK-NEXT:   %8 = select i1 %call365, <2 x double> zeroinitializer, <2 x double> %m0diffea9.sink
; CHECK-NEXT:   %9 = select i1 %call365, <2 x double> %m0diffea9.sink, <2 x double> zeroinitializer
; CHECK-NEXT:   br i1 %call365, label %invertfor.body371, label %invertfor.body388

; CHECK: invertentry:                                      ; preds = %invertfor.body388, %invertfor.body371
; CHECK-NEXT:   call void @diffelast(double** %coerce.dive349, double** %"coerce.dive349'ipc")
; CHECK-NEXT:   %10 = insertelement <2 x double> %m1diffevecinit1.i.i, double 0.000000e+00, i32 1
; CHECK-NEXT:   %11 = extractelement <2 x double> %m1diffevecinit1.i.i, i32 1
; CHECK-NEXT:   %12 = extractelement <2 x double> %10, i32 0
; CHECK-NEXT:   %13 = fadd fast double %11, %12
; CHECK-NEXT:   %14 = load double, double* %"b'", align 8
; CHECK-NEXT:   %15 = fadd fast double %14, %13
; CHECK-NEXT:   store double %15, double* %"b'", align 8
; CHECK-NEXT:   tail call void @free(i8* nonnull %"malloccall'mi")
; CHECK-NEXT:   ret void

; CHECK: invertfor.body371:                                ; preds = %entry
; CHECK-NEXT:   %16 = load <2 x double>, <2 x double>* %"a7'il_phi", align 16
; CHECK-NEXT:   %17 = fadd fast <2 x double> %16, %9
; CHECK-NEXT:   store <2 x double> %17, <2 x double>* %"a7'il_phi", align 16
; CHECK-NEXT:   br label %invertentry

; CHECK: invertfor.body388:                                ; preds = %entry
; CHECK-NEXT:   %18 = load <2 x double>, <2 x double>* %"a7'il_phi", align 1
; CHECK-NEXT:   %19 = fadd fast <2 x double> %18, %8
; CHECK-NEXT:   store <2 x double> %19, <2 x double>* %"a7'il_phi", align 1
; CHECK-NEXT:   br label %invertentry
; CHECK-NEXT: }