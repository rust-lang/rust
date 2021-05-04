; RUN: if [ %llvmver < 10 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -simplifycfg -dce -S | FileCheck %s; fi
; Note this doesn't run on LLVM 10 as 10 will simplify the cfg to remove a block unlike lower versions
;  The code is still correct but cannot be easily tested in regex
; ModuleID = 'seg.ll'
source_filename = "/home/enzyme/Enzyme/enzyme/test/Integration/simpleeigen-made.cpp"
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

@stderr = external dso_local local_unnamed_addr global %struct._IO_FILE*, align 8
@.str = private unnamed_addr constant [18 x i8] c"W(o=%d, i=%d)=%f\0A\00", align 1
@.str.1 = private unnamed_addr constant [12 x i8] c"M(o=%d)=%f\0A\00", align 1
@.str.2 = private unnamed_addr constant [12 x i8] c"O(i=%d)=%f\0A\00", align 1
@.str.3 = private unnamed_addr constant [68 x i8] c"Assertion Failed: fabs( [%s = %g] - [%s = %g] ) > %g at %s:%d (%s)\0A\00", align 1
@.str.4 = private unnamed_addr constant [9 x i8] c"Wp(i, o)\00", align 1
@.str.5 = private unnamed_addr constant [18 x i8] c"M(o) * Op_orig(i)\00", align 1
@.str.6 = private unnamed_addr constant [65 x i8] c"/home/enzyme/Enzyme/enzyme/test/Integration/simpleeigen-made.cpp\00", align 1
@__PRETTY_FUNCTION__.main = private unnamed_addr constant [23 x i8] c"int main(int, char **)\00", align 1
@.str.7 = private unnamed_addr constant [19 x i8] c"Wp(o=%d, i=%d)=%f\0A\00", align 1
@.str.8 = private unnamed_addr constant [6 x i8] c"Mp(o)\00", align 1
@.str.9 = private unnamed_addr constant [4 x i8] c"res\00", align 1
@.str.10 = private unnamed_addr constant [13 x i8] c"Mp(o=%d)=%f\0A\00", align 1
@.str.11 = private unnamed_addr constant [6 x i8] c"Op(i)\00", align 1
@.str.12 = private unnamed_addr constant [3 x i8] c"0.\00", align 1
@.str.13 = private unnamed_addr constant [13 x i8] c"Op(i=%d)=%f\0A\00", align 1
@.str.14 = private unnamed_addr constant [140 x i8] c"lhs.cols() == rhs.rows() && \22invalid matrix product\22 && \22if you wanted a coeff-wise or a dot product use the respective explicit functions\22\00", align 1
@.str.15 = private unnamed_addr constant [51 x i8] c"/usr/local/include/eigen3/Eigen/src/Core/Product.h\00", align 1
@__PRETTY_FUNCTION__._ZN5Eigen7ProductINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS1_IdLin1ELi1ELi0ELin1ELi1EEELi0EEC2ERKS2_RKS3_ = private unnamed_addr constant [274 x i8] c"Eigen::Product<Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, 0>::Product(const Eigen::Product::Lhs &, const Eigen::Product::Rhs &) [Lhs = Eigen::Matrix<double, -1, -1, 0, -1, -1>, Rhs = Eigen::Matrix<double, -1, 1, 0, -1, 1>, Option = 0]\00", align 1
@.str.16 = private unnamed_addr constant [399 x i8] c"(!(RowsAtCompileTime!=Dynamic) || (rows==RowsAtCompileTime)) && (!(ColsAtCompileTime!=Dynamic) || (cols==ColsAtCompileTime)) && (!(RowsAtCompileTime==Dynamic && MaxRowsAtCompileTime!=Dynamic) || (rows<=MaxRowsAtCompileTime)) && (!(ColsAtCompileTime==Dynamic && MaxColsAtCompileTime!=Dynamic) || (cols<=MaxColsAtCompileTime)) && rows>=0 && cols>=0 && \22Invalid sizes when resizing a matrix or array.\22\00", align 1
@.str.17 = private unnamed_addr constant [59 x i8] c"/usr/local/include/eigen3/Eigen/src/Core/PlainObjectBase.h\00", align 1
@__PRETTY_FUNCTION__._ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEEE6resizeEll = private unnamed_addr constant [156 x i8] c"void Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >::resize(Eigen::Index, Eigen::Index) [Derived = Eigen::Matrix<double, -1, 1, 0, -1, 1>]\00", align 1
@.str.18 = private unnamed_addr constant [186 x i8] c"(size<16 || (std::size_t(result)%16)==0) && \22System's malloc returned an unaligned pointer. Compile with EIGEN_MALLOC_ALREADY_ALIGNED=0 to fallback to handmade alignd memory allocator.\22\00", align 1
@.str.19 = private unnamed_addr constant [55 x i8] c"/usr/local/include/eigen3/Eigen/src/Core/util/Memory.h\00", align 1
@__PRETTY_FUNCTION__._ZN5Eigen8internal14aligned_mallocEm = private unnamed_addr constant [51 x i8] c"void *Eigen::internal::aligned_malloc(std::size_t)\00", align 1
@.str.20 = private unnamed_addr constant [149 x i8] c"rows >= 0 && (RowsAtCompileTime == Dynamic || RowsAtCompileTime == rows) && cols >= 0 && (ColsAtCompileTime == Dynamic || ColsAtCompileTime == cols)\00", align 1
@.str.21 = private unnamed_addr constant [58 x i8] c"/usr/local/include/eigen3/Eigen/src/Core/CwiseNullaryOp.h\00", align 1
@__PRETTY_FUNCTION__._ZN5Eigen14CwiseNullaryOpINS_8internal18scalar_constant_opIdEENS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEEEC2EllRKS3_ = private unnamed_addr constant [282 x i8] c"Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<double>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >::CwiseNullaryOp(Eigen::Index, Eigen::Index, const NullaryOp &) [NullaryOp = Eigen::internal::scalar_constant_op<double>, MatrixType = Eigen::Matrix<double, -1, 1, 0, -1, 1>]\00", align 1
@.str.22 = private unnamed_addr constant [14 x i8] c"v == T(Value)\00", align 1
@.str.23 = private unnamed_addr constant [58 x i8] c"/usr/local/include/eigen3/Eigen/src/Core/util/XprHelper.h\00", align 1
@__PRETTY_FUNCTION__._ZN5Eigen8internal19variable_if_dynamicIlLi1EEC2El = private unnamed_addr constant [92 x i8] c"Eigen::internal::variable_if_dynamic<long, 1>::variable_if_dynamic(T) [T = long, Value = 1]\00", align 1
@.str.24 = private unnamed_addr constant [47 x i8] c"dst.rows() == dstRows && dst.cols() == dstCols\00", align 1
@.str.25 = private unnamed_addr constant [59 x i8] c"/usr/local/include/eigen3/Eigen/src/Core/AssignEvaluator.h\00", align 1
@__PRETTY_FUNCTION__._ZN5Eigen8internal17resize_if_allowedINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EEddEEvRT_RKT0_RKNS0_9assign_opIT1_T2_EE = private unnamed_addr constant [313 x i8] c"void Eigen::internal::resize_if_allowed(DstXprType &, const SrcXprType &, const internal::assign_op<T1, T2> &) [DstXprType = Eigen::Matrix<double, -1, 1, 0, -1, 1>, SrcXprType = Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<double>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >, T1 = double, T2 = double]\00", align 1
@__PRETTY_FUNCTION__._ZN5Eigen8internal19variable_if_dynamicIlLi0EEC2El = private unnamed_addr constant [92 x i8] c"Eigen::internal::variable_if_dynamic<long, 0>::variable_if_dynamic(T) [T = long, Value = 0]\00", align 1
@__PRETTY_FUNCTION__._ZN5Eigen8internal17resize_if_allowedINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEES3_ddEEvRT_RKT0_RKNS0_9assign_opIT1_T2_EE = private unnamed_addr constant [244 x i8] c"void Eigen::internal::resize_if_allowed(DstXprType &, const SrcXprType &, const internal::assign_op<T1, T2> &) [DstXprType = Eigen::Matrix<double, -1, 1, 0, -1, 1>, SrcXprType = Eigen::Matrix<double, -1, 1, 0, -1, 1>, T1 = double, T2 = double]\00", align 1
@__PRETTY_FUNCTION__._ZN5Eigen14CwiseNullaryOpINS_8internal18scalar_constant_opIdEENS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEEC2EllRKS3_ = private unnamed_addr constant [286 x i8] c"Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<double>, Eigen::Matrix<double, -1, -1, 0, -1, -1> >::CwiseNullaryOp(Eigen::Index, Eigen::Index, const NullaryOp &) [NullaryOp = Eigen::internal::scalar_constant_op<double>, MatrixType = Eigen::Matrix<double, -1, -1, 0, -1, -1>]\00", align 1
@__PRETTY_FUNCTION__._ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEE6resizeEll = private unnamed_addr constant [160 x i8] c"void Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >::resize(Eigen::Index, Eigen::Index) [Derived = Eigen::Matrix<double, -1, -1, 0, -1, -1>]\00", align 1
@__PRETTY_FUNCTION__._ZN5Eigen8internal17resize_if_allowedINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EEddEEvRT_RKT0_RKNS0_9assign_opIT1_T2_EE = private unnamed_addr constant [317 x i8] c"void Eigen::internal::resize_if_allowed(DstXprType &, const SrcXprType &, const internal::assign_op<T1, T2> &) [DstXprType = Eigen::Matrix<double, -1, -1, 0, -1, -1>, SrcXprType = Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<double>, Eigen::Matrix<double, -1, -1, 0, -1, -1> >, T1 = double, T2 = double]\00", align 1
@.str.26 = private unnamed_addr constant [39 x i8] c"other.rows() == 1 || other.cols() == 1\00", align 1
@__PRETTY_FUNCTION__._ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEEE10resizeLikeINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEEvRKNS_9EigenBaseIT_EE = private unnamed_addr constant [289 x i8] c"void Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >::resizeLike(const EigenBase<OtherDerived> &) [Derived = Eigen::Matrix<double, -1, 1, 0, -1, 1>, OtherDerived = Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<double>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >]\00", align 1
@.str.27 = private unnamed_addr constant [53 x i8] c"row >= 0 && row < rows() && col >= 0 && col < cols()\00", align 1
@.str.28 = private unnamed_addr constant [59 x i8] c"/usr/local/include/eigen3/Eigen/src/Core/DenseCoeffsBase.h\00", align 1
@__PRETTY_FUNCTION__._ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll = private unnamed_addr constant [227 x i8] c"Eigen::DenseCoeffsBase<type-parameter-0-0, 1>::Scalar &Eigen::DenseCoeffsBase<Eigen::Matrix<double, -1, -1, 0, -1, -1>, 1>::operator()(Eigen::Index, Eigen::Index) [Derived = Eigen::Matrix<double, -1, -1, 0, -1, -1>, Level = 1]\00", align 1
@.str.29 = private unnamed_addr constant [29 x i8] c"index >= 0 && index < size()\00", align 1
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
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %0) #9
  call void @llvm.memset.p0i8.i64(i8* align 8 %0, i8 0, i64 24, i1 false) #9
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
  call void @free(i8* %4) #9
  %call.i.i.i.i7 = call noalias i8* @malloc(i64 32) #9
  %5 = ptrtoint i8* %call.i.i.i.i7 to i64
  %rem.i.i.i.i = and i64 %5, 15
  %cmp1.i.i.i.i9 = icmp eq i64 %rem.i.i.i.i, 0
  br i1 %cmp1.i.i.i.i9, label %cond.end.i.i.i.i12, label %cond.false.i.i.i.i10

cond.false.i.i.i.i10:                             ; preds = %if.then.i
  call void @__assert_fail(i8* getelementptr inbounds ([186 x i8], [186 x i8]* @.str.18, i64 0, i64 0), i8* getelementptr inbounds ([55 x i8], [55 x i8]* @.str.19, i64 0, i64 0), i32 161, i8* getelementptr inbounds ([51 x i8], [51 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen8internal14aligned_mallocEm, i64 0, i64 0)) #10
  unreachable

cond.end.i.i.i.i12:                               ; preds = %if.then.i
  %tobool.i.i.i.i = icmp eq i8* %call.i.i.i.i7, null
  br i1 %tobool.i.i.i.i, label %if.then.i.i.i.i14, label %_ZN5Eigen8internal26conditional_aligned_mallocILb1EEEPvm.exit.i.i

if.then.i.i.i.i14:                                ; preds = %cond.end.i.i.i.i12
  %call.i.i.i.i.i13 = call i8* @_Znwm(i64 -1) #9
  br label %_ZN5Eigen8internal26conditional_aligned_mallocILb1EEEPvm.exit.i.i

_ZN5Eigen8internal26conditional_aligned_mallocILb1EEEPvm.exit.i.i: ; preds = %if.then.i.i.i.i14, %cond.end.i.i.i.i12
  store i8* %call.i.i.i.i7, i8** %3, align 8, !tbaa !9
  br label %_ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EE6resizeElll.exit

_ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EE6resizeElll.exit: ; preds = %_ZN5Eigen8internal26conditional_aligned_mallocILb1EEEPvm.exit.i.i, %entry
  store i64 2, i64* %m_rows.i1, align 8, !tbaa !2
  store i64 2, i64* %m_cols.i2, align 8, !tbaa !8
  %m_data.i.i.i140 = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %W, i64 0, i32 0, i32 0, i32 0
  %6 = load double*, double** %m_data.i.i.i140, align 8, !tbaa !9
  br label %for.body.i.i.i.i.i.i.i

for.body.i.i.i.i.i.i.i:                           ; preds = %for.body.i.i.i.i.i.i.i, %_ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EE6resizeElll.exit
  %index.014.i.i.i.i.i.i.i = phi i64 [ %add1.i.i.i.i.i.i.i, %for.body.i.i.i.i.i.i.i ], [ 0, %_ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EE6resizeElll.exit ]
  %arrayidx.i34 = getelementptr inbounds double, double* %6, i64 %index.014.i.i.i.i.i.i.i
  %7 = bitcast double* %arrayidx.i34 to <2 x double>*
  store <2 x double> <double 3.000000e+00, double 3.000000e+00>, <2 x double>* %7, align 16, !tbaa !10
  %add1.i.i.i.i.i.i.i = add nuw nsw i64 %index.014.i.i.i.i.i.i.i, 2
  %cmp.i3.i.i.i.i.i.i = icmp ult i64 %add1.i.i.i.i.i.i.i, 4
  br i1 %cmp.i3.i.i.i.i.i.i, label %for.body.i.i.i.i.i.i.i, label %_ZN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEC2INS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES1_EEEERKNS_9EigenBaseIT_EE.exit

_ZN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEC2INS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES1_EEEERKNS_9EigenBaseIT_EE.exit: ; preds = %for.body.i.i.i.i.i.i.i
  %8 = bitcast %"class.Eigen::Matrix.6"* %M to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %8) #9
  call void @llvm.memset.p0i8.i64(i8* align 8 %8, i8 0, i64 16, i1 false) #9
  %m_rows.i.i.i.i.i = getelementptr inbounds %"class.Eigen::Matrix.6", %"class.Eigen::Matrix.6"* %M, i64 0, i32 0, i32 0, i32 1
  %9 = load i64, i64* %m_rows.i.i.i.i.i, align 8, !tbaa !11
  %cmp.i1.i.i.i.i = icmp eq i64 %9, 2
  br i1 %cmp.i1.i.i.i.i, label %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEEE10resizeLikeINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEEvRKNS_9EigenBaseIT_EE.exit.i.i, label %if.then.i2.i.i.i.i

if.then.i2.i.i.i.i:                               ; preds = %_ZN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEC2INS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES1_EEEERKNS_9EigenBaseIT_EE.exit
  %10 = bitcast %"class.Eigen::Matrix.6"* %M to i8**
  %11 = load i8*, i8** %10, align 8, !tbaa !13
  call void @free(i8* %11) #9
  %call.i.i.i.i.i.i.i.i29 = call noalias i8* @malloc(i64 16) #9
  %12 = ptrtoint i8* %call.i.i.i.i.i.i.i.i29 to i64
  %rem.i.i.i.i.i.i.i.i = and i64 %12, 15
  %cmp1.i.i.i.i.i.i.i.i = icmp eq i64 %rem.i.i.i.i.i.i.i.i, 0
  br i1 %cmp1.i.i.i.i.i.i.i.i, label %cond.end.i.i.i.i.i.i.i.i31, label %cond.false.i.i.i.i.i.i.i.i30

cond.false.i.i.i.i.i.i.i.i30:                     ; preds = %if.then.i2.i.i.i.i
  call void @__assert_fail(i8* getelementptr inbounds ([186 x i8], [186 x i8]* @.str.18, i64 0, i64 0), i8* getelementptr inbounds ([55 x i8], [55 x i8]* @.str.19, i64 0, i64 0), i32 161, i8* getelementptr inbounds ([51 x i8], [51 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen8internal14aligned_mallocEm, i64 0, i64 0)) #10
  unreachable

cond.end.i.i.i.i.i.i.i.i31:                       ; preds = %if.then.i2.i.i.i.i
  %tobool.i.i.i.i.i.i.i.i = icmp eq i8* %call.i.i.i.i.i.i.i.i29, null
  br i1 %tobool.i.i.i.i.i.i.i.i, label %if.then.i.i.i.i.i.i.i.i, label %_ZN5Eigen8internal26conditional_aligned_mallocILb1EEEPvm.exit.i.i.i.i.i.i

if.then.i.i.i.i.i.i.i.i:                          ; preds = %cond.end.i.i.i.i.i.i.i.i31
  %call.i.i.i.i.i.i.i.i.i32 = call i8* @_Znwm(i64 -1) #9
  br label %_ZN5Eigen8internal26conditional_aligned_mallocILb1EEEPvm.exit.i.i.i.i.i.i

_ZN5Eigen8internal26conditional_aligned_mallocILb1EEEPvm.exit.i.i.i.i.i.i: ; preds = %if.then.i.i.i.i.i.i.i.i, %cond.end.i.i.i.i.i.i.i.i31
  store i8* %call.i.i.i.i.i.i.i.i29, i8** %10, align 8, !tbaa !13
  br label %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEEE10resizeLikeINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEEvRKNS_9EigenBaseIT_EE.exit.i.i

_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEEE10resizeLikeINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEEvRKNS_9EigenBaseIT_EE.exit.i.i: ; preds = %_ZN5Eigen8internal26conditional_aligned_mallocILb1EEEPvm.exit.i.i.i.i.i.i, %_ZN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEC2INS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES1_EEEERKNS_9EigenBaseIT_EE.exit
  store i64 2, i64* %m_rows.i.i.i.i.i, align 8, !tbaa !11
  %m_data.i121 = getelementptr inbounds %"class.Eigen::Matrix.6", %"class.Eigen::Matrix.6"* %M, i64 0, i32 0, i32 0, i32 0
  %13 = bitcast %"class.Eigen::Matrix.6"* %M to <2 x double>**
  %14 = load <2 x double>*, <2 x double>** %13, align 8, !tbaa !13
  store <2 x double> <double 2.000000e+00, double 2.000000e+00>, <2 x double>* %14, align 16, !tbaa !10
  %15 = bitcast %"class.Eigen::Matrix.6"* %O to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %15) #9
  call void @llvm.memset.p0i8.i64(i8* align 8 %15, i8 0, i64 16, i1 false) #9
  %m_rows.i.i.i.i.i107 = getelementptr inbounds %"class.Eigen::Matrix.6", %"class.Eigen::Matrix.6"* %O, i64 0, i32 0, i32 0, i32 1
  %16 = load i64, i64* %m_rows.i.i.i.i.i107, align 8, !tbaa !11
  %cmp.i1.i.i.i.i108 = icmp eq i64 %16, 2
  br i1 %cmp.i1.i.i.i.i108, label %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEEE10resizeLikeINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEEvRKNS_9EigenBaseIT_EE.exit.i.i140, label %if.then.i2.i.i.i.i112

if.then.i2.i.i.i.i112:                            ; preds = %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEEE10resizeLikeINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEEvRKNS_9EigenBaseIT_EE.exit.i.i
  %17 = bitcast %"class.Eigen::Matrix.6"* %O to i8**
  %18 = load i8*, i8** %17, align 8, !tbaa !13
  call void @free(i8* %18) #9
  %call.i.i.i.i.i.i.i.i118 = call noalias i8* @malloc(i64 16) #9
  %19 = ptrtoint i8* %call.i.i.i.i.i.i.i.i118 to i64
  %rem.i.i.i.i.i.i.i.i121 = and i64 %19, 15
  %cmp1.i.i.i.i.i.i.i.i122 = icmp eq i64 %rem.i.i.i.i.i.i.i.i121, 0
  br i1 %cmp1.i.i.i.i.i.i.i.i122, label %cond.end.i.i.i.i.i.i.i.i128, label %cond.false.i.i.i.i.i.i.i.i124

cond.false.i.i.i.i.i.i.i.i124:                    ; preds = %if.then.i2.i.i.i.i112
  call void @__assert_fail(i8* getelementptr inbounds ([186 x i8], [186 x i8]* @.str.18, i64 0, i64 0), i8* getelementptr inbounds ([55 x i8], [55 x i8]* @.str.19, i64 0, i64 0), i32 161, i8* getelementptr inbounds ([51 x i8], [51 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen8internal14aligned_mallocEm, i64 0, i64 0)) #10
  unreachable

cond.end.i.i.i.i.i.i.i.i128:                      ; preds = %if.then.i2.i.i.i.i112
  %tobool.i.i.i.i.i.i.i.i125 = icmp eq i8* %call.i.i.i.i.i.i.i.i118, null
  br i1 %tobool.i.i.i.i.i.i.i.i125, label %if.then.i.i.i.i.i.i.i.i130, label %_ZN5Eigen8internal26conditional_aligned_mallocILb1EEEPvm.exit.i.i.i.i.i.i131

if.then.i.i.i.i.i.i.i.i130:                       ; preds = %cond.end.i.i.i.i.i.i.i.i128
  %call.i.i.i.i.i.i.i.i.i129 = call i8* @_Znwm(i64 -1) #9
  br label %_ZN5Eigen8internal26conditional_aligned_mallocILb1EEEPvm.exit.i.i.i.i.i.i131

_ZN5Eigen8internal26conditional_aligned_mallocILb1EEEPvm.exit.i.i.i.i.i.i131: ; preds = %if.then.i.i.i.i.i.i.i.i130, %cond.end.i.i.i.i.i.i.i.i128
  store i8* %call.i.i.i.i.i.i.i.i118, i8** %17, align 8, !tbaa !13
  br label %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEEE10resizeLikeINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEEvRKNS_9EigenBaseIT_EE.exit.i.i140

_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEEE10resizeLikeINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEEvRKNS_9EigenBaseIT_EE.exit.i.i140: ; preds = %_ZN5Eigen8internal26conditional_aligned_mallocILb1EEEPvm.exit.i.i.i.i.i.i131, %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEEE10resizeLikeINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEEvRKNS_9EigenBaseIT_EE.exit.i.i
  store i64 2, i64* %m_rows.i.i.i.i.i107, align 8, !tbaa !11
  %m_data.i106 = getelementptr inbounds %"class.Eigen::Matrix.6", %"class.Eigen::Matrix.6"* %O, i64 0, i32 0, i32 0, i32 0
  %20 = bitcast %"class.Eigen::Matrix.6"* %O to <2 x double>**
  %21 = load <2 x double>*, <2 x double>** %20, align 8, !tbaa !13
  store <2 x double> zeroinitializer, <2 x double>* %21, align 16, !tbaa !10
  %22 = bitcast %"class.Eigen::Matrix"* %Wp to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %22) #9
  call void @llvm.memset.p0i8.i64(i8* align 8 %22, i8 0, i64 24, i1 false) #9
  %m_rows.i61 = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %Wp, i64 0, i32 0, i32 0, i32 1
  %23 = load i64, i64* %m_rows.i61, align 8, !tbaa !2
  %m_cols.i62 = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %Wp, i64 0, i32 0, i32 0, i32 2
  %24 = load i64, i64* %m_cols.i62, align 8, !tbaa !8
  %mul.i63 = mul nsw i64 %24, %23
  %cmp.i64 = icmp eq i64 %mul.i63, 4
  br i1 %cmp.i64, label %_ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EE6resizeElll.exit89, label %if.then.i67

if.then.i67:                                      ; preds = %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEEE10resizeLikeINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEEvRKNS_9EigenBaseIT_EE.exit.i.i140
  %25 = bitcast %"class.Eigen::Matrix"* %Wp to i8**
  %26 = load i8*, i8** %25, align 8, !tbaa !9
  call void @free(i8* %26) #9
  %call.i.i.i.i73 = call noalias i8* @malloc(i64 32) #9
  %27 = ptrtoint i8* %call.i.i.i.i73 to i64
  %rem.i.i.i.i76 = and i64 %27, 15
  %cmp1.i.i.i.i77 = icmp eq i64 %rem.i.i.i.i76, 0
  br i1 %cmp1.i.i.i.i77, label %cond.end.i.i.i.i83, label %cond.false.i.i.i.i79

cond.false.i.i.i.i79:                             ; preds = %if.then.i67
  call void @__assert_fail(i8* getelementptr inbounds ([186 x i8], [186 x i8]* @.str.18, i64 0, i64 0), i8* getelementptr inbounds ([55 x i8], [55 x i8]* @.str.19, i64 0, i64 0), i32 161, i8* getelementptr inbounds ([51 x i8], [51 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen8internal14aligned_mallocEm, i64 0, i64 0)) #10
  unreachable

cond.end.i.i.i.i83:                               ; preds = %if.then.i67
  %tobool.i.i.i.i80 = icmp eq i8* %call.i.i.i.i73, null
  br i1 %tobool.i.i.i.i80, label %if.then.i.i.i.i85, label %_ZN5Eigen8internal26conditional_aligned_mallocILb1EEEPvm.exit.i.i86

if.then.i.i.i.i85:                                ; preds = %cond.end.i.i.i.i83
  %call.i.i.i.i.i84 = call i8* @_Znwm(i64 -1) #9
  br label %_ZN5Eigen8internal26conditional_aligned_mallocILb1EEEPvm.exit.i.i86

_ZN5Eigen8internal26conditional_aligned_mallocILb1EEEPvm.exit.i.i86: ; preds = %if.then.i.i.i.i85, %cond.end.i.i.i.i83
  store i8* %call.i.i.i.i73, i8** %25, align 8, !tbaa !9
  br label %_ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EE6resizeElll.exit89

_ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EE6resizeElll.exit89: ; preds = %_ZN5Eigen8internal26conditional_aligned_mallocILb1EEEPvm.exit.i.i86, %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEEE10resizeLikeINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEEvRKNS_9EigenBaseIT_EE.exit.i.i140
  store i64 2, i64* %m_rows.i61, align 8, !tbaa !2
  store i64 2, i64* %m_cols.i62, align 8, !tbaa !8
  %m_data.i.i.i84 = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %Wp, i64 0, i32 0, i32 0, i32 0
  %28 = load double*, double** %m_data.i.i.i84, align 8, !tbaa !9
  br label %for.body.i.i.i.i.i.i.i308

for.body.i.i.i.i.i.i.i308:                        ; preds = %for.body.i.i.i.i.i.i.i308, %_ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EE6resizeElll.exit89
  %index.014.i.i.i.i.i.i.i301 = phi i64 [ %add1.i.i.i.i.i.i.i306, %for.body.i.i.i.i.i.i.i308 ], [ 0, %_ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EE6resizeElll.exit89 ]
  %arrayidx.i169 = getelementptr inbounds double, double* %28, i64 %index.014.i.i.i.i.i.i.i301
  %29 = bitcast double* %arrayidx.i169 to <2 x double>*
  store <2 x double> zeroinitializer, <2 x double>* %29, align 16, !tbaa !10
  %add1.i.i.i.i.i.i.i306 = add nuw nsw i64 %index.014.i.i.i.i.i.i.i301, 2
  %cmp.i3.i.i.i.i.i.i307 = icmp ult i64 %add1.i.i.i.i.i.i.i306, 4
  br i1 %cmp.i3.i.i.i.i.i.i307, label %for.body.i.i.i.i.i.i.i308, label %_ZN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEC2INS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES1_EEEERKNS_9EigenBaseIT_EE.exit310

_ZN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEC2INS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES1_EEEERKNS_9EigenBaseIT_EE.exit310: ; preds = %for.body.i.i.i.i.i.i.i308
  %30 = bitcast %"class.Eigen::Matrix.6"* %Mp to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %30) #9
  call void @llvm.memset.p0i8.i64(i8* align 8 %30, i8 0, i64 16, i1 false) #9
  %m_rows.i.i.i.i.i343 = getelementptr inbounds %"class.Eigen::Matrix.6", %"class.Eigen::Matrix.6"* %Mp, i64 0, i32 0, i32 0, i32 1
  %31 = load i64, i64* %m_rows.i.i.i.i.i343, align 8, !tbaa !11
  %cmp.i1.i.i.i.i344 = icmp eq i64 %31, 2
  br i1 %cmp.i1.i.i.i.i344, label %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEEE10resizeLikeINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEEvRKNS_9EigenBaseIT_EE.exit.i.i376, label %if.then.i2.i.i.i.i348

if.then.i2.i.i.i.i348:                            ; preds = %_ZN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEC2INS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES1_EEEERKNS_9EigenBaseIT_EE.exit310
  %32 = bitcast %"class.Eigen::Matrix.6"* %Mp to i8**
  %33 = load i8*, i8** %32, align 8, !tbaa !13
  call void @free(i8* %33) #9
  %call.i.i.i.i.i.i.i.i354 = call noalias i8* @malloc(i64 16) #9
  %34 = ptrtoint i8* %call.i.i.i.i.i.i.i.i354 to i64
  %rem.i.i.i.i.i.i.i.i357 = and i64 %34, 15
  %cmp1.i.i.i.i.i.i.i.i358 = icmp eq i64 %rem.i.i.i.i.i.i.i.i357, 0
  br i1 %cmp1.i.i.i.i.i.i.i.i358, label %cond.end.i.i.i.i.i.i.i.i364, label %cond.false.i.i.i.i.i.i.i.i360

cond.false.i.i.i.i.i.i.i.i360:                    ; preds = %if.then.i2.i.i.i.i348
  call void @__assert_fail(i8* getelementptr inbounds ([186 x i8], [186 x i8]* @.str.18, i64 0, i64 0), i8* getelementptr inbounds ([55 x i8], [55 x i8]* @.str.19, i64 0, i64 0), i32 161, i8* getelementptr inbounds ([51 x i8], [51 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen8internal14aligned_mallocEm, i64 0, i64 0)) #10
  unreachable

cond.end.i.i.i.i.i.i.i.i364:                      ; preds = %if.then.i2.i.i.i.i348
  %tobool.i.i.i.i.i.i.i.i361 = icmp eq i8* %call.i.i.i.i.i.i.i.i354, null
  br i1 %tobool.i.i.i.i.i.i.i.i361, label %if.then.i.i.i.i.i.i.i.i366, label %_ZN5Eigen8internal26conditional_aligned_mallocILb1EEEPvm.exit.i.i.i.i.i.i367

if.then.i.i.i.i.i.i.i.i366:                       ; preds = %cond.end.i.i.i.i.i.i.i.i364
  %call.i.i.i.i.i.i.i.i.i365 = call i8* @_Znwm(i64 -1) #9
  br label %_ZN5Eigen8internal26conditional_aligned_mallocILb1EEEPvm.exit.i.i.i.i.i.i367

_ZN5Eigen8internal26conditional_aligned_mallocILb1EEEPvm.exit.i.i.i.i.i.i367: ; preds = %if.then.i.i.i.i.i.i.i.i366, %cond.end.i.i.i.i.i.i.i.i364
  store i8* %call.i.i.i.i.i.i.i.i354, i8** %32, align 8, !tbaa !13
  br label %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEEE10resizeLikeINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEEvRKNS_9EigenBaseIT_EE.exit.i.i376

_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEEE10resizeLikeINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEEvRKNS_9EigenBaseIT_EE.exit.i.i376: ; preds = %_ZN5Eigen8internal26conditional_aligned_mallocILb1EEEPvm.exit.i.i.i.i.i.i367, %_ZN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEC2INS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES1_EEEERKNS_9EigenBaseIT_EE.exit310
  store i64 2, i64* %m_rows.i.i.i.i.i343, align 8, !tbaa !11
  %m_data.i69 = getelementptr inbounds %"class.Eigen::Matrix.6", %"class.Eigen::Matrix.6"* %Mp, i64 0, i32 0, i32 0, i32 0
  %35 = bitcast %"class.Eigen::Matrix.6"* %Mp to <2 x double>**
  %36 = load <2 x double>*, <2 x double>** %35, align 8, !tbaa !13
  store <2 x double> zeroinitializer, <2 x double>* %36, align 16, !tbaa !10
  %37 = bitcast %"class.Eigen::Matrix.6"* %Op to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %37) #9
  call void @llvm.memset.p0i8.i64(i8* align 8 %37, i8 0, i64 16, i1 false) #9
  %m_rows.i.i.i.i.i487 = getelementptr inbounds %"class.Eigen::Matrix.6", %"class.Eigen::Matrix.6"* %Op, i64 0, i32 0, i32 0, i32 1
  %38 = load i64, i64* %m_rows.i.i.i.i.i487, align 8, !tbaa !11
  %cmp.i1.i.i.i.i488 = icmp eq i64 %38, 2
  br i1 %cmp.i1.i.i.i.i488, label %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEEE10resizeLikeINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEEvRKNS_9EigenBaseIT_EE.exit.i.i520, label %if.then.i2.i.i.i.i492

if.then.i2.i.i.i.i492:                            ; preds = %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEEE10resizeLikeINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEEvRKNS_9EigenBaseIT_EE.exit.i.i376
  %39 = bitcast %"class.Eigen::Matrix.6"* %Op to i8**
  %40 = load i8*, i8** %39, align 8, !tbaa !13
  call void @free(i8* %40) #9
  %call.i.i.i.i.i.i.i.i498 = call noalias i8* @malloc(i64 16) #9
  %41 = ptrtoint i8* %call.i.i.i.i.i.i.i.i498 to i64
  %rem.i.i.i.i.i.i.i.i501 = and i64 %41, 15
  %cmp1.i.i.i.i.i.i.i.i502 = icmp eq i64 %rem.i.i.i.i.i.i.i.i501, 0
  br i1 %cmp1.i.i.i.i.i.i.i.i502, label %cond.end.i.i.i.i.i.i.i.i508, label %cond.false.i.i.i.i.i.i.i.i504

cond.false.i.i.i.i.i.i.i.i504:                    ; preds = %if.then.i2.i.i.i.i492
  call void @__assert_fail(i8* getelementptr inbounds ([186 x i8], [186 x i8]* @.str.18, i64 0, i64 0), i8* getelementptr inbounds ([55 x i8], [55 x i8]* @.str.19, i64 0, i64 0), i32 161, i8* getelementptr inbounds ([51 x i8], [51 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen8internal14aligned_mallocEm, i64 0, i64 0)) #10
  unreachable

cond.end.i.i.i.i.i.i.i.i508:                      ; preds = %if.then.i2.i.i.i.i492
  %tobool.i.i.i.i.i.i.i.i505 = icmp eq i8* %call.i.i.i.i.i.i.i.i498, null
  br i1 %tobool.i.i.i.i.i.i.i.i505, label %if.then.i.i.i.i.i.i.i.i510, label %_ZN5Eigen8internal26conditional_aligned_mallocILb1EEEPvm.exit.i.i.i.i.i.i511

if.then.i.i.i.i.i.i.i.i510:                       ; preds = %cond.end.i.i.i.i.i.i.i.i508
  %call.i.i.i.i.i.i.i.i.i509 = call i8* @_Znwm(i64 -1) #9
  br label %_ZN5Eigen8internal26conditional_aligned_mallocILb1EEEPvm.exit.i.i.i.i.i.i511

_ZN5Eigen8internal26conditional_aligned_mallocILb1EEEPvm.exit.i.i.i.i.i.i511: ; preds = %if.then.i.i.i.i.i.i.i.i510, %cond.end.i.i.i.i.i.i.i.i508
  store i8* %call.i.i.i.i.i.i.i.i498, i8** %39, align 8, !tbaa !13
  br label %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEEE10resizeLikeINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEEvRKNS_9EigenBaseIT_EE.exit.i.i520

_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEEE10resizeLikeINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEEvRKNS_9EigenBaseIT_EE.exit.i.i520: ; preds = %_ZN5Eigen8internal26conditional_aligned_mallocILb1EEEPvm.exit.i.i.i.i.i.i511, %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEEE10resizeLikeINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEEvRKNS_9EigenBaseIT_EE.exit.i.i376
  store i64 2, i64* %m_rows.i.i.i.i.i487, align 8, !tbaa !11
  %m_data.i58 = getelementptr inbounds %"class.Eigen::Matrix.6", %"class.Eigen::Matrix.6"* %Op, i64 0, i32 0, i32 0, i32 0
  %42 = bitcast %"class.Eigen::Matrix.6"* %Op to <2 x double>**
  %43 = load <2 x double>*, <2 x double>** %42, align 8, !tbaa !13
  store <2 x double> <double 1.000000e+00, double 1.000000e+00>, <2 x double>* %43, align 16, !tbaa !10
  %44 = load i64, i64* %m_rows.i.i.i.i.i487, align 8, !tbaa !11
  %cmp.i.i = icmp eq i64 %44, 0
  br i1 %cmp.i.i, label %_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit.i, label %if.end.i.i134

if.end.i.i134:                                    ; preds = %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEEE10resizeLikeINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEEvRKNS_9EigenBaseIT_EE.exit.i.i520
  %cmp.i.i.i133 = icmp ugt i64 %44, 2305843009213693951
  br i1 %cmp.i.i.i133, label %if.then.i.i.i136, label %_ZN5Eigen8internal23check_size_for_overflowIdEEvm.exit.i.i140

if.then.i.i.i136:                                 ; preds = %if.end.i.i134
  %call.i.i.i135 = call i8* @_Znwm(i64 -1) #9
  br label %_ZN5Eigen8internal23check_size_for_overflowIdEEvm.exit.i.i140

_ZN5Eigen8internal23check_size_for_overflowIdEEvm.exit.i.i140: ; preds = %if.then.i.i.i136, %if.end.i.i134
  %mul.i.i137 = shl i64 %44, 3
  %call.i.i.i.i138 = call noalias i8* @malloc(i64 %mul.i.i137) #9
  %cmp.i.i.i.i139 = icmp ult i64 %mul.i.i137, 16
  br i1 %cmp.i.i.i.i139, label %cond.end.i.i.i.i148, label %lor.lhs.false.i.i.i.i143

lor.lhs.false.i.i.i.i143:                         ; preds = %_ZN5Eigen8internal23check_size_for_overflowIdEEvm.exit.i.i140
  %45 = ptrtoint i8* %call.i.i.i.i138 to i64
  %rem.i.i.i.i141 = and i64 %45, 15
  %cmp1.i.i.i.i142 = icmp eq i64 %rem.i.i.i.i141, 0
  br i1 %cmp1.i.i.i.i142, label %cond.end.i.i.i.i148, label %cond.false.i.i.i.i144

cond.false.i.i.i.i144:                            ; preds = %lor.lhs.false.i.i.i.i143
  call void @__assert_fail(i8* getelementptr inbounds ([186 x i8], [186 x i8]* @.str.18, i64 0, i64 0), i8* getelementptr inbounds ([55 x i8], [55 x i8]* @.str.19, i64 0, i64 0), i32 161, i8* getelementptr inbounds ([51 x i8], [51 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen8internal14aligned_mallocEm, i64 0, i64 0)) #10
  unreachable

cond.end.i.i.i.i148:                              ; preds = %lor.lhs.false.i.i.i.i143, %_ZN5Eigen8internal23check_size_for_overflowIdEEvm.exit.i.i140
  %tobool.i.i.i.i145 = icmp eq i8* %call.i.i.i.i138, null
  %tobool2.i.i.i.i146 = icmp ne i64 %mul.i.i137, 0
  %or.cond.i.i.i.i147 = and i1 %tobool2.i.i.i.i146, %tobool.i.i.i.i145
  br i1 %or.cond.i.i.i.i147, label %if.then.i.i.i.i150, label %_ZN5Eigen8internal26conditional_aligned_mallocILb1EEEPvm.exit.i.i151

if.then.i.i.i.i150:                               ; preds = %cond.end.i.i.i.i148
  %call.i.i.i.i.i149 = call i8* @_Znwm(i64 -1) #9
  br label %_ZN5Eigen8internal26conditional_aligned_mallocILb1EEEPvm.exit.i.i151

_ZN5Eigen8internal26conditional_aligned_mallocILb1EEEPvm.exit.i.i151: ; preds = %if.then.i.i.i.i150, %cond.end.i.i.i.i148
  %46 = bitcast i8* %call.i.i.i.i138 to double*
  br label %_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit.i

_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit.i: ; preds = %_ZN5Eigen8internal26conditional_aligned_mallocILb1EEEPvm.exit.i.i151, %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEEE10resizeLikeINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEEvRKNS_9EigenBaseIT_EE.exit.i.i520
  %47 = phi i8* [ %call.i.i.i.i138, %_ZN5Eigen8internal26conditional_aligned_mallocILb1EEEPvm.exit.i.i151 ], [ null, %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEEE10resizeLikeINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEEvRKNS_9EigenBaseIT_EE.exit.i.i520 ]
  %retval.0.i.i = phi double* [ %46, %_ZN5Eigen8internal26conditional_aligned_mallocILb1EEEPvm.exit.i.i151 ], [ null, %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEEE10resizeLikeINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEEvRKNS_9EigenBaseIT_EE.exit.i.i520 ]
  %48 = load i64, i64* %m_rows.i.i.i.i.i487, align 8, !tbaa !11
  %cmp.i.i1.i = icmp eq i64 %48, 0
  br i1 %cmp.i.i1.i, label %_ZN5Eigen12DenseStorageIdLin1ELin1ELi1ELi0EEC2ERKS1_.exit, label %if.end.i.i.i

if.end.i.i.i:                                     ; preds = %_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit.i
  %add.ptr.i.idx = shl nuw i64 %48, 3
  %49 = bitcast %"class.Eigen::Matrix.6"* %Op to i8**
  %50 = load i8*, i8** %49, align 8, !tbaa !13
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %47, i8* align 8 %50, i64 %add.ptr.i.idx, i1 false) #9
  br label %_ZN5Eigen12DenseStorageIdLin1ELin1ELi1ELi0EEC2ERKS1_.exit

_ZN5Eigen12DenseStorageIdLin1ELin1ELi1ELi0EEC2ERKS1_.exit: ; preds = %if.end.i.i.i, %_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit.i
  %call = call double @__enzyme_autodiff(i8* bitcast (void (%"class.Eigen::Matrix"*)* @matvec to i8*), i8* nonnull %0, i8* nonnull %22) #9
  br label %for.cond12.preheader

for.cond12.preheader:                             ; preds = %for.cond.cleanup15, %_ZN5Eigen12DenseStorageIdLin1ELin1ELi1ELi0EEC2ERKS1_.exit
  %indvars.iv250 = phi i64 [ 0, %_ZN5Eigen12DenseStorageIdLin1ELin1ELi1ELi0EEC2ERKS1_.exit ], [ %indvars.iv.next251, %for.cond.cleanup15 ]
  %51 = trunc i64 %indvars.iv250 to i32
  br label %for.body16

for.cond.cleanup15:                               ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit
  %indvars.iv.next251 = add nuw nsw i64 %indvars.iv250, 1
  %exitcond252 = icmp eq i64 %indvars.iv.next251, 2
  br i1 %exitcond252, label %for.body29, label %for.cond12.preheader

for.body16:                                       ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit, %for.cond12.preheader
  %indvars.iv247 = phi i64 [ 0, %for.cond12.preheader ], [ %indvars.iv.next248, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit ]
  %52 = load i64, i64* %m_rows.i1, align 8, !tbaa !2
  %cmp2.i = icmp sgt i64 %52, %indvars.iv247
  %53 = load i64, i64* %m_cols.i2, align 8
  %cmp7.i = icmp sgt i64 %53, %indvars.iv250
  %or.cond1 = and i1 %cmp2.i, %cmp7.i
  br i1 %or.cond1, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit, label %cond.false.i

cond.false.i:                                     ; preds = %for.body16
  call void @__assert_fail(i8* getelementptr inbounds ([53 x i8], [53 x i8]* @.str.27, i64 0, i64 0), i8* getelementptr inbounds ([59 x i8], [59 x i8]* @.str.28, i64 0, i64 0), i32 365, i8* getelementptr inbounds ([227 x i8], [227 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll, i64 0, i64 0)) #10
  unreachable

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit: ; preds = %for.body16
  %54 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !14
  %55 = load double*, double** %m_data.i.i.i140, align 8, !tbaa !9
  %mul.i272 = mul nsw i64 %52, %indvars.iv250
  %add.i273 = add nsw i64 %mul.i272, %indvars.iv247
  %arrayidx.i274 = getelementptr inbounds double, double* %55, i64 %add.i273
  %56 = load double, double* %arrayidx.i274, align 8, !tbaa !15
  %57 = trunc i64 %indvars.iv247 to i32
  %call20 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %54, i8* getelementptr inbounds ([18 x i8], [18 x i8]* @.str, i64 0, i64 0), i32 %57, i32 %51, double %56) #11
  %indvars.iv.next248 = add nuw nsw i64 %indvars.iv247, 1
  %exitcond249 = icmp eq i64 %indvars.iv.next248, 2
  br i1 %exitcond249, label %for.cond.cleanup15, label %for.body16

for.body29:                                       ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit, %for.cond.cleanup15
  %indvars.iv244 = phi i64 [ %indvars.iv.next245, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit ], [ 0, %for.cond.cleanup15 ]
  %58 = load i64, i64* %m_rows.i.i.i.i.i, align 8, !tbaa !11
  %cmp2.i604 = icmp sgt i64 %58, %indvars.iv244
  br i1 %cmp2.i604, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit, label %cond.false.i606

cond.false.i606:                                  ; preds = %for.body29
  call void @__assert_fail(i8* getelementptr inbounds ([29 x i8], [29 x i8]* @.str.29, i64 0, i64 0), i8* getelementptr inbounds ([59 x i8], [59 x i8]* @.str.28, i64 0, i64 0), i32 425, i8* getelementptr inbounds ([209 x i8], [209 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl, i64 0, i64 0)) #10
  unreachable

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit: ; preds = %for.body29
  %59 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !14
  %60 = load double*, double** %m_data.i121, align 8, !tbaa !13
  %arrayidx.i248 = getelementptr inbounds double, double* %60, i64 %indvars.iv244
  %61 = load double, double* %arrayidx.i248, align 8, !tbaa !15
  %62 = trunc i64 %indvars.iv244 to i32
  %call32 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %59, i8* getelementptr inbounds ([12 x i8], [12 x i8]* @.str.1, i64 0, i64 0), i32 %62, double %61) #11
  %indvars.iv.next245 = add nuw nsw i64 %indvars.iv244, 1
  %exitcond246 = icmp eq i64 %indvars.iv.next245, 2
  br i1 %exitcond246, label %for.body41, label %for.body29

for.body41:                                       ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit617, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit
  %indvars.iv241 = phi i64 [ %indvars.iv.next242, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit617 ], [ 0, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit ]
  %63 = load i64, i64* %m_rows.i.i.i.i.i107, align 8, !tbaa !11
  %cmp2.i612 = icmp sgt i64 %63, %indvars.iv241
  br i1 %cmp2.i612, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit617, label %cond.false.i614

cond.false.i614:                                  ; preds = %for.body41
  call void @__assert_fail(i8* getelementptr inbounds ([29 x i8], [29 x i8]* @.str.29, i64 0, i64 0), i8* getelementptr inbounds ([59 x i8], [59 x i8]* @.str.28, i64 0, i64 0), i32 425, i8* getelementptr inbounds ([209 x i8], [209 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl, i64 0, i64 0)) #10
  unreachable

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit617: ; preds = %for.body41
  %64 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !14
  %65 = load double*, double** %m_data.i106, align 8, !tbaa !13
  %arrayidx.i232 = getelementptr inbounds double, double* %65, i64 %indvars.iv241
  %66 = load double, double* %arrayidx.i232, align 8, !tbaa !15
  %67 = trunc i64 %indvars.iv241 to i32
  %call44 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %64, i8* getelementptr inbounds ([12 x i8], [12 x i8]* @.str.2, i64 0, i64 0), i32 %67, double %66) #11
  %indvars.iv.next242 = add nuw nsw i64 %indvars.iv241, 1
  %exitcond243 = icmp eq i64 %indvars.iv.next242, 2
  br i1 %exitcond243, label %for.cond55.preheader, label %for.body41

for.cond55.preheader:                             ; preds = %for.cond.cleanup58, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit617
  %indvars.iv239 = phi i64 [ %indvars.iv.next240, %for.cond.cleanup58 ], [ 0, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit617 ]
  %68 = trunc i64 %indvars.iv239 to i32
  br label %for.body59

for.cond.cleanup58:                               ; preds = %if.end
  %indvars.iv.next240 = add nuw nsw i64 %indvars.iv239, 1
  %cmp51 = icmp ult i64 %indvars.iv.next240, 2
  br i1 %cmp51, label %for.cond55.preheader, label %for.cond94.preheader

for.body59:                                       ; preds = %if.end, %for.cond55.preheader
  %indvars.iv237 = phi i64 [ 0, %for.cond55.preheader ], [ %indvars.iv.next238, %if.end ]
  %69 = load i64, i64* %m_rows.i61, align 8, !tbaa !2
  %cmp2.i621 = icmp sgt i64 %69, %indvars.iv237
  %70 = load i64, i64* %m_cols.i62, align 8
  %cmp7.i626 = icmp sgt i64 %70, %indvars.iv239
  %or.cond2 = and i1 %cmp2.i621, %cmp7.i626
  br i1 %or.cond2, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit631, label %cond.false.i628

cond.false.i628:                                  ; preds = %for.body59
  call void @__assert_fail(i8* getelementptr inbounds ([53 x i8], [53 x i8]* @.str.27, i64 0, i64 0), i8* getelementptr inbounds ([59 x i8], [59 x i8]* @.str.28, i64 0, i64 0), i32 365, i8* getelementptr inbounds ([227 x i8], [227 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll, i64 0, i64 0)) #10
  unreachable

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit631: ; preds = %for.body59
  %71 = load double*, double** %m_data.i.i.i84, align 8, !tbaa !9
  %mul.i214 = mul nsw i64 %69, %indvars.iv239
  %add.i215 = add nsw i64 %mul.i214, %indvars.iv237
  %arrayidx.i216 = getelementptr inbounds double, double* %71, i64 %add.i215
  %72 = load double, double* %arrayidx.i216, align 8, !tbaa !15
  %73 = load i64, i64* %m_rows.i.i.i.i.i, align 8, !tbaa !11
  %cmp2.i635 = icmp sgt i64 %73, %indvars.iv239
  br i1 %cmp2.i635, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit640, label %cond.false.i637

cond.false.i637:                                  ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit631
  call void @__assert_fail(i8* getelementptr inbounds ([29 x i8], [29 x i8]* @.str.29, i64 0, i64 0), i8* getelementptr inbounds ([59 x i8], [59 x i8]* @.str.28, i64 0, i64 0), i32 425, i8* getelementptr inbounds ([209 x i8], [209 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl, i64 0, i64 0)) #10
  unreachable

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit640: ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit631
  %cmp2.i644 = icmp sgt i64 %48, %indvars.iv237
  br i1 %cmp2.i644, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit649, label %cond.false.i646

cond.false.i646:                                  ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit640
  call void @__assert_fail(i8* getelementptr inbounds ([29 x i8], [29 x i8]* @.str.29, i64 0, i64 0), i8* getelementptr inbounds ([59 x i8], [59 x i8]* @.str.28, i64 0, i64 0), i32 425, i8* getelementptr inbounds ([209 x i8], [209 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl, i64 0, i64 0)) #10
  unreachable

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit649: ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit640
  %74 = load double*, double** %m_data.i121, align 8, !tbaa !13
  %arrayidx.i192 = getelementptr inbounds double, double* %74, i64 %indvars.iv239
  %75 = load double, double* %arrayidx.i192, align 8, !tbaa !15
  %arrayidx.i180 = getelementptr inbounds double, double* %retval.0.i.i, i64 %indvars.iv237
  %76 = load double, double* %arrayidx.i180, align 8, !tbaa !15
  %mul = fmul double %75, %76
  %sub = fsub double %72, %mul
  %77 = call double @llvm.fabs.f64(double %sub)
  %cmp67 = fcmp ogt double %77, 1.000000e-10
  %78 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !14
  br i1 %cmp67, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit681, label %if.end

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit681: ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit649
  %call76 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %78, i8* getelementptr inbounds ([68 x i8], [68 x i8]* @.str.3, i64 0, i64 0), i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str.4, i64 0, i64 0), double %72, i8* getelementptr inbounds ([18 x i8], [18 x i8]* @.str.5, i64 0, i64 0), double %mul, double 1.000000e-10, i8* getelementptr inbounds ([65 x i8], [65 x i8]* @.str.6, i64 0, i64 0), i32 65, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @__PRETTY_FUNCTION__.main, i64 0, i64 0)) #11
  call void @abort() #10
  unreachable

if.end:                                           ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit649
  %79 = trunc i64 %indvars.iv237 to i32
  %call80 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %78, i8* getelementptr inbounds ([19 x i8], [19 x i8]* @.str.7, i64 0, i64 0), i32 %79, i32 %68, double %72) #11
  %indvars.iv.next238 = add nuw nsw i64 %indvars.iv237, 1
  %cmp57 = icmp ult i64 %indvars.iv.next238, 2
  br i1 %cmp57, label %for.body59, label %for.cond.cleanup58

for.cond94.preheader:                             ; preds = %if.end116, %for.cond.cleanup58
  %indvars.iv235 = phi i64 [ %indvars.iv.next236, %if.end116 ], [ 0, %for.cond.cleanup58 ]
  br label %for.body98

land.lhs.true.i686:                               ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit722
  %80 = load i64, i64* %m_rows.i.i.i.i.i343, align 8, !tbaa !11
  %cmp2.i685 = icmp sgt i64 %80, %indvars.iv235
  br i1 %cmp2.i685, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit690, label %cond.false.i687

cond.false.i687:                                  ; preds = %land.lhs.true.i686
  call void @__assert_fail(i8* getelementptr inbounds ([29 x i8], [29 x i8]* @.str.29, i64 0, i64 0), i8* getelementptr inbounds ([59 x i8], [59 x i8]* @.str.28, i64 0, i64 0), i32 425, i8* getelementptr inbounds ([209 x i8], [209 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl, i64 0, i64 0)) #10
  unreachable

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit690: ; preds = %land.lhs.true.i686
  %81 = load double*, double** %m_data.i69, align 8, !tbaa !13
  %arrayidx.i94 = getelementptr inbounds double, double* %81, i64 %indvars.iv235
  %82 = load double, double* %arrayidx.i94, align 8, !tbaa !15
  %sub110 = fsub double %82, %add
  %83 = call double @llvm.fabs.f64(double %sub110)
  %cmp111 = fcmp ogt double %83, 1.000000e-10
  %84 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !14
  br i1 %cmp111, label %if.then112, label %if.end116

for.body98:                                       ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit722, %for.cond94.preheader
  %indvars.iv233 = phi i64 [ 0, %for.cond94.preheader ], [ %indvars.iv.next234, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit722 ]
  %res.0208 = phi double [ 0.000000e+00, %for.cond94.preheader ], [ %add, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit722 ]
  %85 = load i64, i64* %m_rows.i1, align 8, !tbaa !2
  %cmp2.i703 = icmp sgt i64 %85, %indvars.iv233
  %86 = load i64, i64* %m_cols.i2, align 8
  %cmp7.i708 = icmp sgt i64 %86, %indvars.iv235
  %or.cond = and i1 %cmp2.i703, %cmp7.i708
  br i1 %or.cond, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit713, label %cond.false.i710

cond.false.i710:                                  ; preds = %for.body98
  call void @__assert_fail(i8* getelementptr inbounds ([53 x i8], [53 x i8]* @.str.27, i64 0, i64 0), i8* getelementptr inbounds ([59 x i8], [59 x i8]* @.str.28, i64 0, i64 0), i32 365, i8* getelementptr inbounds ([227 x i8], [227 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll, i64 0, i64 0)) #10
  unreachable

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit713: ; preds = %for.body98
  %cmp2.i717 = icmp sgt i64 %48, %indvars.iv233
  br i1 %cmp2.i717, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit722, label %cond.false.i719

cond.false.i719:                                  ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit713
  call void @__assert_fail(i8* getelementptr inbounds ([29 x i8], [29 x i8]* @.str.29, i64 0, i64 0), i8* getelementptr inbounds ([59 x i8], [59 x i8]* @.str.28, i64 0, i64 0), i32 425, i8* getelementptr inbounds ([209 x i8], [209 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl, i64 0, i64 0)) #10
  unreachable

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit722: ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit713
  %87 = load double*, double** %m_data.i.i.i140, align 8, !tbaa !9
  %mul.i61 = mul nsw i64 %85, %indvars.iv235
  %add.i = add nsw i64 %mul.i61, %indvars.iv233
  %arrayidx.i62 = getelementptr inbounds double, double* %87, i64 %add.i
  %88 = load double, double* %arrayidx.i62, align 8, !tbaa !15
  %arrayidx.i44 = getelementptr inbounds double, double* %retval.0.i.i, i64 %indvars.iv233
  %89 = load double, double* %arrayidx.i44, align 8, !tbaa !15
  %mul104 = fmul double %88, %89
  %add = fadd double %res.0208, %mul104
  %indvars.iv.next234 = add nuw nsw i64 %indvars.iv233, 1
  %exitcond = icmp eq i64 %indvars.iv.next234, 2
  br i1 %exitcond, label %land.lhs.true.i686, label %for.body98

if.then112:                                       ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit690
  %call115 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %84, i8* getelementptr inbounds ([68 x i8], [68 x i8]* @.str.3, i64 0, i64 0), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.8, i64 0, i64 0), double %82, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.9, i64 0, i64 0), double %add, double 1.000000e-10, i8* getelementptr inbounds ([65 x i8], [65 x i8]* @.str.6, i64 0, i64 0), i32 72, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @__PRETTY_FUNCTION__.main, i64 0, i64 0)) #11
  call void @abort() #10
  unreachable

if.end116:                                        ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit690
  %90 = trunc i64 %indvars.iv235 to i32
  %call119 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %84, i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.10, i64 0, i64 0), i32 %90, double %82) #11
  %indvars.iv.next236 = add nuw nsw i64 %indvars.iv235, 1
  %cmp90 = icmp ult i64 %indvars.iv.next236, 2
  br i1 %cmp90, label %for.cond94.preheader, label %for.body128

for.cond.cleanup127:                              ; preds = %if.end137
  call void @free(i8* %47) #9
  %91 = bitcast %"class.Eigen::Matrix.6"* %Op to i8**
  %92 = load i8*, i8** %91, align 8, !tbaa !13
  call void @free(i8* %92) #9
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %37) #9
  %93 = bitcast %"class.Eigen::Matrix.6"* %Mp to i8**
  %94 = load i8*, i8** %93, align 8, !tbaa !13
  call void @free(i8* %94) #9
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %30) #9
  %95 = bitcast %"class.Eigen::Matrix"* %Wp to i8**
  %96 = load i8*, i8** %95, align 8, !tbaa !9
  call void @free(i8* %96) #9
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %22) #9
  %97 = bitcast %"class.Eigen::Matrix.6"* %O to i8**
  %98 = load i8*, i8** %97, align 8, !tbaa !13
  call void @free(i8* %98) #9
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %15) #9
  %99 = bitcast %"class.Eigen::Matrix.6"* %M to i8**
  %100 = load i8*, i8** %99, align 8, !tbaa !13
  call void @free(i8* %100) #9
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %8) #9
  %101 = bitcast %"class.Eigen::Matrix"* %W to i8**
  %102 = load i8*, i8** %101, align 8, !tbaa !9
  call void @free(i8* %102) #9
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %0) #9
  ret i32 0

for.body128:                                      ; preds = %if.end137, %if.end116
  %indvars.iv = phi i64 [ %indvars.iv.next, %if.end137 ], [ 0, %if.end116 ]
  %103 = load i64, i64* %m_rows.i.i.i.i.i487, align 8, !tbaa !11
  %cmp2.i732 = icmp sgt i64 %103, %indvars.iv
  br i1 %cmp2.i732, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit737, label %cond.false.i734

cond.false.i734:                                  ; preds = %for.body128
  call void @__assert_fail(i8* getelementptr inbounds ([29 x i8], [29 x i8]* @.str.29, i64 0, i64 0), i8* getelementptr inbounds ([59 x i8], [59 x i8]* @.str.28, i64 0, i64 0), i32 425, i8* getelementptr inbounds ([209 x i8], [209 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl, i64 0, i64 0)) #10
  unreachable

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit737: ; preds = %for.body128
  %104 = load double*, double** %m_data.i58, align 8, !tbaa !13
  %arrayidx.i15 = getelementptr inbounds double, double* %104, i64 %indvars.iv
  %105 = load double, double* %arrayidx.i15, align 8, !tbaa !15
  %106 = call double @llvm.fabs.f64(double %105)
  %cmp132 = fcmp ogt double %106, 1.000000e-10
  %107 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !14
  br i1 %cmp132, label %if.then133, label %if.end137

if.then133:                                       ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit737
  %call136 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %107, i8* getelementptr inbounds ([68 x i8], [68 x i8]* @.str.3, i64 0, i64 0), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.11, i64 0, i64 0), double %105, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.12, i64 0, i64 0), double 0.000000e+00, double 1.000000e-10, i8* getelementptr inbounds ([65 x i8], [65 x i8]* @.str.6, i64 0, i64 0), i32 77, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @__PRETTY_FUNCTION__.main, i64 0, i64 0)) #11
  call void @abort() #10
  unreachable

if.end137:                                        ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit737
  %108 = trunc i64 %indvars.iv to i32
  %call140 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %107, i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.13, i64 0, i64 0), i32 %108, double %105) #11
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %cmp126 = icmp ult i64 %indvars.iv.next, 2
  br i1 %cmp126, label %for.body128, label %for.cond.cleanup127
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

; Function Attrs: alwaysinline
declare dso_local double @__enzyme_autodiff(i8*, i8*, i8*) local_unnamed_addr #2

; Function Attrs: alwaysinline nounwind uwtable
define internal void @matvec(%"class.Eigen::Matrix"* noalias %W) #3 {
entry:
  %m_rows.i19 = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %W, i64 0, i32 0, i32 0, i32 1
  %a8 = load i64, i64* %m_rows.i19, align 8, !tbaa !2
  %m_data.i17 = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %W, i64 0, i32 0, i32 0, i32 0
  %a9 = load double*, double** %m_data.i17, align 8, !tbaa !9
  %subcall = call i64 @sub(double* %a9, i64 %a8) #9
  %mvcond = icmp slt i64 %subcall, 0
  br i1 %mvcond, label %one, label %two

one:                                              ; preds = %entry
  store double 1.000000e+00, double* %a9, align 8, !tbaa !15
  ret void

two:                                              ; preds = %entry
  ret void
}

; Function Attrs: nounwind
declare dso_local i32 @fprintf(%struct._IO_FILE* nocapture, i8* nocapture readonly, ...) local_unnamed_addr #4

; Function Attrs: nounwind readnone speculatable
declare double @llvm.fabs.f64(double) #5

; Function Attrs: noreturn nounwind
declare dso_local void @abort() local_unnamed_addr #6

; Function Attrs: noreturn nounwind
declare dso_local void @__assert_fail(i8*, i8*, i32, i8*) local_unnamed_addr #6

; Function Attrs: nobuiltin
declare dso_local noalias nonnull i8* @_Znwm(i64) local_unnamed_addr #7

; Function Attrs: nounwind
declare dso_local void @free(i8* nocapture) local_unnamed_addr #4

; Function Attrs: nounwind
declare dso_local noalias i8* @malloc(i64) local_unnamed_addr #4

; Function Attrs: inlinehint norecurse nounwind uwtable
define linkonce_odr dso_local i64 @metasub(double* %array, i64 %metasize) local_unnamed_addr #8 {
entry:
  %finalcall = tail call i64 @final(double* %array, i64 %metasize)
  ret i64 %finalcall
}

define linkonce_odr dso_local i64 @sub(double* %in, i64 %subsize) local_unnamed_addr {
entry:
  %0 = ptrtoint double* %in to i64
  %tobool = icmp eq i64 %0, 0
  br i1 %tobool, label %if.end, label %return

if.end:                                           ; preds = %entry
  %metacall = tail call i64 @metasub(double* %in, i64 %subsize)
  br label %return

return:                                           ; preds = %if.end, %entry
  %subret = phi i64 [ %metacall, %if.end ], [ -1, %entry ]
  ret i64 %subret
}

; Function Attrs: inlinehint norecurse nounwind uwtable
define linkonce_odr dso_local i64 @final(double* %array, i64 %finalsize) local_unnamed_addr #8 {
entry:
  %0 = ptrtoint double* %array to i64
  %and = and i64 %0, 7
  %tobool = icmp eq i64 %and, 0
  br i1 %tobool, label %if.else, label %cleanup

if.else:                                          ; preds = %entry
  %div = lshr i64 %0, 3
  %and2 = and i64 %div, 1
  %cmp = icmp slt i64 %and2, %finalsize
  %cond = select i1 %cmp, i64 %and2, i64 %finalsize
  br label %cleanup

cleanup:                                          ; preds = %if.else, %entry
  %finalret = phi i64 [ %cond, %if.else ], [ %finalsize, %entry ]
  ret i64 %finalret
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1) #1

attributes #0 = { alwaysinline norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="128" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { alwaysinline "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { alwaysinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="128" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { nounwind readnone speculatable }
attributes #6 = { noreturn nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #7 = { nobuiltin "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #8 = { inlinehint norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #9 = { nounwind }
attributes #10 = { noreturn nounwind }
attributes #11 = { cold }

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


; CHECK: define internal void @diffematvec(%"class.Eigen::Matrix"* noalias %W, %"class.Eigen::Matrix"* %"W'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %m_rows.i19 = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %W, i64 0, i32 0, i32 0, i32 1
; CHECK-NEXT:   %a8 = load i64, i64* %m_rows.i19, align 8, !tbaa !2
; CHECK-NEXT:   %[[m_datai17ipge:.+]] = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %"W'", i64 0, i32 0, i32 0, i32 0
; CHECK-NEXT:   %m_data.i17 = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %W, i64 0, i32 0, i32 0, i32 0
; CHECK-NEXT:   %"a9'ipl" = load double*, double** %[[m_datai17ipge:.+]], align 8
; CHECK-NEXT:   %a9 = load double*, double** %m_data.i17, align 8, !tbaa !9
; CHECK-NEXT:   %subcall = call i64 @augmented_sub(double* %a9, double* %"a9'ipl", i64 %a8)
; CHECK-NEXT:   %mvcond = icmp slt i64 %subcall, 0
; CHECK-NEXT:   br i1 %mvcond, label %one, label %invertentry

; CHECK: one:                                              ; preds = %entry
; CHECK-NEXT:   store double 1.000000e+00, double* %a9, align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"a9'ipl", align 8
; CHECK-NEXT:   br label %invertentry

; CHECK: invertentry:                                      ; preds = %entry, %one
; CHECK-NEXT:   call void @diffesub(double* %a9, double* %"a9'ipl", i64 %a8)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal i64 @augmented_final(double* %array, double* %"array'", i64 %finalsize)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = ptrtoint double* %array to i64
; CHECK-NEXT:   %and = and i64 %0, 7
; CHECK-NEXT:   %tobool = icmp eq i64 %and, 0
; CHECK-NEXT:   br i1 %tobool, label %[[ifelse:.+]], label %cleanup

; CHECK: [[ifelse]]:                                          ; preds = %entry
; CHECK-NEXT:   %div = lshr i64 %0, 3
; CHECK-NEXT:   %and2 = and i64 %div, 1
; CHECK-NEXT:   %cmp = icmp slt i64 %and2, %finalsize
; CHECK-NEXT:   %cond = select i1 %cmp, i64 %and2, i64 %finalsize
; CHECK-NEXT:   br label %cleanup

; CHECK: cleanup:                                          ; preds = %if.else, %entry
; CHECK-NEXT:   %finalret = phi i64 [ %cond, %[[ifelse]] ], [ %finalsize, %entry ]
; CHECK-NEXT:   ret i64 %finalret
; CHECK-NEXT: }

; CHECK: define internal i64 @augmented_metasub(double* %array, double* %"array'", i64 %metasize)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %finalcall = call i64 @augmented_final(double* %array, double* %"array'", i64 %metasize)
; CHECK-NEXT:   ret i64 %finalcall
; CHECK-NEXT: }

; CHECK: define internal i64 @augmented_sub(double* %in, double* %"in'", i64 %subsize)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = ptrtoint double* %in to i64
; CHECK-NEXT:   %tobool = icmp eq i64 %0, 0
; CHECK-NEXT:   br i1 %tobool, label %if.end, label %return

; CHECK: if.end:                                           ; preds = %entry
; CHECK-NEXT:   %metacall = call i64 @augmented_metasub(double* %in, double* %"in'", i64 %subsize)
; CHECK-NEXT:   br label %return

; CHECK: return:                                           ; preds = %if.end, %entry
; CHECK-NEXT:   %subret = phi i64 [ %metacall, %if.end ], [ -1, %entry ]
; CHECK-NEXT:   ret i64 %subret
; CHECK-NEXT: }

; CHECK: define internal void @diffesub(double* %in, double* %"in'", i64 %subsize)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = ptrtoint double* %in to i64
; CHECK-NEXT:   %[[tobool:.+]] = icmp eq i64 %0, 0
; CHECK-NEXT:   br i1 %[[tobool]], label %[[ifend:.+]], label %invertentry

; CHECK: invertentry:
; CHECK-NEXT:   ret void

; CHECK: [[ifend]]:                                           ; preds = %entry
; CHECK-NEXT:   call void @diffemetasub(double* %in, double* %"in'", i64 %subsize)
; CHECK-NEXT:   br label %invertentry
; CHECK-NEXT: }

; CHECK: define internal void @diffemetasub(double* %array, double* %"array'", i64 %metasize)
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @diffefinal(double* %array, double* %"array'", i64 %metasize)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @diffefinal(double* %array, double* %"array'", i64 %finalsize)
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
