; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -simplifycfg -instcombine -adce -S | FileCheck %s

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
%"class.Eigen::internal::const_blas_data_mapper" = type { %"class.Eigen::internal::blas_data_mapper" }
%"class.Eigen::internal::blas_data_mapper" = type { double*, i64 }
%"class.Eigen::internal::const_blas_data_mapper.32" = type { %"class.Eigen::internal::blas_data_mapper.33" }
%"class.Eigen::internal::blas_data_mapper.33" = type { double*, i64 }

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
@.str.24 = private unnamed_addr constant [47 x i8] c"dst.rows() == dstRows && dst.cols() == dstCols\00", align 1
@.str.25 = private unnamed_addr constant [59 x i8] c"/usr/local/include/eigen3/Eigen/src/Core/AssignEvaluator.h\00", align 1
@__PRETTY_FUNCTION__._ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEE6resizeEll = private unnamed_addr constant [160 x i8] c"void Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >::resize(Eigen::Index, Eigen::Index) [Derived = Eigen::Matrix<double, -1, -1, 0, -1, -1>]\00", align 1
@__PRETTY_FUNCTION__._ZN5Eigen8internal17resize_if_allowedINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EEddEEvRT_RKT0_RKNS0_9assign_opIT1_T2_EE = private unnamed_addr constant [317 x i8] c"void Eigen::internal::resize_if_allowed(DstXprType &, const SrcXprType &, const internal::assign_op<T1, T2> &) [DstXprType = Eigen::Matrix<double, -1, -1, 0, -1, -1>, SrcXprType = Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<double>, Eigen::Matrix<double, -1, -1, 0, -1, -1> >, T1 = double, T2 = double]\00", align 1
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
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %0) #8
  call void @llvm.memset.p0i8.i64(i8* align 8 %0, i8 0, i64 24, i1 false) #8
  %m_rows.i.i.i = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %W, i64 0, i32 0, i32 0, i32 1
  %1 = load i64, i64* %m_rows.i.i.i, align 8, !tbaa !2
  %m_cols.i.i.i = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %W, i64 0, i32 0, i32 0, i32 2
  %2 = load i64, i64* %m_cols.i.i.i, align 8, !tbaa !8
  %mul.i.i.i1 = mul nsw i64 %2, %1
  %cmp.i9.i.i = icmp eq i64 %mul.i.i.i1, 21
  br i1 %cmp.i9.i.i, label %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEE6resizeEll.exit.i, label %if.then.i10.i.i

if.then.i10.i.i:                                  ; preds = %entry
  %3 = bitcast %"class.Eigen::Matrix"* %W to i8**
  %4 = load i8*, i8** %3, align 8, !tbaa !9
  call void @free(i8* %4) #8
  %call.i.i4.i.i.i = call noalias i8* @malloc(i64 168) #8
  %5 = ptrtoint i8* %call.i.i4.i.i.i to i64
  %rem.i.i.i.i.i = and i64 %5, 15
  %cmp1.i.i.i.i.i = icmp eq i64 %rem.i.i.i.i.i, 0
  br i1 %cmp1.i.i.i.i.i, label %cond.end.i.i.i.i.i, label %cond.false.i.i.i.i.i

cond.false.i.i.i.i.i:                             ; preds = %if.then.i10.i.i
  call void @__assert_fail(i8* getelementptr inbounds ([186 x i8], [186 x i8]* @.str.18, i64 0, i64 0), i8* getelementptr inbounds ([55 x i8], [55 x i8]* @.str.19, i64 0, i64 0), i32 161, i8* getelementptr inbounds ([51 x i8], [51 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen8internal14aligned_mallocEm, i64 0, i64 0)) #9
  unreachable

cond.end.i.i.i.i.i:                               ; preds = %if.then.i10.i.i
  %tobool.i.i.i.i.i = icmp eq i8* %call.i.i4.i.i.i, null
  br i1 %tobool.i.i.i.i.i, label %if.then.i.i.i.i.i4, label %_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit.i.i

if.then.i.i.i.i.i4:                               ; preds = %cond.end.i.i.i.i.i
  %call.i.i.i.i.i.i3 = call i8* @_Znwm(i64 -1) #8
  br label %_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit.i.i

_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit.i.i: ; preds = %if.then.i.i.i.i.i4, %cond.end.i.i.i.i.i
  store i8* %call.i.i4.i.i.i, i8** %3, align 8, !tbaa !9
  br label %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEE6resizeEll.exit.i

_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEE6resizeEll.exit.i: ; preds = %_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit.i.i, %entry
  store i64 3, i64* %m_rows.i.i.i, align 8, !tbaa !2
  store i64 7, i64* %m_cols.i.i.i, align 8, !tbaa !8
  %6 = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %W, i64 0, i32 0, i32 0, i32 0
  %7 = load double*, double** %6, align 8, !tbaa !9
  br label %for.body.i.i.i.i.i.i

for.body.i.i.i.i.i.i.i:                           ; preds = %for.body.i.i.i.i.i.i, %for.body.i.i.i.i.i.i.i
  %index.05.i.i.i.i.i.i.i = phi i64 [ %inc.i.i.i.i.i.i.i, %for.body.i.i.i.i.i.i.i ], [ 20, %for.body.i.i.i.i.i.i ]
  %arrayidx.i.i.i.i.i.i.i.i.i = getelementptr inbounds double, double* %7, i64 %index.05.i.i.i.i.i.i.i
  %8 = bitcast double* %arrayidx.i.i.i.i.i.i.i.i.i to i64*
  store i64 4613937818241073152, i64* %8, align 8, !tbaa !10
  %inc.i.i.i.i.i.i.i = add nuw nsw i64 %index.05.i.i.i.i.i.i.i, 1
  %exitcond.i.i.i.i.i.i.i = icmp eq i64 %inc.i.i.i.i.i.i.i, 21
  br i1 %exitcond.i.i.i.i.i.i.i, label %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEEC2INS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEERKNS_9DenseBaseIT_EE.exit, label %for.body.i.i.i.i.i.i.i

for.body.i.i.i.i.i.i:                             ; preds = %for.body.i.i.i.i.i.i, %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEE6resizeEll.exit.i
  %index.014.i.i.i.i.i.i = phi i64 [ %add1.i.i.i.i.i.i, %for.body.i.i.i.i.i.i ], [ 0, %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEE6resizeEll.exit.i ]
  %arrayidx.i.i.i.i.i.i.i.i = getelementptr inbounds double, double* %7, i64 %index.014.i.i.i.i.i.i
  %9 = bitcast double* %arrayidx.i.i.i.i.i.i.i.i to <2 x double>*
  store <2 x double> <double 3.000000e+00, double 3.000000e+00>, <2 x double>* %9, align 16, !tbaa !12
  %add1.i.i.i.i.i.i = add nuw nsw i64 %index.014.i.i.i.i.i.i, 2
  %cmp.i.i.i.i.i.i6 = icmp ult i64 %add1.i.i.i.i.i.i, 20
  br i1 %cmp.i.i.i.i.i.i6, label %for.body.i.i.i.i.i.i, label %for.body.i.i.i.i.i.i.i

_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEEC2INS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEERKNS_9DenseBaseIT_EE.exit: ; preds = %for.body.i.i.i.i.i.i.i
  %10 = bitcast %"class.Eigen::Matrix.6"* %M to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %10) #8
  call void @llvm.memset.p0i8.i64(i8* nonnull align 8 %10, i8 0, i64 16, i1 false) #8
  %m_rows.i.i.i.i.i = getelementptr inbounds %"class.Eigen::Matrix.6", %"class.Eigen::Matrix.6"* %M, i64 0, i32 0, i32 0, i32 1
  %11 = bitcast %"class.Eigen::Matrix.6"* %M to i8**
  %call.i.i4.i.i.i.i.i.i = call noalias i8* @malloc(i64 56) #8
  %12 = ptrtoint i8* %call.i.i4.i.i.i.i.i.i to i64
  %rem.i.i.i.i.i.i.i.i = and i64 %12, 15
  %cmp1.i.i.i.i.i.i.i.i = icmp eq i64 %rem.i.i.i.i.i.i.i.i, 0
  br i1 %cmp1.i.i.i.i.i.i.i.i, label %cond.end.i.i.i.i.i.i.i.i, label %cond.false.i.i.i.i.i.i.i.i

cond.false.i.i.i.i.i.i.i.i:                       ; preds = %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEEC2INS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEERKNS_9DenseBaseIT_EE.exit
  call void @__assert_fail(i8* getelementptr inbounds ([186 x i8], [186 x i8]* @.str.18, i64 0, i64 0), i8* getelementptr inbounds ([55 x i8], [55 x i8]* @.str.19, i64 0, i64 0), i32 161, i8* getelementptr inbounds ([51 x i8], [51 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen8internal14aligned_mallocEm, i64 0, i64 0)) #9
  unreachable

cond.end.i.i.i.i.i.i.i.i:                         ; preds = %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEEC2INS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEERKNS_9DenseBaseIT_EE.exit
  %tobool.i.i.i.i.i.i.i.i = icmp eq i8* %call.i.i4.i.i.i.i.i.i, null
  br i1 %tobool.i.i.i.i.i.i.i.i, label %if.then.i.i.i.i.i.i.i.i, label %_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit.i.i.i.i.i

if.then.i.i.i.i.i.i.i.i:                          ; preds = %cond.end.i.i.i.i.i.i.i.i
  %call.i.i.i.i.i.i.i.i.i = call i8* @_Znwm(i64 -1) #8
  br label %_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit.i.i.i.i.i

_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit.i.i.i.i.i: ; preds = %if.then.i.i.i.i.i.i.i.i, %cond.end.i.i.i.i.i.i.i.i
  store i8* %call.i.i4.i.i.i.i.i.i, i8** %11, align 8, !tbaa !13
  store i64 7, i64* %m_rows.i.i.i.i.i, align 8, !tbaa !15
  %.cast = bitcast i8* %call.i.i4.i.i.i.i.i.i to double*
  br label %for.body.i.i.i.i

for.body.i.i.i.i.i:                               ; preds = %for.body.i.i.i.i, %for.body.i.i.i.i.i
  %index.05.i.i.i.i.i = phi i64 [ %inc.i.i.i.i.i, %for.body.i.i.i.i.i ], [ 6, %for.body.i.i.i.i ]
  %arrayidx.i.i.i.i.i.i.i = getelementptr inbounds double, double* %.cast, i64 %index.05.i.i.i.i.i
  %13 = bitcast double* %arrayidx.i.i.i.i.i.i.i to i64*
  store i64 4611686018427387904, i64* %13, align 8, !tbaa !10
  %inc.i.i.i.i.i = add nuw nsw i64 %index.05.i.i.i.i.i, 1
  %exitcond.i.i.i.i.i = icmp eq i64 %inc.i.i.i.i.i, 7
  br i1 %exitcond.i.i.i.i.i, label %_ZN5Eigen8internal24call_assignment_no_aliasINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EENS0_9assign_opIddEEEEvRT_RKT0_RKT1_.exit, label %for.body.i.i.i.i.i

for.body.i.i.i.i:                                 ; preds = %for.body.i.i.i.i, %_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit.i.i.i.i.i
  %index.014.i.i.i.i = phi i64 [ %add1.i.i.i.i, %for.body.i.i.i.i ], [ 0, %_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit.i.i.i.i.i ]
  %arrayidx.i.i.i.i.i.i = getelementptr inbounds double, double* %.cast, i64 %index.014.i.i.i.i
  %14 = bitcast double* %arrayidx.i.i.i.i.i.i to <2 x double>*
  store <2 x double> <double 2.000000e+00, double 2.000000e+00>, <2 x double>* %14, align 16, !tbaa !12
  %add1.i.i.i.i = add nuw nsw i64 %index.014.i.i.i.i, 2
  %cmp.i.i.i.i30 = icmp ult i64 %add1.i.i.i.i, 6
  br i1 %cmp.i.i.i.i30, label %for.body.i.i.i.i, label %for.body.i.i.i.i.i

_ZN5Eigen8internal24call_assignment_no_aliasINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EENS0_9assign_opIddEEEEvRT_RKT0_RKT1_.exit: ; preds = %for.body.i.i.i.i.i
  %15 = bitcast %"class.Eigen::Matrix.6"* %O to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %15) #8
  call void @llvm.memset.p0i8.i64(i8* nonnull align 8 %15, i8 0, i64 16, i1 false) #8
  %m_rows.i.i.i.i.i210 = getelementptr inbounds %"class.Eigen::Matrix.6", %"class.Eigen::Matrix.6"* %O, i64 0, i32 0, i32 0, i32 1
  %16 = bitcast %"class.Eigen::Matrix.6"* %O to i8**
  %call.i.i4.i.i.i.i.i.i217 = call noalias i8* @malloc(i64 24) #8
  %17 = ptrtoint i8* %call.i.i4.i.i.i.i.i.i217 to i64
  %rem.i.i.i.i.i.i.i.i220 = and i64 %17, 15
  %cmp1.i.i.i.i.i.i.i.i221 = icmp eq i64 %rem.i.i.i.i.i.i.i.i220, 0
  br i1 %cmp1.i.i.i.i.i.i.i.i221, label %cond.end.i.i.i.i.i.i.i.i227, label %cond.false.i.i.i.i.i.i.i.i223

cond.false.i.i.i.i.i.i.i.i223:                    ; preds = %_ZN5Eigen8internal24call_assignment_no_aliasINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EENS0_9assign_opIddEEEEvRT_RKT0_RKT1_.exit
  call void @__assert_fail(i8* getelementptr inbounds ([186 x i8], [186 x i8]* @.str.18, i64 0, i64 0), i8* getelementptr inbounds ([55 x i8], [55 x i8]* @.str.19, i64 0, i64 0), i32 161, i8* getelementptr inbounds ([51 x i8], [51 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen8internal14aligned_mallocEm, i64 0, i64 0)) #9
  unreachable

cond.end.i.i.i.i.i.i.i.i227:                      ; preds = %_ZN5Eigen8internal24call_assignment_no_aliasINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EENS0_9assign_opIddEEEEvRT_RKT0_RKT1_.exit
  %tobool.i.i.i.i.i.i.i.i224 = icmp eq i8* %call.i.i4.i.i.i.i.i.i217, null
  br i1 %tobool.i.i.i.i.i.i.i.i224, label %if.then.i.i.i.i.i.i.i.i229, label %_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit.i.i.i.i.i230

if.then.i.i.i.i.i.i.i.i229:                       ; preds = %cond.end.i.i.i.i.i.i.i.i227
  %call.i.i.i.i.i.i.i.i.i228 = call i8* @_Znwm(i64 -1) #8
  br label %_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit.i.i.i.i.i230

_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit.i.i.i.i.i230: ; preds = %if.then.i.i.i.i.i.i.i.i229, %cond.end.i.i.i.i.i.i.i.i227
  store i8* %call.i.i4.i.i.i.i.i.i217, i8** %16, align 8, !tbaa !13
  store i64 3, i64* %m_rows.i.i.i.i.i210, align 8, !tbaa !15
  %.cast1 = bitcast i8* %call.i.i4.i.i.i.i.i.i217 to double*
  %18 = bitcast i8* %call.i.i4.i.i.i.i.i.i217 to <2 x double>*
  store <2 x double> zeroinitializer, <2 x double>* %18, align 16, !tbaa !12
  br label %for.body.i.i.i.i.i76

for.body.i.i.i.i.i76:                             ; preds = %for.body.i.i.i.i.i76, %_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit.i.i.i.i.i230
  %index.05.i.i.i.i.i72 = phi i64 [ 2, %_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit.i.i.i.i.i230 ], [ %inc.i.i.i.i.i74, %for.body.i.i.i.i.i76 ]
  %arrayidx.i.i.i.i.i.i.i73 = getelementptr inbounds double, double* %.cast1, i64 %index.05.i.i.i.i.i72
  %19 = bitcast double* %arrayidx.i.i.i.i.i.i.i73 to i64*
  store i64 0, i64* %19, align 8, !tbaa !10
  %inc.i.i.i.i.i74 = add nuw nsw i64 %index.05.i.i.i.i.i72, 1
  %exitcond.i.i.i.i.i75 = icmp eq i64 %inc.i.i.i.i.i74, 3
  br i1 %exitcond.i.i.i.i.i75, label %_ZN5Eigen8internal24call_assignment_no_aliasINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EENS0_9assign_opIddEEEEvRT_RKT0_RKT1_.exit82, label %for.body.i.i.i.i.i76

_ZN5Eigen8internal24call_assignment_no_aliasINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EENS0_9assign_opIddEEEEvRT_RKT0_RKT1_.exit82: ; preds = %for.body.i.i.i.i.i76
  %20 = bitcast %"class.Eigen::Matrix"* %Wp to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %20) #8
  call void @llvm.memset.p0i8.i64(i8* align 8 %20, i8 0, i64 24, i1 false) #8
  %m_rows.i.i.i107 = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %Wp, i64 0, i32 0, i32 0, i32 1
  %21 = load i64, i64* %m_rows.i.i.i107, align 8, !tbaa !2
  %m_cols.i.i.i108 = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %Wp, i64 0, i32 0, i32 0, i32 2
  %22 = load i64, i64* %m_cols.i.i.i108, align 8, !tbaa !8
  %mul.i.i.i109 = mul nsw i64 %22, %21
  %cmp.i9.i.i110 = icmp eq i64 %mul.i.i.i109, 21
  br i1 %cmp.i9.i.i110, label %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEE6resizeEll.exit.i137, label %if.then.i10.i.i113

if.then.i10.i.i113:                               ; preds = %_ZN5Eigen8internal24call_assignment_no_aliasINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EENS0_9assign_opIddEEEEvRT_RKT0_RKT1_.exit82
  %23 = bitcast %"class.Eigen::Matrix"* %Wp to i8**
  %24 = load i8*, i8** %23, align 8, !tbaa !9
  call void @free(i8* %24) #8
  %call.i.i4.i.i.i119 = call noalias i8* @malloc(i64 168) #8
  %25 = ptrtoint i8* %call.i.i4.i.i.i119 to i64
  %rem.i.i.i.i.i122 = and i64 %25, 15
  %cmp1.i.i.i.i.i123 = icmp eq i64 %rem.i.i.i.i.i122, 0
  br i1 %cmp1.i.i.i.i.i123, label %cond.end.i.i.i.i.i129, label %cond.false.i.i.i.i.i125

cond.false.i.i.i.i.i125:                          ; preds = %if.then.i10.i.i113
  call void @__assert_fail(i8* getelementptr inbounds ([186 x i8], [186 x i8]* @.str.18, i64 0, i64 0), i8* getelementptr inbounds ([55 x i8], [55 x i8]* @.str.19, i64 0, i64 0), i32 161, i8* getelementptr inbounds ([51 x i8], [51 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen8internal14aligned_mallocEm, i64 0, i64 0)) #9
  unreachable

cond.end.i.i.i.i.i129:                            ; preds = %if.then.i10.i.i113
  %tobool.i.i.i.i.i126 = icmp eq i8* %call.i.i4.i.i.i119, null
  br i1 %tobool.i.i.i.i.i126, label %if.then.i.i.i.i.i131, label %_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit.i.i132

if.then.i.i.i.i.i131:                             ; preds = %cond.end.i.i.i.i.i129
  %call.i.i.i.i.i.i130 = call i8* @_Znwm(i64 -1) #8
  br label %_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit.i.i132

_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit.i.i132: ; preds = %if.then.i.i.i.i.i131, %cond.end.i.i.i.i.i129
  store i8* %call.i.i4.i.i.i119, i8** %23, align 8, !tbaa !9
  br label %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEE6resizeEll.exit.i137

_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEE6resizeEll.exit.i137: ; preds = %_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit.i.i132, %_ZN5Eigen8internal24call_assignment_no_aliasINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EENS0_9assign_opIddEEEEvRT_RKT0_RKT1_.exit82
  store i64 3, i64* %m_rows.i.i.i107, align 8, !tbaa !2
  store i64 7, i64* %m_cols.i.i.i108, align 8, !tbaa !8
  %26 = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %Wp, i64 0, i32 0, i32 0, i32 0
  %27 = load double*, double** %26, align 8, !tbaa !9
  br label %for.body.i.i.i.i.i.i199

for.body.i.i.i.i.i.i.i194:                        ; preds = %for.body.i.i.i.i.i.i199, %for.body.i.i.i.i.i.i.i194
  %index.05.i.i.i.i.i.i.i190 = phi i64 [ %inc.i.i.i.i.i.i.i192, %for.body.i.i.i.i.i.i.i194 ], [ 20, %for.body.i.i.i.i.i.i199 ]
  %arrayidx.i.i.i.i.i.i.i.i.i191 = getelementptr inbounds double, double* %27, i64 %index.05.i.i.i.i.i.i.i190
  %28 = bitcast double* %arrayidx.i.i.i.i.i.i.i.i.i191 to i64*
  store i64 0, i64* %28, align 8, !tbaa !10
  %inc.i.i.i.i.i.i.i192 = add nuw nsw i64 %index.05.i.i.i.i.i.i.i190, 1
  %exitcond.i.i.i.i.i.i.i193 = icmp eq i64 %inc.i.i.i.i.i.i.i192, 21
  br i1 %exitcond.i.i.i.i.i.i.i193, label %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEEC2INS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEERKNS_9DenseBaseIT_EE.exit200, label %for.body.i.i.i.i.i.i.i194

for.body.i.i.i.i.i.i199:                          ; preds = %for.body.i.i.i.i.i.i199, %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEE6resizeEll.exit.i137
  %index.014.i.i.i.i.i.i195 = phi i64 [ %add1.i.i.i.i.i.i197, %for.body.i.i.i.i.i.i199 ], [ 0, %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEE6resizeEll.exit.i137 ]
  %arrayidx.i.i.i.i.i.i.i.i196 = getelementptr inbounds double, double* %27, i64 %index.014.i.i.i.i.i.i195
  %29 = bitcast double* %arrayidx.i.i.i.i.i.i.i.i196 to <2 x double>*
  store <2 x double> zeroinitializer, <2 x double>* %29, align 16, !tbaa !12
  %add1.i.i.i.i.i.i197 = add nuw nsw i64 %index.014.i.i.i.i.i.i195, 2
  %cmp.i.i.i.i.i.i198 = icmp ult i64 %add1.i.i.i.i.i.i197, 20
  br i1 %cmp.i.i.i.i.i.i198, label %for.body.i.i.i.i.i.i199, label %for.body.i.i.i.i.i.i.i194

_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEEC2INS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEERKNS_9DenseBaseIT_EE.exit200: ; preds = %for.body.i.i.i.i.i.i.i194
  %30 = bitcast %"class.Eigen::Matrix.6"* %Mp to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %30) #8
  call void @llvm.memset.p0i8.i64(i8* nonnull align 8 %30, i8 0, i64 16, i1 false) #8
  %m_rows.i.i.i.i.i242 = getelementptr inbounds %"class.Eigen::Matrix.6", %"class.Eigen::Matrix.6"* %Mp, i64 0, i32 0, i32 0, i32 1
  %31 = bitcast %"class.Eigen::Matrix.6"* %Mp to i8**
  %call.i.i4.i.i.i.i.i.i249 = call noalias i8* @malloc(i64 56) #8
  %32 = ptrtoint i8* %call.i.i4.i.i.i.i.i.i249 to i64
  %rem.i.i.i.i.i.i.i.i252 = and i64 %32, 15
  %cmp1.i.i.i.i.i.i.i.i253 = icmp eq i64 %rem.i.i.i.i.i.i.i.i252, 0
  br i1 %cmp1.i.i.i.i.i.i.i.i253, label %cond.end.i.i.i.i.i.i.i.i259, label %cond.false.i.i.i.i.i.i.i.i255

cond.false.i.i.i.i.i.i.i.i255:                    ; preds = %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEEC2INS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEERKNS_9DenseBaseIT_EE.exit200
  call void @__assert_fail(i8* getelementptr inbounds ([186 x i8], [186 x i8]* @.str.18, i64 0, i64 0), i8* getelementptr inbounds ([55 x i8], [55 x i8]* @.str.19, i64 0, i64 0), i32 161, i8* getelementptr inbounds ([51 x i8], [51 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen8internal14aligned_mallocEm, i64 0, i64 0)) #9
  unreachable

cond.end.i.i.i.i.i.i.i.i259:                      ; preds = %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEEC2INS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEERKNS_9DenseBaseIT_EE.exit200
  %tobool.i.i.i.i.i.i.i.i256 = icmp eq i8* %call.i.i4.i.i.i.i.i.i249, null
  br i1 %tobool.i.i.i.i.i.i.i.i256, label %if.then.i.i.i.i.i.i.i.i261, label %_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit.i.i.i.i.i262

if.then.i.i.i.i.i.i.i.i261:                       ; preds = %cond.end.i.i.i.i.i.i.i.i259
  %call.i.i.i.i.i.i.i.i.i260 = call i8* @_Znwm(i64 -1) #8
  br label %_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit.i.i.i.i.i262

_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit.i.i.i.i.i262: ; preds = %if.then.i.i.i.i.i.i.i.i261, %cond.end.i.i.i.i.i.i.i.i259
  store i8* %call.i.i4.i.i.i.i.i.i249, i8** %31, align 8, !tbaa !13
  store i64 7, i64* %m_rows.i.i.i.i.i242, align 8, !tbaa !15
  %.cast2 = bitcast i8* %call.i.i4.i.i.i.i.i.i249 to double*
  br label %for.body.i.i.i.i251

for.body.i.i.i.i.i246:                            ; preds = %for.body.i.i.i.i251, %for.body.i.i.i.i.i246
  %index.05.i.i.i.i.i242 = phi i64 [ %inc.i.i.i.i.i244, %for.body.i.i.i.i.i246 ], [ 6, %for.body.i.i.i.i251 ]
  %arrayidx.i.i.i.i.i.i.i243 = getelementptr inbounds double, double* %.cast2, i64 %index.05.i.i.i.i.i242
  %33 = bitcast double* %arrayidx.i.i.i.i.i.i.i243 to i64*
  store i64 0, i64* %33, align 8, !tbaa !10
  %inc.i.i.i.i.i244 = add nuw nsw i64 %index.05.i.i.i.i.i242, 1
  %exitcond.i.i.i.i.i245 = icmp eq i64 %inc.i.i.i.i.i244, 7
  br i1 %exitcond.i.i.i.i.i245, label %_ZN5Eigen8internal24call_assignment_no_aliasINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EENS0_9assign_opIddEEEEvRT_RKT0_RKT1_.exit252, label %for.body.i.i.i.i.i246

for.body.i.i.i.i251:                              ; preds = %for.body.i.i.i.i251, %_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit.i.i.i.i.i262
  %index.014.i.i.i.i247 = phi i64 [ %add1.i.i.i.i249, %for.body.i.i.i.i251 ], [ 0, %_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit.i.i.i.i.i262 ]
  %arrayidx.i.i.i.i.i.i248 = getelementptr inbounds double, double* %.cast2, i64 %index.014.i.i.i.i247
  %34 = bitcast double* %arrayidx.i.i.i.i.i.i248 to <2 x double>*
  store <2 x double> zeroinitializer, <2 x double>* %34, align 16, !tbaa !12
  %add1.i.i.i.i249 = add nuw nsw i64 %index.014.i.i.i.i247, 2
  %cmp.i.i.i.i250 = icmp ult i64 %add1.i.i.i.i249, 6
  br i1 %cmp.i.i.i.i250, label %for.body.i.i.i.i251, label %for.body.i.i.i.i.i246

_ZN5Eigen8internal24call_assignment_no_aliasINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EENS0_9assign_opIddEEEEvRT_RKT0_RKT1_.exit252: ; preds = %for.body.i.i.i.i.i246
  %35 = bitcast %"class.Eigen::Matrix.6"* %Op to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %35) #8
  call void @llvm.memset.p0i8.i64(i8* nonnull align 8 %35, i8 0, i64 16, i1 false) #8
  %m_rows.i.i.i.i.i271 = getelementptr inbounds %"class.Eigen::Matrix.6", %"class.Eigen::Matrix.6"* %Op, i64 0, i32 0, i32 0, i32 1
  %36 = bitcast %"class.Eigen::Matrix.6"* %Op to i8**
  %call.i.i4.i.i.i.i.i.i278 = call noalias i8* @malloc(i64 24) #8
  %37 = ptrtoint i8* %call.i.i4.i.i.i.i.i.i278 to i64
  %rem.i.i.i.i.i.i.i.i281 = and i64 %37, 15
  %cmp1.i.i.i.i.i.i.i.i282 = icmp eq i64 %rem.i.i.i.i.i.i.i.i281, 0
  br i1 %cmp1.i.i.i.i.i.i.i.i282, label %cond.end.i.i.i.i.i.i.i.i288, label %cond.false.i.i.i.i.i.i.i.i284

cond.false.i.i.i.i.i.i.i.i284:                    ; preds = %_ZN5Eigen8internal24call_assignment_no_aliasINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EENS0_9assign_opIddEEEEvRT_RKT0_RKT1_.exit252
  call void @__assert_fail(i8* getelementptr inbounds ([186 x i8], [186 x i8]* @.str.18, i64 0, i64 0), i8* getelementptr inbounds ([55 x i8], [55 x i8]* @.str.19, i64 0, i64 0), i32 161, i8* getelementptr inbounds ([51 x i8], [51 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen8internal14aligned_mallocEm, i64 0, i64 0)) #9
  unreachable

cond.end.i.i.i.i.i.i.i.i288:                      ; preds = %_ZN5Eigen8internal24call_assignment_no_aliasINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EENS0_9assign_opIddEEEEvRT_RKT0_RKT1_.exit252
  %tobool.i.i.i.i.i.i.i.i285 = icmp eq i8* %call.i.i4.i.i.i.i.i.i278, null
  br i1 %tobool.i.i.i.i.i.i.i.i285, label %if.then.i.i.i.i.i.i.i.i290, label %_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit.i.i.i.i.i291

if.then.i.i.i.i.i.i.i.i290:                       ; preds = %cond.end.i.i.i.i.i.i.i.i288
  %call.i.i.i.i.i.i.i.i.i289 = call i8* @_Znwm(i64 -1) #8
  br label %_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit.i.i.i.i.i291

_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit.i.i.i.i.i291: ; preds = %if.then.i.i.i.i.i.i.i.i290, %cond.end.i.i.i.i.i.i.i.i288
  store i8* %call.i.i4.i.i.i.i.i.i278, i8** %36, align 8, !tbaa !13
  store i64 3, i64* %m_rows.i.i.i.i.i271, align 8, !tbaa !15
  %.cast3 = bitcast i8* %call.i.i4.i.i.i.i.i.i278 to double*
  %38 = bitcast i8* %call.i.i4.i.i.i.i.i.i278 to <2 x double>*
  store <2 x double> <double 1.000000e+00, double 1.000000e+00>, <2 x double>* %38, align 16, !tbaa !12
  br label %for.body.i.i.i.i.i298

for.body.i.i.i.i.i298:                            ; preds = %for.body.i.i.i.i.i298, %_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit.i.i.i.i.i291
  %index.05.i.i.i.i.i294 = phi i64 [ 2, %_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit.i.i.i.i.i291 ], [ %inc.i.i.i.i.i296, %for.body.i.i.i.i.i298 ]
  %arrayidx.i.i.i.i.i.i.i295 = getelementptr inbounds double, double* %.cast3, i64 %index.05.i.i.i.i.i294
  %39 = bitcast double* %arrayidx.i.i.i.i.i.i.i295 to i64*
  store i64 4607182418800017408, i64* %39, align 8, !tbaa !10
  %inc.i.i.i.i.i296 = add nuw nsw i64 %index.05.i.i.i.i.i294, 1
  %exitcond.i.i.i.i.i297 = icmp eq i64 %inc.i.i.i.i.i296, 3
  br i1 %exitcond.i.i.i.i.i297, label %_ZN5Eigen8internal23check_size_for_overflowIdEEvm.exit.i.i.i.i, label %for.body.i.i.i.i.i298

_ZN5Eigen8internal23check_size_for_overflowIdEEvm.exit.i.i.i.i: ; preds = %for.body.i.i.i.i.i298
  %call.i.i4.i.i.i.i = call noalias i8* @malloc(i64 24) #8
  %40 = ptrtoint i8* %call.i.i4.i.i.i.i to i64
  %rem.i.i.i.i.i.i = and i64 %40, 15
  %cmp1.i.i.i.i.i.i = icmp eq i64 %rem.i.i.i.i.i.i, 0
  br i1 %cmp1.i.i.i.i.i.i, label %cond.end.i.i.i.i.i.i, label %cond.false.i.i.i.i.i.i

cond.false.i.i.i.i.i.i:                           ; preds = %_ZN5Eigen8internal23check_size_for_overflowIdEEvm.exit.i.i.i.i
  call void @__assert_fail(i8* getelementptr inbounds ([186 x i8], [186 x i8]* @.str.18, i64 0, i64 0), i8* getelementptr inbounds ([55 x i8], [55 x i8]* @.str.19, i64 0, i64 0), i32 161, i8* getelementptr inbounds ([51 x i8], [51 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen8internal14aligned_mallocEm, i64 0, i64 0)) #9
  unreachable

cond.end.i.i.i.i.i.i:                             ; preds = %_ZN5Eigen8internal23check_size_for_overflowIdEEvm.exit.i.i.i.i
  %tobool.i.i.i.i.i.i = icmp eq i8* %call.i.i4.i.i.i.i, null
  br i1 %tobool.i.i.i.i.i.i, label %if.then.i.i.i.i.i.i, label %_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit.i.i.i

if.then.i.i.i.i.i.i:                              ; preds = %cond.end.i.i.i.i.i.i
  %call.i.i.i.i.i.i.i = call i8* @_Znwm(i64 -1) #8
  br label %_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit.i.i.i

_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit.i.i.i: ; preds = %if.then.i.i.i.i.i.i, %cond.end.i.i.i.i.i.i
  %.pr = load i64, i64* %m_rows.i.i.i.i.i271, align 8, !tbaa !15
  %41 = bitcast i8* %call.i.i4.i.i.i.i to double*
  %cmp.i.i13.i.i.i = icmp eq i64 %.pr, 0
  br i1 %cmp.i.i13.i.i.i, label %_ZN5Eigen6MatrixIdLin1ELi1ELi0ELin1ELi1EEC2ERKS1_.exit, label %if.end.i.i.i.i.i

if.end.i.i.i.i.i:                                 ; preds = %_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit.i.i.i
  %add.ptr.idx.i.i.i = shl nuw i64 %.pr, 3
  %42 = load i8*, i8** %36, align 8, !tbaa !13
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %call.i.i4.i.i.i.i, i8* align 8 %42, i64 %add.ptr.idx.i.i.i, i1 false) #8
  br label %_ZN5Eigen6MatrixIdLin1ELi1ELi0ELin1ELi1EEC2ERKS1_.exit

_ZN5Eigen6MatrixIdLin1ELi1ELi0ELin1ELi1EEC2ERKS1_.exit: ; preds = %if.end.i.i.i.i.i, %_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit.i.i.i
  %call = call fast double @__enzyme_autodiff(i8* bitcast (void (%"class.Eigen::Matrix"*, %"class.Eigen::Matrix.6"*, %"class.Eigen::Matrix.6"*)* @_ZL6matvecPKN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEEPKNS0_IdLin1ELi1ELi0ELin1ELi1EEEPS4_ to i8*), i8* nonnull %0, i8* nonnull %20, i8* nonnull %10, i8* nonnull %30, i8* nonnull %15, i8* nonnull %35) #8
  br label %for.cond12.preheader

for.cond12.preheader:                             ; preds = %for.cond.cleanup15, %_ZN5Eigen6MatrixIdLin1ELi1ELi0ELin1ELi1EEC2ERKS1_.exit
  %indvars.iv445 = phi i64 [ 0, %_ZN5Eigen6MatrixIdLin1ELi1ELi0ELin1ELi1EEC2ERKS1_.exit ], [ %indvars.iv.next446, %for.cond.cleanup15 ]
  %43 = trunc i64 %indvars.iv445 to i32
  br label %for.body16

for.cond25.preheader:                             ; preds = %for.cond.cleanup15
  %44 = getelementptr inbounds %"class.Eigen::Matrix.6", %"class.Eigen::Matrix.6"* %M, i64 0, i32 0, i32 0, i32 0
  br label %for.body29

for.cond.cleanup15:                               ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit
  %indvars.iv.next446 = add nuw nsw i64 %indvars.iv445, 1
  %cmp = icmp ult i64 %indvars.iv.next446, 7
  br i1 %cmp, label %for.cond12.preheader, label %for.cond25.preheader

for.body16:                                       ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit, %for.cond12.preheader
  %indvars.iv443 = phi i64 [ 0, %for.cond12.preheader ], [ %indvars.iv.next444, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit ]
  %45 = load i64, i64* %m_rows.i.i.i, align 8, !tbaa !2
  %cmp2.i294 = icmp sgt i64 %45, %indvars.iv443
  %46 = load i64, i64* %m_cols.i.i.i, align 8
  %cmp7.i = icmp sgt i64 %46, %indvars.iv445
  %or.cond = and i1 %cmp2.i294, %cmp7.i
  br i1 %or.cond, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit, label %cond.false.i295

cond.false.i295:                                  ; preds = %for.body16
  call void @__assert_fail(i8* getelementptr inbounds ([53 x i8], [53 x i8]* @.str.27, i64 0, i64 0), i8* getelementptr inbounds ([59 x i8], [59 x i8]* @.str.28, i64 0, i64 0), i32 365, i8* getelementptr inbounds ([227 x i8], [227 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll, i64 0, i64 0)) #9
  unreachable

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit: ; preds = %for.body16
  %47 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !16
  %48 = load double*, double** %6, align 8, !tbaa !9
  %mul.i.i.i = mul nsw i64 %45, %indvars.iv445
  %add.i.i.i = add nsw i64 %mul.i.i.i, %indvars.iv443
  %arrayidx.i.i.i296 = getelementptr inbounds double, double* %48, i64 %add.i.i.i
  %49 = load double, double* %arrayidx.i.i.i296, align 8, !tbaa !10
  %50 = trunc i64 %indvars.iv443 to i32
  %call20 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %47, i8* getelementptr inbounds ([18 x i8], [18 x i8]* @.str, i64 0, i64 0), i32 %50, i32 %43, double %49) #10
  %indvars.iv.next444 = add nuw nsw i64 %indvars.iv443, 1
  %cmp14 = icmp ult i64 %indvars.iv.next444, 3
  br i1 %cmp14, label %for.body16, label %for.cond.cleanup15

for.cond37.preheader:                             ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit301
  %51 = getelementptr inbounds %"class.Eigen::Matrix.6", %"class.Eigen::Matrix.6"* %O, i64 0, i32 0, i32 0, i32 0
  br label %for.body41

for.body29:                                       ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit301, %for.cond25.preheader
  %indvars.iv441 = phi i64 [ 0, %for.cond25.preheader ], [ %indvars.iv.next442, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit301 ]
  %52 = load i64, i64* %m_rows.i.i.i.i.i, align 8, !tbaa !15
  %cmp2.i298 = icmp sgt i64 %52, %indvars.iv441
  br i1 %cmp2.i298, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit301, label %cond.false.i299

cond.false.i299:                                  ; preds = %for.body29
  call void @__assert_fail(i8* getelementptr inbounds ([29 x i8], [29 x i8]* @.str.29, i64 0, i64 0), i8* getelementptr inbounds ([59 x i8], [59 x i8]* @.str.28, i64 0, i64 0), i32 425, i8* getelementptr inbounds ([209 x i8], [209 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl, i64 0, i64 0)) #9
  unreachable

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit301: ; preds = %for.body29
  %53 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !16
  %54 = load double*, double** %44, align 8, !tbaa !13
  %arrayidx.i.i.i300 = getelementptr inbounds double, double* %54, i64 %indvars.iv441
  %55 = load double, double* %arrayidx.i.i.i300, align 8, !tbaa !10
  %56 = trunc i64 %indvars.iv441 to i32
  %call32 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %53, i8* getelementptr inbounds ([12 x i8], [12 x i8]* @.str.1, i64 0, i64 0), i32 %56, double %55) #10
  %indvars.iv.next442 = add nuw nsw i64 %indvars.iv441, 1
  %cmp27 = icmp ult i64 %indvars.iv.next442, 7
  br i1 %cmp27, label %for.body29, label %for.cond37.preheader

for.body41:                                       ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit306, %for.cond37.preheader
  %indvars.iv439 = phi i64 [ 0, %for.cond37.preheader ], [ %indvars.iv.next440, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit306 ]
  %57 = load i64, i64* %m_rows.i.i.i.i.i210, align 8, !tbaa !15
  %cmp2.i303 = icmp sgt i64 %57, %indvars.iv439
  br i1 %cmp2.i303, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit306, label %cond.false.i304

cond.false.i304:                                  ; preds = %for.body41
  call void @__assert_fail(i8* getelementptr inbounds ([29 x i8], [29 x i8]* @.str.29, i64 0, i64 0), i8* getelementptr inbounds ([59 x i8], [59 x i8]* @.str.28, i64 0, i64 0), i32 425, i8* getelementptr inbounds ([209 x i8], [209 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl, i64 0, i64 0)) #9
  unreachable

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit306: ; preds = %for.body41
  %58 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !16
  %59 = load double*, double** %51, align 8, !tbaa !13
  %arrayidx.i.i.i305 = getelementptr inbounds double, double* %59, i64 %indvars.iv439
  %60 = load double, double* %arrayidx.i.i.i305, align 8, !tbaa !10
  %61 = trunc i64 %indvars.iv439 to i32
  %call44 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %58, i8* getelementptr inbounds ([12 x i8], [12 x i8]* @.str.2, i64 0, i64 0), i32 %61, double %60) #10
  %indvars.iv.next440 = add nuw nsw i64 %indvars.iv439, 1
  %cmp39 = icmp ult i64 %indvars.iv.next440, 3
  br i1 %cmp39, label %for.body41, label %for.cond55.preheader

for.cond55.preheader:                             ; preds = %for.cond.cleanup58, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit306
  %indvars.iv437 = phi i64 [ %indvars.iv.next438, %for.cond.cleanup58 ], [ 0, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit306 ]
  %62 = trunc i64 %indvars.iv437 to i32
  br label %for.body59

for.cond88.preheader:                             ; preds = %for.cond.cleanup58
  %63 = getelementptr inbounds %"class.Eigen::Matrix.6", %"class.Eigen::Matrix.6"* %Mp, i64 0, i32 0, i32 0, i32 0
  br label %for.cond94.preheader

for.cond.cleanup58:                               ; preds = %if.end
  %indvars.iv.next438 = add nuw nsw i64 %indvars.iv437, 1
  %cmp51 = icmp ult i64 %indvars.iv.next438, 7
  br i1 %cmp51, label %for.cond55.preheader, label %for.cond88.preheader

for.body59:                                       ; preds = %if.end, %for.cond55.preheader
  %indvars.iv435 = phi i64 [ 0, %for.cond55.preheader ], [ %indvars.iv.next436, %if.end ]
  %64 = load i64, i64* %m_rows.i.i.i107, align 8, !tbaa !2
  %cmp2.i308 = icmp sgt i64 %64, %indvars.iv435
  %65 = load i64, i64* %m_cols.i.i.i108, align 8
  %cmp7.i310 = icmp sgt i64 %65, %indvars.iv437
  %or.cond405 = and i1 %cmp2.i308, %cmp7.i310
  br i1 %or.cond405, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit316, label %cond.false.i312

cond.false.i312:                                  ; preds = %for.body59
  call void @__assert_fail(i8* getelementptr inbounds ([53 x i8], [53 x i8]* @.str.27, i64 0, i64 0), i8* getelementptr inbounds ([59 x i8], [59 x i8]* @.str.28, i64 0, i64 0), i32 365, i8* getelementptr inbounds ([227 x i8], [227 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll, i64 0, i64 0)) #9
  unreachable

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit316: ; preds = %for.body59
  %66 = load double*, double** %26, align 8, !tbaa !9
  %mul.i.i.i313 = mul nsw i64 %64, %indvars.iv437
  %add.i.i.i314 = add nsw i64 %mul.i.i.i313, %indvars.iv435
  %arrayidx.i.i.i315 = getelementptr inbounds double, double* %66, i64 %add.i.i.i314
  %67 = load double, double* %arrayidx.i.i.i315, align 8, !tbaa !10
  %68 = load i64, i64* %m_rows.i.i.i.i.i, align 8, !tbaa !15
  %cmp2.i318 = icmp sgt i64 %68, %indvars.iv437
  br i1 %cmp2.i318, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit321, label %cond.false.i319

cond.false.i319:                                  ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit316
  call void @__assert_fail(i8* getelementptr inbounds ([29 x i8], [29 x i8]* @.str.29, i64 0, i64 0), i8* getelementptr inbounds ([59 x i8], [59 x i8]* @.str.28, i64 0, i64 0), i32 425, i8* getelementptr inbounds ([209 x i8], [209 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl, i64 0, i64 0)) #9
  unreachable

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit321: ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit316
  %cmp2.i323 = icmp sgt i64 %.pr, %indvars.iv435
  br i1 %cmp2.i323, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit336, label %cond.false.i324

cond.false.i324:                                  ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit321
  call void @__assert_fail(i8* getelementptr inbounds ([29 x i8], [29 x i8]* @.str.29, i64 0, i64 0), i8* getelementptr inbounds ([59 x i8], [59 x i8]* @.str.28, i64 0, i64 0), i32 425, i8* getelementptr inbounds ([209 x i8], [209 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl, i64 0, i64 0)) #9
  unreachable

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit336: ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit321
  %69 = load double*, double** %44, align 8, !tbaa !13
  %arrayidx.i.i.i320 = getelementptr inbounds double, double* %69, i64 %indvars.iv437
  %70 = load double, double* %arrayidx.i.i.i320, align 8, !tbaa !10
  %arrayidx.i.i.i325 = getelementptr inbounds double, double* %41, i64 %indvars.iv435
  %71 = load double, double* %arrayidx.i.i.i325, align 8, !tbaa !10
  %mul = fmul fast double %71, %70
  %sub = fsub fast double %67, %mul
  %72 = call fast double @llvm.fabs.f64(double %sub)
  %cmp67 = fcmp fast ogt double %72, 1.000000e-10
  %73 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !16
  br i1 %cmp67, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit305, label %if.end

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit305: ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit336
  %call76 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %73, i8* getelementptr inbounds ([68 x i8], [68 x i8]* @.str.3, i64 0, i64 0), i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str.4, i64 0, i64 0), double %67, i8* getelementptr inbounds ([18 x i8], [18 x i8]* @.str.5, i64 0, i64 0), double %mul, double 1.000000e-10, i8* getelementptr inbounds ([65 x i8], [65 x i8]* @.str.6, i64 0, i64 0), i32 64, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @__PRETTY_FUNCTION__.main, i64 0, i64 0)) #10
  call void @abort() #9
  unreachable

if.end:                                           ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit336
  %74 = trunc i64 %indvars.iv435 to i32
  %call80 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %73, i8* getelementptr inbounds ([19 x i8], [19 x i8]* @.str.7, i64 0, i64 0), i32 %74, i32 %62, double %67) #10
  %indvars.iv.next436 = add nuw nsw i64 %indvars.iv435, 1
  %cmp57 = icmp ult i64 %indvars.iv.next436, 3
  br i1 %cmp57, label %for.body59, label %for.cond.cleanup58

for.cond94.preheader:                             ; preds = %if.end116, %for.cond88.preheader
  %indvars.iv433 = phi i64 [ 0, %for.cond88.preheader ], [ %indvars.iv.next434, %if.end116 ]
  %75 = load i64, i64* %m_rows.i.i.i, align 8, !tbaa !2
  %76 = load i64, i64* %m_cols.i.i.i, align 8
  %cmp7.i350 = icmp sgt i64 %76, %indvars.iv433
  %77 = load double*, double** %6, align 8
  %mul.i.i.i353 = mul nsw i64 %75, %indvars.iv433
  br i1 %cmp7.i350, label %for.body98.us, label %cond.false.i352

for.body98.us:                                    ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit361.us, %for.cond94.preheader
  %indvars.iv431 = phi i64 [ %indvars.iv.next432, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit361.us ], [ 0, %for.cond94.preheader ]
  %res.0396.us = phi double [ %add.us, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit361.us ], [ 0.000000e+00, %for.cond94.preheader ]
  %cmp2.i348.us = icmp sgt i64 %75, %indvars.iv431
  br i1 %cmp2.i348.us, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit356.us, label %cond.false.i352

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit356.us: ; preds = %for.body98.us
  %cmp2.i358.us = icmp sgt i64 %.pr, %indvars.iv431
  br i1 %cmp2.i358.us, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit361.us, label %cond.false.i359

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit361.us: ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit356.us
  %add.i.i.i354.us = add nsw i64 %mul.i.i.i353, %indvars.iv431
  %arrayidx.i.i.i355.us = getelementptr inbounds double, double* %77, i64 %add.i.i.i354.us
  %78 = load double, double* %arrayidx.i.i.i355.us, align 8, !tbaa !10
  %arrayidx.i.i.i360.us = getelementptr inbounds double, double* %41, i64 %indvars.iv431
  %79 = load double, double* %arrayidx.i.i.i360.us, align 8, !tbaa !10
  %mul104.us = fmul fast double %79, %78
  %add.us = fadd fast double %mul104.us, %res.0396.us
  %indvars.iv.next432 = add nuw nsw i64 %indvars.iv431, 1
  %cmp96.us = icmp ult i64 %indvars.iv.next432, 3
  br i1 %cmp96.us, label %for.body98.us, label %for.cond.cleanup97

for.cond124.preheader:                            ; preds = %if.end116
  %80 = getelementptr inbounds %"class.Eigen::Matrix.6", %"class.Eigen::Matrix.6"* %Op, i64 0, i32 0, i32 0, i32 0
  br label %for.body128

for.cond.cleanup97:                               ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit361.us
  %81 = load i64, i64* %m_rows.i.i.i.i.i242, align 8, !tbaa !15
  %cmp2.i338 = icmp sgt i64 %81, %indvars.iv433
  br i1 %cmp2.i338, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit346, label %cond.false.i339

cond.false.i339:                                  ; preds = %for.cond.cleanup97
  call void @__assert_fail(i8* getelementptr inbounds ([29 x i8], [29 x i8]* @.str.29, i64 0, i64 0), i8* getelementptr inbounds ([59 x i8], [59 x i8]* @.str.28, i64 0, i64 0), i32 425, i8* getelementptr inbounds ([209 x i8], [209 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl, i64 0, i64 0)) #9
  unreachable

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit346: ; preds = %for.cond.cleanup97
  %82 = load double*, double** %63, align 8, !tbaa !13
  %arrayidx.i.i.i340 = getelementptr inbounds double, double* %82, i64 %indvars.iv433
  %83 = load double, double* %arrayidx.i.i.i340, align 8, !tbaa !10
  %sub110 = fsub fast double %83, %add.us
  %84 = call fast double @llvm.fabs.f64(double %sub110)
  %cmp111 = fcmp fast ogt double %84, 1.000000e-10
  %85 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !16
  br i1 %cmp111, label %if.then112, label %if.end116

cond.false.i352:                                  ; preds = %for.body98.us, %for.cond94.preheader
  call void @__assert_fail(i8* getelementptr inbounds ([53 x i8], [53 x i8]* @.str.27, i64 0, i64 0), i8* getelementptr inbounds ([59 x i8], [59 x i8]* @.str.28, i64 0, i64 0), i32 365, i8* getelementptr inbounds ([227 x i8], [227 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll, i64 0, i64 0)) #9
  unreachable

cond.false.i359:                                  ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit356.us
  call void @__assert_fail(i8* getelementptr inbounds ([29 x i8], [29 x i8]* @.str.29, i64 0, i64 0), i8* getelementptr inbounds ([59 x i8], [59 x i8]* @.str.28, i64 0, i64 0), i32 425, i8* getelementptr inbounds ([209 x i8], [209 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl, i64 0, i64 0)) #9
  unreachable

if.then112:                                       ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit346
  %call115 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %85, i8* getelementptr inbounds ([68 x i8], [68 x i8]* @.str.3, i64 0, i64 0), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.8, i64 0, i64 0), double %83, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.9, i64 0, i64 0), double %add.us, double 1.000000e-10, i8* getelementptr inbounds ([65 x i8], [65 x i8]* @.str.6, i64 0, i64 0), i32 71, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @__PRETTY_FUNCTION__.main, i64 0, i64 0)) #10
  call void @abort() #9
  unreachable

if.end116:                                        ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit346
  %86 = trunc i64 %indvars.iv433 to i32
  %call119 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %85, i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.10, i64 0, i64 0), i32 %86, double %83) #10
  %indvars.iv.next434 = add nuw nsw i64 %indvars.iv433, 1
  %cmp90 = icmp ult i64 %indvars.iv.next434, 7
  br i1 %cmp90, label %for.cond94.preheader, label %for.cond124.preheader

for.cond.cleanup127:                              ; preds = %if.end137
  call void @free(i8* %call.i.i4.i.i.i.i) #8
  %87 = load i8*, i8** %36, align 8, !tbaa !13
  call void @free(i8* %87) #8
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %35) #8
  %88 = load i8*, i8** %31, align 8, !tbaa !13
  call void @free(i8* %88) #8
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %30) #8
  %89 = bitcast %"class.Eigen::Matrix"* %Wp to i8**
  %90 = load i8*, i8** %89, align 8, !tbaa !9
  call void @free(i8* %90) #8
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %20) #8
  %91 = load i8*, i8** %16, align 8, !tbaa !13
  call void @free(i8* %91) #8
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %15) #8
  %92 = load i8*, i8** %11, align 8, !tbaa !13
  call void @free(i8* %92) #8
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %10) #8
  %93 = bitcast %"class.Eigen::Matrix"* %W to i8**
  %94 = load i8*, i8** %93, align 8, !tbaa !9
  call void @free(i8* %94) #8
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %0) #8
  ret i32 0

for.body128:                                      ; preds = %if.end137, %for.cond124.preheader
  %indvars.iv = phi i64 [ 0, %for.cond124.preheader ], [ %indvars.iv.next, %if.end137 ]
  %95 = load i64, i64* %m_rows.i.i.i.i.i271, align 8, !tbaa !15
  %cmp2.i199 = icmp sgt i64 %95, %indvars.iv
  br i1 %cmp2.i199, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit, label %cond.false.i200

cond.false.i200:                                  ; preds = %for.body128
  call void @__assert_fail(i8* getelementptr inbounds ([29 x i8], [29 x i8]* @.str.29, i64 0, i64 0), i8* getelementptr inbounds ([59 x i8], [59 x i8]* @.str.28, i64 0, i64 0), i32 425, i8* getelementptr inbounds ([209 x i8], [209 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl, i64 0, i64 0)) #9
  unreachable

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit: ; preds = %for.body128
  %96 = load double*, double** %80, align 8, !tbaa !13
  %arrayidx.i.i.i201 = getelementptr inbounds double, double* %96, i64 %indvars.iv
  %97 = load double, double* %arrayidx.i.i.i201, align 8, !tbaa !10
  %98 = call fast double @llvm.fabs.f64(double %97)
  %cmp132 = fcmp fast ogt double %98, 1.000000e-10
  %99 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !16
  br i1 %cmp132, label %if.then133, label %if.end137

if.then133:                                       ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit
  %call136 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %99, i8* getelementptr inbounds ([68 x i8], [68 x i8]* @.str.3, i64 0, i64 0), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.11, i64 0, i64 0), double %97, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.12, i64 0, i64 0), double 0.000000e+00, double 1.000000e-10, i8* getelementptr inbounds ([65 x i8], [65 x i8]* @.str.6, i64 0, i64 0), i32 76, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @__PRETTY_FUNCTION__.main, i64 0, i64 0)) #10
  call void @abort() #9
  unreachable

if.end137:                                        ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit
  %100 = trunc i64 %indvars.iv to i32
  %call140 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %99, i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.13, i64 0, i64 0), i32 %100, double %97) #10
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %cmp126 = icmp ult i64 %indvars.iv.next, 3
  br i1 %cmp126, label %for.body128, label %for.cond.cleanup127
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

; Function Attrs: alwaysinline
declare dso_local double @__enzyme_autodiff(i8*, i8*, i8*, i8*, i8*, i8*, i8*) local_unnamed_addr #2

; Function Attrs: noinline nounwind uwtable
define internal void @_ZL6matvecPKN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEEPKNS0_IdLin1ELi1ELi0ELin1ELi1EEEPS4_(%"class.Eigen::Matrix"* noalias %W, %"class.Eigen::Matrix.6"* noalias %b, %"class.Eigen::Matrix.6"* noalias %output) #3 {
entry:
  %ref.tmp.i.i.i.i.i.i.i.i.i.i = alloca %"class.Eigen::internal::const_blas_data_mapper", align 8
  %ref.tmp10.i.i.i.i.i.i.i.i.i.i = alloca %"class.Eigen::internal::const_blas_data_mapper.32", align 8
  %m_cols.i.i.i.i = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %W, i64 0, i32 0, i32 0, i32 2
  %m_rows.i.i.i.i.i.i.i.i.i = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %W, i64 0, i32 0, i32 0, i32 1
  %0 = load i64, i64* %m_rows.i.i.i.i.i.i.i.i.i, align 8, !tbaa !2
  %cmp.i.i.i.i.i.i = icmp eq i64 %0, 0
  %mul.i.i.i.i = shl i64 %0, 3
  %call.i.i4.i.i.i.i = call noalias i8* @malloc(i64 %mul.i.i.i.i) #8
  %1 = bitcast %"class.Eigen::Matrix.6"* %b to i64*
  %div.i.i.i.i6.i = sdiv i64 %0, 2
  %mul.i.i.i.i7.i = shl nsw i64 %div.i.i.i.i6.i, 1
  %cmp13.i.i.i.i8.i = icmp sgt i64 %0, 1
  %2 = bitcast i8* %call.i.i4.i.i.i.i to double*
  br label %for.body.i.i.i.i22.i

for.cond.cleanup.i.i.i.i11.i:                     ; preds = %for.body.i.i.i.i22.i
  %cmp4.i.i.i.i.i10.i = icmp slt i64 %mul.i.i.i.i7.i, %0
  br i1 %cmp4.i.i.i.i.i10.i, label %for.body.i.i.i.i.i17.i, label %_ZN5Eigen8internal24call_assignment_no_aliasINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EENS0_9assign_opIddEEEEvRT_RKT0_RKT1_.exit.i

for.body.i.i.i.i.i17.i:                           ; preds = %for.body.i.i.i.i.i17.i, %for.cond.cleanup.i.i.i.i11.i
  %index.05.i.i.i.i.i13.i = phi i64 [ %inc.i.i.i.i.i15.i, %for.body.i.i.i.i.i17.i ], [ %mul.i.i.i.i7.i, %for.cond.cleanup.i.i.i.i11.i ]
  %arrayidx.i.i.i.i.i.i.i14.i = getelementptr inbounds double, double* %2, i64 %index.05.i.i.i.i.i13.i
  %3 = bitcast double* %arrayidx.i.i.i.i.i.i.i14.i to i64*
  store i64 0, i64* %3, align 8, !tbaa !10
  %inc.i.i.i.i.i15.i = add nsw i64 %index.05.i.i.i.i.i13.i, 1
  %exitcond.i.i.i.i.i16.i = icmp eq i64 %inc.i.i.i.i.i15.i, %0
  br i1 %exitcond.i.i.i.i.i16.i, label %_ZN5Eigen8internal24call_assignment_no_aliasINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EENS0_9assign_opIddEEEEvRT_RKT0_RKT1_.exit.i, label %for.body.i.i.i.i.i17.i

for.body.i.i.i.i22.i:                             ; preds = %for.body.i.i.i.i22.i, %entry
  %index.014.i.i.i.i18.i = phi i64 [ %add1.i.i.i.i20.i, %for.body.i.i.i.i22.i ], [ 0, %entry ]
  %arrayidx.i.i.i.i.i.i19.i = getelementptr inbounds double, double* %2, i64 %index.014.i.i.i.i18.i
  %4 = bitcast double* %arrayidx.i.i.i.i.i.i19.i to <2 x double>*
  store <2 x double> zeroinitializer, <2 x double>* %4, align 16, !tbaa !12
  %add1.i.i.i.i20.i = add nuw nsw i64 %index.014.i.i.i.i18.i, 2
  %cmp.i.i.i.i21.i = icmp slt i64 %add1.i.i.i.i20.i, %mul.i.i.i.i7.i
  br i1 %cmp.i.i.i.i21.i, label %for.body.i.i.i.i22.i, label %for.cond.cleanup.i.i.i.i11.i

_ZN5Eigen8internal24call_assignment_no_aliasINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EENS0_9assign_opIddEEEEvRT_RKT0_RKT1_.exit.i: ; preds = %for.body.i.i.i.i.i17.i, %for.cond.cleanup.i.i.i.i11.i
  %5 = load i64, i64* %m_cols.i.i.i.i, align 8, !tbaa !8
  %6 = bitcast %"class.Eigen::internal::const_blas_data_mapper"* %ref.tmp.i.i.i.i.i.i.i.i.i.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %6) #8
  %7 = bitcast %"class.Eigen::Matrix"* %W to i64*
  %8 = load i64, i64* %7, align 8, !tbaa !9
  %9 = bitcast %"class.Eigen::internal::const_blas_data_mapper"* %ref.tmp.i.i.i.i.i.i.i.i.i.i to i64*
  store i64 %8, i64* %9, align 8, !tbaa !17
  %m_stride.i.i26.i.i.i.i.i.i.i.i.i.i = getelementptr inbounds %"class.Eigen::internal::const_blas_data_mapper", %"class.Eigen::internal::const_blas_data_mapper"* %ref.tmp.i.i.i.i.i.i.i.i.i.i, i64 0, i32 0, i32 1
  store i64 %0, i64* %m_stride.i.i26.i.i.i.i.i.i.i.i.i.i, align 8, !tbaa !19
  %10 = bitcast %"class.Eigen::internal::const_blas_data_mapper.32"* %ref.tmp10.i.i.i.i.i.i.i.i.i.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %10) #8
  %11 = load i64, i64* %1, align 8, !tbaa !13
  %12 = bitcast %"class.Eigen::internal::const_blas_data_mapper.32"* %ref.tmp10.i.i.i.i.i.i.i.i.i.i to i64*
  store i64 %11, i64* %12, align 8, !tbaa !20
  %m_stride.i.i.i.i.i.i.i.i.i.i.i.i = getelementptr inbounds %"class.Eigen::internal::const_blas_data_mapper.32", %"class.Eigen::internal::const_blas_data_mapper.32"* %ref.tmp10.i.i.i.i.i.i.i.i.i.i, i64 0, i32 0, i32 1
  store i64 1, i64* %m_stride.i.i.i.i.i.i.i.i.i.i.i.i, align 8, !tbaa !22
  call void @subfn(i64 %0, i64 %5, %"class.Eigen::internal::const_blas_data_mapper"* nonnull dereferenceable(16) %ref.tmp.i.i.i.i.i.i.i.i.i.i, %"class.Eigen::internal::const_blas_data_mapper.32"* nonnull dereferenceable(16) %ref.tmp10.i.i.i.i.i.i.i.i.i.i, double* %2) #8
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %10) #8
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %6) #8
  %m_rows.i.i21.i.i.i.i.i = getelementptr inbounds %"class.Eigen::Matrix.6", %"class.Eigen::Matrix.6"* %output, i64 0, i32 0, i32 0, i32 1
  %13 = load i64, i64* %m_rows.i.i21.i.i.i.i.i, align 8, !tbaa !15
  %cmp.i5.i.i.i.i = icmp eq i64 %13, %0
  br i1 %cmp.i5.i.i.i.i, label %_ZN5Eigen8internal17resize_if_allowedINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEES3_ddEEvRT_RKT0_RKNS0_9assign_opIT1_T2_EE.exit.i.i.i.i, label %if.end.i.i.i.i.i

if.end.i.i.i.i.i:                                 ; preds = %_ZN5Eigen8internal24call_assignment_no_aliasINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EENS0_9assign_opIddEEEEvRT_RKT0_RKT1_.exit.i
  %14 = bitcast %"class.Eigen::Matrix.6"* %output to i8**
  %15 = load i8*, i8** %14, align 8, !tbaa !13
  call void @free(i8* %15) #8
  br i1 %cmp.i.i.i.i.i.i, label %if.else.i.i31.i, label %if.end.i.i.i12.i

if.end.i.i.i12.i:                                 ; preds = %if.end.i.i.i.i.i
  %call.i.i4.i.i.i16.i = call noalias i8* @malloc(i64 %mul.i.i.i.i) #8
  store i8* %call.i.i4.i.i.i16.i, i8** %14, align 8, !tbaa !13
  br label %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEEE6resizeEll.exit32.i

if.else.i.i31.i:                                  ; preds = %if.end.i.i.i.i.i
  %m_data.i.i30.i = getelementptr inbounds %"class.Eigen::Matrix.6", %"class.Eigen::Matrix.6"* %output, i64 0, i32 0, i32 0, i32 0
  store double* null, double** %m_data.i.i30.i, align 8, !tbaa !13
  br label %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEEE6resizeEll.exit32.i

_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEEE6resizeEll.exit32.i: ; preds = %if.else.i.i31.i, %if.end.i.i.i12.i
  store i64 %0, i64* %m_rows.i.i21.i.i.i.i.i, align 8, !tbaa !15
  br label %_ZN5Eigen8internal17resize_if_allowedINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEES3_ddEEvRT_RKT0_RKNS0_9assign_opIT1_T2_EE.exit.i.i.i.i

_ZN5Eigen8internal17resize_if_allowedINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEES3_ddEEvRT_RKT0_RKNS0_9assign_opIT1_T2_EE.exit.i.i.i.i: ; preds = %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEEE6resizeEll.exit32.i, %_ZN5Eigen8internal24call_assignment_no_aliasINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EENS0_9assign_opIddEEEEvRT_RKT0_RKT1_.exit.i
  %16 = bitcast %"class.Eigen::Matrix.6"* %output to i64*
  %17 = load i64, i64* %16, align 8, !tbaa !13
  br i1 %cmp13.i.i.i.i8.i, label %for.body.i.preheader.i.i.i.i, label %for.cond.cleanup.i.i.i.i.i

for.body.i.preheader.i.i.i.i:                     ; preds = %_ZN5Eigen8internal17resize_if_allowedINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEES3_ddEEvRT_RKT0_RKNS0_9assign_opIT1_T2_EE.exit.i.i.i.i
  %18 = inttoptr i64 %17 to double*
  br label %for.body.i.i.i.i.i

for.cond.cleanup.i.i.i.i.i:                       ; preds = %for.body.i.i.i.i.i, %_ZN5Eigen8internal17resize_if_allowedINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEES3_ddEEvRT_RKT0_RKNS0_9assign_opIT1_T2_EE.exit.i.i.i.i
  br i1 %cmp4.i.i.i.i.i10.i, label %for.body.lr.ph.i.i.i.i.i.i, label %_ZN5Eigen8internal15call_assignmentINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEENS_7ProductINS2_IdLin1ELin1ELi0ELin1ELin1EEES3_Li0EEENS0_9assign_opIddEEEEvRT_RKT0_RKT1_NS0_9enable_ifIXsr25evaluator_assume_aliasingISB_EE5valueEPvE4typeE.exit

for.body.lr.ph.i.i.i.i.i.i:                       ; preds = %for.cond.cleanup.i.i.i.i.i
  %19 = inttoptr i64 %17 to double*
  br label %for.body.i.i.i.i.i.i

for.body.i.i.i.i.i.i:                             ; preds = %for.body.i.i.i.i.i.i, %for.body.lr.ph.i.i.i.i.i.i
  %index.05.i.i.i.i.i.i = phi i64 [ %mul.i.i.i.i7.i, %for.body.lr.ph.i.i.i.i.i.i ], [ %inc.i.i.i.i.i.i, %for.body.i.i.i.i.i.i ]
  %arrayidx.i.i.i.i.i.i.i.i = getelementptr inbounds double, double* %19, i64 %index.05.i.i.i.i.i.i
  %arrayidx.i5.i.i.i.i.i.i.i = getelementptr inbounds double, double* %2, i64 %index.05.i.i.i.i.i.i
  %20 = bitcast double* %arrayidx.i5.i.i.i.i.i.i.i to i64*
  %21 = load i64, i64* %20, align 8, !tbaa !10
  %22 = bitcast double* %arrayidx.i.i.i.i.i.i.i.i to i64*
  store i64 %21, i64* %22, align 8, !tbaa !10
  %inc.i.i.i.i.i.i = add nsw i64 %index.05.i.i.i.i.i.i, 1
  %exitcond.i.i.i.i.i.i = icmp eq i64 %inc.i.i.i.i.i.i, %0
  br i1 %exitcond.i.i.i.i.i.i, label %_ZN5Eigen8internal15call_assignmentINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEENS_7ProductINS2_IdLin1ELin1ELi0ELin1ELin1EEES3_Li0EEENS0_9assign_opIddEEEEvRT_RKT0_RKT1_NS0_9enable_ifIXsr25evaluator_assume_aliasingISB_EE5valueEPvE4typeE.exit, label %for.body.i.i.i.i.i.i

for.body.i.i.i.i.i:                               ; preds = %for.body.i.i.i.i.i, %for.body.i.preheader.i.i.i.i
  %index.014.i.i.i.i.i = phi i64 [ %add1.i.i.i.i.i, %for.body.i.i.i.i.i ], [ 0, %for.body.i.preheader.i.i.i.i ]
  %arrayidx.i.i.i.i.i.i.i = getelementptr inbounds double, double* %18, i64 %index.014.i.i.i.i.i
  %add.ptr.i.i.i.i.i.i.i = getelementptr inbounds double, double* %2, i64 %index.014.i.i.i.i.i
  %23 = bitcast double* %add.ptr.i.i.i.i.i.i.i to <2 x double>*
  %24 = load <2 x double>, <2 x double>* %23, align 16, !tbaa !12
  %25 = bitcast double* %arrayidx.i.i.i.i.i.i.i to <2 x double>*
  store <2 x double> %24, <2 x double>* %25, align 16, !tbaa !12
  %add1.i.i.i.i.i = add nuw nsw i64 %index.014.i.i.i.i.i, 2
  %cmp.i.i.i.i.i = icmp slt i64 %add1.i.i.i.i.i, %mul.i.i.i.i7.i
  br i1 %cmp.i.i.i.i.i, label %for.body.i.i.i.i.i, label %for.cond.cleanup.i.i.i.i.i

_ZN5Eigen8internal15call_assignmentINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEENS_7ProductINS2_IdLin1ELin1ELi0ELin1ELin1EEES3_Li0EEENS0_9assign_opIddEEEEvRT_RKT0_RKT1_NS0_9enable_ifIXsr25evaluator_assume_aliasingISB_EE5valueEPvE4typeE.exit: ; preds = %for.body.i.i.i.i.i.i, %for.cond.cleanup.i.i.i.i.i
  call void @free(i8* %call.i.i4.i.i.i.i) #8
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

; Function Attrs: nobuiltin
declare dso_local noalias nonnull i8* @_Znwm(i64) local_unnamed_addr #7

; Function Attrs: alwaysinline nounwind
declare dso_local void @free(i8* nocapture) local_unnamed_addr #4

; Function Attrs: alwaysinline nounwind
declare dso_local noalias i8* @malloc(i64) local_unnamed_addr #4

; Function Attrs: noinline nounwind uwtable
define void @subfn(i64 %rows, i64 %cols, %"class.Eigen::internal::const_blas_data_mapper"* dereferenceable(16) %lhs, %"class.Eigen::internal::const_blas_data_mapper.32"* dereferenceable(16) %rhs, double* %res) #3 {
entry:
  %a0 = bitcast %"class.Eigen::internal::const_blas_data_mapper"* %lhs to i64*
  %theload = load i64, i64* %a0, align 8, !tbaa !17
  %toptr = inttoptr i64 %theload to double*
  %and2.i.i.i = and i64 %theload, 1
  %cmp11 = icmp slt i64 %and2.i.i.i, %rows
  br i1 %cmp11, label %land.end, label %if.end22

land.end:                                         ; preds = %land.end, %entry
  br label %land.end

if.end22:                                         ; preds = %entry
  %m_data.i835 = getelementptr inbounds %"class.Eigen::internal::const_blas_data_mapper.32", %"class.Eigen::internal::const_blas_data_mapper.32"* %rhs, i64 0, i32 0, i32 0
  %a2 = load double*, double** %m_data.i835, align 8, !tbaa !20
  %a3 = load double, double* %a2, align 8, !tbaa !10
  %arrayidx.i.i810 = getelementptr inbounds double, double* %toptr, i64 1
  %a4 = load double, double* %toptr, align 8, !tbaa !10
  %mul.i.i.i794 = fadd fast double %a4, %a3
  store double %mul.i.i.i794, double* %res, align 8, !tbaa !10
  %a5 = bitcast double* %arrayidx.i.i810 to i64*
  %a71 = load i64, i64* %a5, align 8, !tbaa !10
  %a6 = bitcast double* %res to i64*
  store i64 %a71, i64* %a6, align 8, !tbaa !10
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1) #1

attributes #0 = { alwaysinline norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="128" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { alwaysinline "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #3 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="128" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #4 = { alwaysinline nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #5 = { nounwind readnone speculatable }
attributes #6 = { alwaysinline noreturn nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #7 = { nobuiltin "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
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
!10 = !{!11, !11, i64 0}
!11 = !{!"double", !5, i64 0}
!12 = !{!5, !5, i64 0}
!13 = !{!14, !4, i64 0}
!14 = !{!"_ZTSN5Eigen12DenseStorageIdLin1ELin1ELi1ELi0EEE", !4, i64 0, !7, i64 8}
!15 = !{!14, !7, i64 8}
!16 = !{!4, !4, i64 0}
!17 = !{!18, !4, i64 0}
!18 = !{!"_ZTSN5Eigen8internal16blas_data_mapperIKdlLi0ELi0EEE", !4, i64 0, !7, i64 8}
!19 = !{!18, !7, i64 8}
!20 = !{!21, !4, i64 0}
!21 = !{!"_ZTSN5Eigen8internal16blas_data_mapperIKdlLi1ELi0EEE", !4, i64 0, !7, i64 8}
!22 = !{!21, !7, i64 8}

; CHECK: define internal void @diffe_ZL6matvecPKN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEEPKNS0_IdLin1ELi1ELi0ELin1ELi1EEEPS4_(%"class.Eigen::Matrix"* noalias %W, %"class.Eigen::Matrix"* %"W'", %"class.Eigen::Matrix.6"* noalias %b, %"class.Eigen::Matrix.6"* %"b'", %"class.Eigen::Matrix.6"* noalias %output, %"class.Eigen::Matrix.6"* %"output'")
