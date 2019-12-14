; RUN: %opt < %s %loadEnzyme -enzyme -enzyme_preopt=false -mem2reg -sroa -simplifycfg -instcombine -adce -S | FileCheck %s

source_filename = "/home/wmoses/Enzyme/enzyme/test/Integration/simpleeigen-made.cpp"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }
%"struct.Eigen::internal::evaluator.36" = type <{ %"struct.Eigen::internal::scalar_constant_op", %"struct.Eigen::internal::nullary_wrapper", [7 x i8] }>
%"struct.Eigen::internal::scalar_constant_op" = type { double }
%"struct.Eigen::internal::nullary_wrapper" = type { i8 }
%"struct.Eigen::internal::evaluator.39" = type { %"struct.Eigen::internal::evaluator.40" }
%"struct.Eigen::internal::evaluator.40" = type { double*, %"class.Eigen::internal::variable_if_dynamic" }
%"class.Eigen::internal::variable_if_dynamic" = type { i64 }
%"class.Eigen::internal::generic_dense_assignment_kernel.42" = type { %"struct.Eigen::internal::evaluator.39"*, %"struct.Eigen::internal::evaluator.36"*, %"struct.Eigen::internal::assign_op"*, %"class.Eigen::Matrix"* }
%"struct.Eigen::internal::assign_op" = type { i8 }
%"class.Eigen::Matrix" = type { %"class.Eigen::PlainObjectBase" }
%"class.Eigen::PlainObjectBase" = type { %"class.Eigen::DenseStorage" }
%"class.Eigen::DenseStorage" = type { double*, i64, i64 }

@stderr = external dso_local local_unnamed_addr global %struct._IO_FILE*, align 8
@.str = private unnamed_addr constant [18 x i8] c"W(o=%d, i=%d)=%f\0A\00", align 1
@.str.1 = private unnamed_addr constant [12 x i8] c"M(o=%d)=%f\0A\00", align 1
@.str.2 = private unnamed_addr constant [12 x i8] c"O(i=%d)=%f\0A\00", align 1
@.str.3 = private unnamed_addr constant [68 x i8] c"Assertion Failed: fabs( [%s = %g] - [%s = %g] ) > %g at %s:%d (%s)\0A\00", align 1
@.str.4 = private unnamed_addr constant [9 x i8] c"Wp(i, o)\00", align 1
@.str.5 = private unnamed_addr constant [18 x i8] c"M(o) * Op_orig(i)\00", align 1
@.str.6 = private unnamed_addr constant [65 x i8] c"/home/wmoses/Enzyme/enzyme/test/Integration/simpleeigen-made.cpp\00", align 1
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

; Function Attrs: norecurse nounwind uwtable
define dso_local i32 @main(i32 %argc, i8** nocapture readnone %argv) local_unnamed_addr #0 {
entry:
  %srcEvaluator.i.i.i.i.i.i35 = alloca %"struct.Eigen::internal::evaluator.36", align 8
  %dstEvaluator.i.i.i.i.i.i36 = alloca %"struct.Eigen::internal::evaluator.39", align 8
  %kernel.i.i.i.i.i.i37 = alloca %"class.Eigen::internal::generic_dense_assignment_kernel.42", align 8
  %ref.tmp.i.i.i38 = alloca %"struct.Eigen::internal::assign_op", align 1
  %srcEvaluator.i.i.i.i.i.i = alloca %"struct.Eigen::internal::evaluator.36", align 8
  %dstEvaluator.i.i.i.i.i.i = alloca %"struct.Eigen::internal::evaluator.39", align 8
  %kernel.i.i.i.i.i.i = alloca %"class.Eigen::internal::generic_dense_assignment_kernel.42", align 8
  %ref.tmp.i.i.i = alloca %"struct.Eigen::internal::assign_op", align 1
  %W = alloca %"class.Eigen::Matrix", align 8
  %Wp = alloca %"class.Eigen::Matrix", align 8
  %0 = bitcast %"class.Eigen::Matrix"* %W to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %0) #10
  call void @llvm.memset.p0i8.i64(i8* align 8 %0, i8 0, i64 24, i1 false) #10
  %m_rows.i.i.i = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %W, i64 0, i32 0, i32 0, i32 1
  %1 = load i64, i64* %m_rows.i.i.i, align 8, !tbaa !2
  %m_cols.i.i.i = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %W, i64 0, i32 0, i32 0, i32 2
  %2 = load i64, i64* %m_cols.i.i.i, align 8, !tbaa !8
  %mul.i.i.i1 = mul nsw i64 %2, %1
  %cmp.i1.i.i = icmp eq i64 %mul.i.i.i1, 4
  br i1 %cmp.i1.i.i, label %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEE10resizeLikeINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEEvRKNS_9EigenBaseIT_EE.exit, label %if.then.i2.i.i

if.then.i2.i.i:                                   ; preds = %entry
  %m_data.i.i.i2 = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %W, i64 0, i32 0, i32 0, i32 0
  %3 = bitcast %"class.Eigen::Matrix"* %W to i8**
  %4 = load i8*, i8** %3, align 8, !tbaa !9
  call void @free(i8* %4) #10
  %call.i.i.i1 = call noalias i8* @malloc(i64 32) #10
  %tobool.i.i.i = icmp eq i8* %call.i.i.i1, null
  br i1 %tobool.i.i.i, label %if.then.i.i.i, label %_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit

if.then.i.i.i:                                    ; preds = %if.then.i2.i.i
  %call.i1.i = call i8* @_Znwm(i64 -1) #10
  br label %_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit

_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit: ; preds = %if.then.i.i.i, %if.then.i2.i.i
  %5 = bitcast i8* %call.i.i.i1 to double*
  store double* %5, double** %m_data.i.i.i2, align 8, !tbaa !9
  br label %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEE10resizeLikeINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEEvRKNS_9EigenBaseIT_EE.exit

_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEE10resizeLikeINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEEvRKNS_9EigenBaseIT_EE.exit: ; preds = %_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit, %entry
  store i64 2, i64* %m_rows.i.i.i, align 8, !tbaa !2
  store i64 2, i64* %m_cols.i.i.i, align 8, !tbaa !8
  %6 = getelementptr inbounds %"struct.Eigen::internal::assign_op", %"struct.Eigen::internal::assign_op"* %ref.tmp.i.i.i, i64 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %6) #10
  %7 = bitcast %"struct.Eigen::internal::evaluator.36"* %srcEvaluator.i.i.i.i.i.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %7) #10
  %8 = bitcast %"struct.Eigen::internal::evaluator.36"* %srcEvaluator.i.i.i.i.i.i to i64*
  store i64 4613937818241073152, i64* %8, align 8, !tbaa !10
  %9 = bitcast %"struct.Eigen::internal::evaluator.39"* %dstEvaluator.i.i.i.i.i.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %9) #10
  %10 = bitcast %"class.Eigen::Matrix"* %W to i64*
  %11 = load i64, i64* %10, align 8, !tbaa !9
  %12 = bitcast %"struct.Eigen::internal::evaluator.39"* %dstEvaluator.i.i.i.i.i.i to i64*
  store i64 %11, i64* %12, align 8, !tbaa !13
  %m_value.i.i.i39 = getelementptr inbounds %"struct.Eigen::internal::evaluator.39", %"struct.Eigen::internal::evaluator.39"* %dstEvaluator.i.i.i.i.i.i, i64 0, i32 0, i32 1, i32 0
  store i64 2, i64* %m_value.i.i.i39, align 8, !tbaa !16
  %13 = bitcast %"class.Eigen::internal::generic_dense_assignment_kernel.42"* %kernel.i.i.i.i.i.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %13) #10
  %m_dst.i = getelementptr inbounds %"class.Eigen::internal::generic_dense_assignment_kernel.42", %"class.Eigen::internal::generic_dense_assignment_kernel.42"* %kernel.i.i.i.i.i.i, i64 0, i32 0
  store %"struct.Eigen::internal::evaluator.39"* %dstEvaluator.i.i.i.i.i.i, %"struct.Eigen::internal::evaluator.39"** %m_dst.i, align 8, !tbaa !17
  %m_src.i = getelementptr inbounds %"class.Eigen::internal::generic_dense_assignment_kernel.42", %"class.Eigen::internal::generic_dense_assignment_kernel.42"* %kernel.i.i.i.i.i.i, i64 0, i32 1
  store %"struct.Eigen::internal::evaluator.36"* %srcEvaluator.i.i.i.i.i.i, %"struct.Eigen::internal::evaluator.36"** %m_src.i, align 8, !tbaa !17
  %m_functor.i = getelementptr inbounds %"class.Eigen::internal::generic_dense_assignment_kernel.42", %"class.Eigen::internal::generic_dense_assignment_kernel.42"* %kernel.i.i.i.i.i.i, i64 0, i32 2
  store %"struct.Eigen::internal::assign_op"* %ref.tmp.i.i.i, %"struct.Eigen::internal::assign_op"** %m_functor.i, align 8, !tbaa !17
  %m_dstExpr.i = getelementptr inbounds %"class.Eigen::internal::generic_dense_assignment_kernel.42", %"class.Eigen::internal::generic_dense_assignment_kernel.42"* %kernel.i.i.i.i.i.i, i64 0, i32 3
  store %"class.Eigen::Matrix"* %W, %"class.Eigen::Matrix"** %m_dstExpr.i, align 8, !tbaa !17
  br label %for.body.i.i.i.i.i.i.i

for.body.i.i.i.i.i.i.i:                           ; preds = %for.body.i.i.i.i.i.i.i, %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEE10resizeLikeINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEEvRKNS_9EigenBaseIT_EE.exit
  %index.014.i.i.i.i.i.i.i = phi i64 [ %add1.i.i.i.i.i.i.i, %for.body.i.i.i.i.i.i.i ], [ 0, %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEE10resizeLikeINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEEvRKNS_9EigenBaseIT_EE.exit ]
  %14 = bitcast %"class.Eigen::internal::generic_dense_assignment_kernel.42"* %kernel.i.i.i.i.i.i to %"struct.Eigen::internal::evaluator.40"**
  %15 = load %"struct.Eigen::internal::evaluator.40"*, %"struct.Eigen::internal::evaluator.40"** %14, align 8, !tbaa !18
  %m_data.i.i.i1.i.i.i.i.i.i = getelementptr inbounds %"struct.Eigen::internal::evaluator.40", %"struct.Eigen::internal::evaluator.40"* %15, i64 0, i32 0
  %16 = load double*, double** %m_data.i.i.i1.i.i.i.i.i.i, align 8, !tbaa !13
  %arrayidx.i.i.i.i.i.i.i.i.i = getelementptr inbounds double, double* %16, i64 %index.014.i.i.i.i.i.i.i
  %17 = load %"struct.Eigen::internal::evaluator.36"*, %"struct.Eigen::internal::evaluator.36"** %m_src.i, align 8, !tbaa !20
  %m_other.i.i.i.i.i.i.i.i.i.i.i = getelementptr inbounds %"struct.Eigen::internal::evaluator.36", %"struct.Eigen::internal::evaluator.36"* %17, i64 0, i32 0, i32 0
  %18 = load double, double* %m_other.i.i.i.i.i.i.i.i.i.i.i, align 8, !tbaa !21
  %vecinit.i.i.i.i.i.i.i.i.i.i.i.i.i = insertelement <2 x double> undef, double %18, i32 0
  %vecinit1.i.i.i.i.i.i.i.i.i.i.i.i.i = shufflevector <2 x double> %vecinit.i.i.i.i.i.i.i.i.i.i.i.i.i, <2 x double> undef, <2 x i32> zeroinitializer
  %19 = bitcast double* %arrayidx.i.i.i.i.i.i.i.i.i to <2 x double>*
  store <2 x double> %vecinit1.i.i.i.i.i.i.i.i.i.i.i.i.i, <2 x double>* %19, align 16, !tbaa !22
  %add1.i.i.i.i.i.i.i = add nuw nsw i64 %index.014.i.i.i.i.i.i.i, 2
  %cmp.i2.i.i.i.i.i.i = icmp ult i64 %add1.i.i.i.i.i.i.i, 4
  br i1 %cmp.i2.i.i.i.i.i.i, label %for.body.i.i.i.i.i.i.i, label %_ZN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEC2INS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES1_EEEERKNS_9EigenBaseIT_EE.exit

_ZN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEC2INS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES1_EEEERKNS_9EigenBaseIT_EE.exit: ; preds = %for.body.i.i.i.i.i.i.i
  call void @kern(%"class.Eigen::internal::generic_dense_assignment_kernel.42"* nonnull dereferenceable(32) %kernel.i.i.i.i.i.i, i64 4, i64 4) #10
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %13) #10
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %9) #10
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %7) #10
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %6) #10
  %call.i.i.i = call noalias i8* @malloc(i64 16) #10
  %tobool.i.i.i2 = icmp eq i8* %call.i.i.i, null
  br i1 %tobool.i.i.i2, label %if.then.i.i.i4, label %_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit5

if.then.i.i.i4:                                   ; preds = %_ZN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEC2INS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES1_EEEERKNS_9EigenBaseIT_EE.exit
  %call.i1.i3 = call i8* @_Znwm(i64 -1) #10
  br label %_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit5

_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit5: ; preds = %if.then.i.i.i4, %_ZN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEC2INS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES1_EEEERKNS_9EigenBaseIT_EE.exit
  %20 = bitcast i8* %call.i.i.i to double*
  %21 = bitcast double* %20 to <2 x double>*
  store <2 x double> <double 2.000000e+00, double 2.000000e+00>, <2 x double>* %21, align 16, !tbaa !22
  %call.i.i.i6 = call noalias i8* @malloc(i64 16) #10
  %tobool.i.i.i7 = icmp eq i8* %call.i.i.i6, null
  br i1 %tobool.i.i.i7, label %if.then.i.i.i9, label %_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit10

if.then.i.i.i9:                                   ; preds = %_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit5
  %call.i1.i8 = call i8* @_Znwm(i64 -1) #10
  br label %_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit10

_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit10: ; preds = %if.then.i.i.i9, %_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit5
  %22 = bitcast i8* %call.i.i.i6 to double*
  %23 = bitcast double* %22 to <2 x double>*
  store <2 x double> zeroinitializer, <2 x double>* %23, align 16, !tbaa !22
  %24 = bitcast %"class.Eigen::Matrix"* %Wp to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %24) #10
  call void @llvm.memset.p0i8.i64(i8* align 8 %24, i8 0, i64 24, i1 false) #10
  %m_rows.i.i.i6 = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %Wp, i64 0, i32 0, i32 0, i32 1
  %25 = load i64, i64* %m_rows.i.i.i6, align 8, !tbaa !2
  %m_cols.i.i.i7 = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %Wp, i64 0, i32 0, i32 0, i32 2
  %26 = load i64, i64* %m_cols.i.i.i7, align 8, !tbaa !8
  %mul.i.i.i8 = mul nsw i64 %26, %25
  %cmp.i1.i.i9 = icmp eq i64 %mul.i.i.i8, 4
  br i1 %cmp.i1.i.i9, label %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEE10resizeLikeINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEEvRKNS_9EigenBaseIT_EE.exit17, label %if.then.i2.i.i12

if.then.i2.i.i12:                                 ; preds = %_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit10
  %m_data.i.i.i10 = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %Wp, i64 0, i32 0, i32 0, i32 0
  %27 = bitcast %"class.Eigen::Matrix"* %Wp to i8**
  %28 = load i8*, i8** %27, align 8, !tbaa !9
  call void @free(i8* %28) #10
  %call.i.i.i11 = call noalias i8* @malloc(i64 32) #10
  %tobool.i.i.i12 = icmp eq i8* %call.i.i.i11, null
  br i1 %tobool.i.i.i12, label %if.then.i.i.i14, label %_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit15

if.then.i.i.i14:                                  ; preds = %if.then.i2.i.i12
  %call.i1.i13 = call i8* @_Znwm(i64 -1) #10
  br label %_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit15

_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit15: ; preds = %if.then.i.i.i14, %if.then.i2.i.i12
  %29 = bitcast i8* %call.i.i.i11 to double*
  store double* %29, double** %m_data.i.i.i10, align 8, !tbaa !9
  br label %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEE10resizeLikeINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEEvRKNS_9EigenBaseIT_EE.exit17

_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEE10resizeLikeINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEEvRKNS_9EigenBaseIT_EE.exit17: ; preds = %_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit15, %_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit10
  store i64 2, i64* %m_rows.i.i.i6, align 8, !tbaa !2
  store i64 2, i64* %m_cols.i.i.i7, align 8, !tbaa !8
  %30 = getelementptr inbounds %"struct.Eigen::internal::assign_op", %"struct.Eigen::internal::assign_op"* %ref.tmp.i.i.i38, i64 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %30) #10
  %31 = bitcast %"struct.Eigen::internal::evaluator.36"* %srcEvaluator.i.i.i.i.i.i35 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %31) #10
  %32 = bitcast %"struct.Eigen::internal::evaluator.36"* %srcEvaluator.i.i.i.i.i.i35 to i64*
  store i64 0, i64* %32, align 8, !tbaa !10
  %33 = bitcast %"struct.Eigen::internal::evaluator.39"* %dstEvaluator.i.i.i.i.i.i36 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %33) #10
  %34 = bitcast %"class.Eigen::Matrix"* %Wp to i64*
  %35 = load i64, i64* %34, align 8, !tbaa !9
  %36 = bitcast %"struct.Eigen::internal::evaluator.39"* %dstEvaluator.i.i.i.i.i.i36 to i64*
  store i64 %35, i64* %36, align 8, !tbaa !13
  %m_value.i.i.i52 = getelementptr inbounds %"struct.Eigen::internal::evaluator.39", %"struct.Eigen::internal::evaluator.39"* %dstEvaluator.i.i.i.i.i.i36, i64 0, i32 0, i32 1, i32 0
  store i64 2, i64* %m_value.i.i.i52, align 8, !tbaa !16
  %37 = bitcast %"class.Eigen::internal::generic_dense_assignment_kernel.42"* %kernel.i.i.i.i.i.i37 to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %37) #10
  %m_dst.i47 = getelementptr inbounds %"class.Eigen::internal::generic_dense_assignment_kernel.42", %"class.Eigen::internal::generic_dense_assignment_kernel.42"* %kernel.i.i.i.i.i.i37, i64 0, i32 0
  store %"struct.Eigen::internal::evaluator.39"* %dstEvaluator.i.i.i.i.i.i36, %"struct.Eigen::internal::evaluator.39"** %m_dst.i47, align 8, !tbaa !17
  %m_src.i48 = getelementptr inbounds %"class.Eigen::internal::generic_dense_assignment_kernel.42", %"class.Eigen::internal::generic_dense_assignment_kernel.42"* %kernel.i.i.i.i.i.i37, i64 0, i32 1
  store %"struct.Eigen::internal::evaluator.36"* %srcEvaluator.i.i.i.i.i.i35, %"struct.Eigen::internal::evaluator.36"** %m_src.i48, align 8, !tbaa !17
  %m_functor.i49 = getelementptr inbounds %"class.Eigen::internal::generic_dense_assignment_kernel.42", %"class.Eigen::internal::generic_dense_assignment_kernel.42"* %kernel.i.i.i.i.i.i37, i64 0, i32 2
  store %"struct.Eigen::internal::assign_op"* %ref.tmp.i.i.i38, %"struct.Eigen::internal::assign_op"** %m_functor.i49, align 8, !tbaa !17
  %m_dstExpr.i50 = getelementptr inbounds %"class.Eigen::internal::generic_dense_assignment_kernel.42", %"class.Eigen::internal::generic_dense_assignment_kernel.42"* %kernel.i.i.i.i.i.i37, i64 0, i32 3
  store %"class.Eigen::Matrix"* %Wp, %"class.Eigen::Matrix"** %m_dstExpr.i50, align 8, !tbaa !17
  br label %for.body.i.i.i.i.i.i.i96

for.body.i.i.i.i.i.i.i96:                         ; preds = %for.body.i.i.i.i.i.i.i96, %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEE10resizeLikeINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEEvRKNS_9EigenBaseIT_EE.exit17
  %index.014.i.i.i.i.i.i.i87 = phi i64 [ %add1.i.i.i.i.i.i.i94, %for.body.i.i.i.i.i.i.i96 ], [ 0, %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEE10resizeLikeINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEEvRKNS_9EigenBaseIT_EE.exit17 ]
  %38 = bitcast %"class.Eigen::internal::generic_dense_assignment_kernel.42"* %kernel.i.i.i.i.i.i37 to %"struct.Eigen::internal::evaluator.40"**
  %39 = load %"struct.Eigen::internal::evaluator.40"*, %"struct.Eigen::internal::evaluator.40"** %38, align 8, !tbaa !18
  %m_data.i.i.i1.i.i.i.i.i.i88 = getelementptr inbounds %"struct.Eigen::internal::evaluator.40", %"struct.Eigen::internal::evaluator.40"* %39, i64 0, i32 0
  %40 = load double*, double** %m_data.i.i.i1.i.i.i.i.i.i88, align 8, !tbaa !13
  %arrayidx.i.i.i.i.i.i.i.i.i89 = getelementptr inbounds double, double* %40, i64 %index.014.i.i.i.i.i.i.i87
  %41 = load %"struct.Eigen::internal::evaluator.36"*, %"struct.Eigen::internal::evaluator.36"** %m_src.i48, align 8, !tbaa !20
  %m_other.i.i.i.i.i.i.i.i.i.i.i91 = getelementptr inbounds %"struct.Eigen::internal::evaluator.36", %"struct.Eigen::internal::evaluator.36"* %41, i64 0, i32 0, i32 0
  %42 = load double, double* %m_other.i.i.i.i.i.i.i.i.i.i.i91, align 8, !tbaa !21
  %vecinit.i.i.i.i.i.i.i.i.i.i.i.i.i92 = insertelement <2 x double> undef, double %42, i32 0
  %vecinit1.i.i.i.i.i.i.i.i.i.i.i.i.i93 = shufflevector <2 x double> %vecinit.i.i.i.i.i.i.i.i.i.i.i.i.i92, <2 x double> undef, <2 x i32> zeroinitializer
  %43 = bitcast double* %arrayidx.i.i.i.i.i.i.i.i.i89 to <2 x double>*
  store <2 x double> %vecinit1.i.i.i.i.i.i.i.i.i.i.i.i.i93, <2 x double>* %43, align 16, !tbaa !22
  %add1.i.i.i.i.i.i.i94 = add nuw nsw i64 %index.014.i.i.i.i.i.i.i87, 2
  %cmp.i2.i.i.i.i.i.i95 = icmp ult i64 %add1.i.i.i.i.i.i.i94, 4
  br i1 %cmp.i2.i.i.i.i.i.i95, label %for.body.i.i.i.i.i.i.i96, label %_ZN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEC2INS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES1_EEEERKNS_9EigenBaseIT_EE.exit98

_ZN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEC2INS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES1_EEEERKNS_9EigenBaseIT_EE.exit98: ; preds = %for.body.i.i.i.i.i.i.i96
  call void @kern(%"class.Eigen::internal::generic_dense_assignment_kernel.42"* nonnull dereferenceable(32) %kernel.i.i.i.i.i.i37, i64 4, i64 4) #10
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %37) #10
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %33) #10
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %31) #10
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %30) #10
  %call.i.i.i16 = call noalias i8* @malloc(i64 16) #10
  %tobool.i.i.i17 = icmp eq i8* %call.i.i.i16, null
  br i1 %tobool.i.i.i17, label %if.then.i.i.i19, label %_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit20

if.then.i.i.i19:                                  ; preds = %_ZN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEC2INS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES1_EEEERKNS_9EigenBaseIT_EE.exit98
  %call.i1.i18 = call i8* @_Znwm(i64 -1) #10
  br label %_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit20

_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit20: ; preds = %if.then.i.i.i19, %_ZN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEC2INS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES1_EEEERKNS_9EigenBaseIT_EE.exit98
  %44 = bitcast i8* %call.i.i.i16 to double*
  %45 = bitcast double* %44 to <2 x double>*
  store <2 x double> zeroinitializer, <2 x double>* %45, align 16, !tbaa !22
  %call.i.i.i21 = call noalias i8* @malloc(i64 16) #10
  %tobool.i.i.i22 = icmp eq i8* %call.i.i.i21, null
  br i1 %tobool.i.i.i22, label %if.then.i.i.i24, label %_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit25

if.then.i.i.i24:                                  ; preds = %_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit20
  %call.i1.i23 = call i8* @_Znwm(i64 -1) #10
  br label %_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit25

_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit25: ; preds = %if.then.i.i.i24, %_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit20
  %46 = bitcast i8* %call.i.i.i21 to double*
  %47 = bitcast double* %46 to <2 x double>*
  store <2 x double> <double 1.000000e+00, double 1.000000e+00>, <2 x double>* %47, align 16, !tbaa !22
  %call.i.i.i26 = call noalias i8* @malloc(i64 16) #10
  %tobool.i.i.i27 = icmp eq i8* %call.i.i.i26, null
  br i1 %tobool.i.i.i27, label %if.then.i.i.i29, label %_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit30

if.then.i.i.i29:                                  ; preds = %_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit25
  %call.i1.i28 = call i8* @_Znwm(i64 -1) #10
  br label %_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit30

_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit30: ; preds = %if.then.i.i.i29, %_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit25
  %48 = bitcast i8* %call.i.i.i26 to double*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %call.i.i.i26, i8* align 8 %call.i.i.i21, i64 16, i1 false) #10
  %Wdp = bitcast %"class.Eigen::Matrix"* %W to double**
  %Wz = load double*, double** %Wdp, align 8, !tbaa !9
  %pWdp = bitcast %"class.Eigen::Matrix"* %Wp to double**
  %pWz = load double*, double** %pWdp, align 8, !tbaa !9
  %call = call double @__enzyme_autodiff(i8* bitcast (void (double*, double*, double*)* @matvec to i8*), double* %Wz, double* %pWz, double* %20, double* %44, double* %22, double* %46) #10
  br label %for.cond12.preheader

for.cond12.preheader:                             ; preds = %for.cond.cleanup15, %_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit30
  %indvars.iv250 = phi i64 [ 0, %_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit30 ], [ %indvars.iv.next251, %for.cond.cleanup15 ]
  %49 = trunc i64 %indvars.iv250 to i32
  br label %for.body16

for.cond.cleanup15:                               ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit
  %indvars.iv.next251 = add nuw nsw i64 %indvars.iv250, 1
  %exitcond252 = icmp eq i64 %indvars.iv.next251, 2
  br i1 %exitcond252, label %for.body29, label %for.cond12.preheader

for.body16:                                       ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit, %for.cond12.preheader
  %indvars.iv247 = phi i64 [ 0, %for.cond12.preheader ], [ %indvars.iv.next248, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit ]
  %50 = load i64, i64* %m_rows.i.i.i, align 8, !tbaa !2
  %cmp2.i = icmp sgt i64 %50, %indvars.iv247
  %51 = load i64, i64* %m_cols.i.i.i, align 8
  %cmp7.i = icmp sgt i64 %51, %indvars.iv250
  %or.cond = and i1 %cmp2.i, %cmp7.i
  br i1 %or.cond, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit, label %cond.false.i

cond.false.i:                                     ; preds = %for.body16
  call void @__assert_fail(i8* getelementptr inbounds ([53 x i8], [53 x i8]* @.str.27, i64 0, i64 0), i8* getelementptr inbounds ([59 x i8], [59 x i8]* @.str.28, i64 0, i64 0), i32 365, i8* getelementptr inbounds ([227 x i8], [227 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll, i64 0, i64 0)) #11
  unreachable

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit: ; preds = %for.body16
  %52 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !17
  %53 = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %W, i64 0, i32 0, i32 0, i32 0
  %54 = load double*, double** %53, align 8, !tbaa !9
  %mul.i.i.i = mul nsw i64 %50, %indvars.iv250
  %add.i.i.i = add nsw i64 %mul.i.i.i, %indvars.iv247
  %arrayidx.i.i.i = getelementptr inbounds double, double* %54, i64 %add.i.i.i
  %55 = load double, double* %arrayidx.i.i.i, align 8, !tbaa !21
  %56 = trunc i64 %indvars.iv247 to i32
  %call20 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %52, i8* getelementptr inbounds ([18 x i8], [18 x i8]* @.str, i64 0, i64 0), i32 %56, i32 %49, double %55) #12
  %indvars.iv.next248 = add nuw nsw i64 %indvars.iv247, 1
  %exitcond249 = icmp eq i64 %indvars.iv.next248, 2
  br i1 %exitcond249, label %for.cond.cleanup15, label %for.body16

for.body29:                                       ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit, %for.cond.cleanup15
  %indvars.iv244 = phi i64 [ %indvars.iv.next245, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit ], [ 0, %for.cond.cleanup15 ]
  %cmp2.i153 = icmp sgt i64 2, %indvars.iv244
  br i1 %cmp2.i153, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit, label %cond.false.i155

cond.false.i155:                                  ; preds = %for.body29
  call void @__assert_fail(i8* getelementptr inbounds ([29 x i8], [29 x i8]* @.str.29, i64 0, i64 0), i8* getelementptr inbounds ([59 x i8], [59 x i8]* @.str.28, i64 0, i64 0), i32 425, i8* getelementptr inbounds ([209 x i8], [209 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl, i64 0, i64 0)) #11
  unreachable

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit: ; preds = %for.body29
  %57 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !17
  %arrayidx.i.i.i158 = getelementptr inbounds double, double* %20, i64 %indvars.iv244
  %58 = load double, double* %arrayidx.i.i.i158, align 8, !tbaa !21
  %59 = trunc i64 %indvars.iv244 to i32
  %call32 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %57, i8* getelementptr inbounds ([12 x i8], [12 x i8]* @.str.1, i64 0, i64 0), i32 %59, double %58) #12
  %indvars.iv.next245 = add nuw nsw i64 %indvars.iv244, 1
  %exitcond246 = icmp eq i64 %indvars.iv.next245, 2
  br i1 %exitcond246, label %for.body41, label %for.body29

for.body41:                                       ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit175, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit
  %indvars.iv241 = phi i64 [ %indvars.iv.next242, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit175 ], [ 0, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit ]
  %cmp2.i167 = icmp sgt i64 2, %indvars.iv241
  br i1 %cmp2.i167, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit175, label %cond.false.i169

cond.false.i169:                                  ; preds = %for.body41
  call void @__assert_fail(i8* getelementptr inbounds ([29 x i8], [29 x i8]* @.str.29, i64 0, i64 0), i8* getelementptr inbounds ([59 x i8], [59 x i8]* @.str.28, i64 0, i64 0), i32 425, i8* getelementptr inbounds ([209 x i8], [209 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl, i64 0, i64 0)) #11
  unreachable

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit175: ; preds = %for.body41
  %60 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !17
  %arrayidx.i.i.i174 = getelementptr inbounds double, double* %22, i64 %indvars.iv241
  %61 = load double, double* %arrayidx.i.i.i174, align 8, !tbaa !21
  %62 = trunc i64 %indvars.iv241 to i32
  %call44 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %60, i8* getelementptr inbounds ([12 x i8], [12 x i8]* @.str.2, i64 0, i64 0), i32 %62, double %61) #12
  %indvars.iv.next242 = add nuw nsw i64 %indvars.iv241, 1
  %exitcond243 = icmp eq i64 %indvars.iv.next242, 2
  br i1 %exitcond243, label %for.cond55.preheader, label %for.body41

for.cond55.preheader:                             ; preds = %for.cond.cleanup58, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit175
  %indvars.iv239 = phi i64 [ %indvars.iv.next240, %for.cond.cleanup58 ], [ 0, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit175 ]
  %63 = trunc i64 %indvars.iv239 to i32
  br label %for.body59

for.cond.cleanup58:                               ; preds = %if.end
  %indvars.iv.next240 = add nuw nsw i64 %indvars.iv239, 1
  %cmp51 = icmp ult i64 %indvars.iv.next240, 2
  br i1 %cmp51, label %for.cond55.preheader, label %for.cond94.preheader

for.body59:                                       ; preds = %if.end, %for.cond55.preheader
  %indvars.iv237 = phi i64 [ 0, %for.cond55.preheader ], [ %indvars.iv.next238, %if.end ]
  %64 = load i64, i64* %m_rows.i.i.i6, align 8, !tbaa !2
  %cmp2.i181 = icmp sgt i64 %64, %indvars.iv237
  %65 = load i64, i64* %m_cols.i.i.i7, align 8
  %cmp7.i188 = icmp sgt i64 %65, %indvars.iv239
  %or.cond1 = and i1 %cmp2.i181, %cmp7.i188
  br i1 %or.cond1, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit197, label %cond.false.i190

cond.false.i190:                                  ; preds = %for.body59
  call void @__assert_fail(i8* getelementptr inbounds ([53 x i8], [53 x i8]* @.str.27, i64 0, i64 0), i8* getelementptr inbounds ([59 x i8], [59 x i8]* @.str.28, i64 0, i64 0), i32 365, i8* getelementptr inbounds ([227 x i8], [227 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll, i64 0, i64 0)) #11
  unreachable

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit197: ; preds = %for.body59
  %66 = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %Wp, i64 0, i32 0, i32 0, i32 0
  %67 = load double*, double** %66, align 8, !tbaa !9
  %mul.i.i.i194 = mul nsw i64 %64, %indvars.iv239
  %add.i.i.i195 = add nsw i64 %mul.i.i.i194, %indvars.iv237
  %arrayidx.i.i.i196 = getelementptr inbounds double, double* %67, i64 %add.i.i.i195
  %68 = load double, double* %arrayidx.i.i.i196, align 8, !tbaa !21
  %cmp2.i206 = icmp sgt i64 2, %indvars.iv239
  br i1 %cmp2.i206, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit214, label %cond.false.i208

cond.false.i208:                                  ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit197
  call void @__assert_fail(i8* getelementptr inbounds ([29 x i8], [29 x i8]* @.str.29, i64 0, i64 0), i8* getelementptr inbounds ([59 x i8], [59 x i8]* @.str.28, i64 0, i64 0), i32 425, i8* getelementptr inbounds ([209 x i8], [209 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl, i64 0, i64 0)) #11
  unreachable

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit214: ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit197
  %cmp2.i223 = icmp sgt i64 2, %indvars.iv237
  br i1 %cmp2.i223, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit231, label %cond.false.i225

cond.false.i225:                                  ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit214
  call void @__assert_fail(i8* getelementptr inbounds ([29 x i8], [29 x i8]* @.str.29, i64 0, i64 0), i8* getelementptr inbounds ([59 x i8], [59 x i8]* @.str.28, i64 0, i64 0), i32 425, i8* getelementptr inbounds ([209 x i8], [209 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl, i64 0, i64 0)) #11
  unreachable

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit231: ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit214
  %69 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !17
  %arrayidx.i.i.i230 = getelementptr inbounds double, double* %48, i64 %indvars.iv237
  %arrayidx.i.i.i213 = getelementptr inbounds double, double* %20, i64 %indvars.iv239
  %70 = load double, double* %arrayidx.i.i.i213, align 8, !tbaa !21
  %71 = load double, double* %arrayidx.i.i.i230, align 8, !tbaa !21
  %mul = fmul double %70, %71
  %sub = fsub double %68, %mul
  %72 = call double @llvm.fabs.f64(double %sub)
  %cmp67 = fcmp ogt double %72, 1.000000e-10
  br i1 %cmp67, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit287, label %if.end

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit287: ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit231
  %call76 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %69, i8* getelementptr inbounds ([68 x i8], [68 x i8]* @.str.3, i64 0, i64 0), i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str.4, i64 0, i64 0), double %68, i8* getelementptr inbounds ([18 x i8], [18 x i8]* @.str.5, i64 0, i64 0), double %mul, double 1.000000e-10, i8* getelementptr inbounds ([65 x i8], [65 x i8]* @.str.6, i64 0, i64 0), i32 65, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @__PRETTY_FUNCTION__.main, i64 0, i64 0)) #12
  call void @abort() #11
  unreachable

if.end:                                           ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit231
  %73 = trunc i64 %indvars.iv237 to i32
  %call80 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %69, i8* getelementptr inbounds ([19 x i8], [19 x i8]* @.str.7, i64 0, i64 0), i32 %73, i32 %63, double %68) #12
  %indvars.iv.next238 = add nuw nsw i64 %indvars.iv237, 1
  %cmp57 = icmp ult i64 %indvars.iv.next238, 2
  br i1 %cmp57, label %for.body59, label %for.cond.cleanup58

for.cond94.preheader:                             ; preds = %if.end116, %for.cond.cleanup58
  %indvars.iv235 = phi i64 [ %indvars.iv.next236, %if.end116 ], [ 0, %for.cond.cleanup58 ]
  br label %for.body98

land.lhs.true.i297:                               ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit360
  %cmp2.i296 = icmp sgt i64 2, %indvars.iv235
  br i1 %cmp2.i296, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit304, label %cond.false.i298

cond.false.i298:                                  ; preds = %land.lhs.true.i297
  call void @__assert_fail(i8* getelementptr inbounds ([29 x i8], [29 x i8]* @.str.29, i64 0, i64 0), i8* getelementptr inbounds ([59 x i8], [59 x i8]* @.str.28, i64 0, i64 0), i32 425, i8* getelementptr inbounds ([209 x i8], [209 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl, i64 0, i64 0)) #11
  unreachable

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit304: ; preds = %land.lhs.true.i297
  %arrayidx.i.i.i303 = getelementptr inbounds double, double* %44, i64 %indvars.iv235
  %74 = load double, double* %arrayidx.i.i.i303, align 8, !tbaa !21
  %75 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !17
  %sub110 = fsub double %74, %add
  %76 = call double @llvm.fabs.f64(double %sub110)
  %cmp111 = fcmp ogt double %76, 1.000000e-10
  br i1 %cmp111, label %if.then112, label %if.end116

for.body98:                                       ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit360, %for.cond94.preheader
  %indvars.iv233 = phi i64 [ 0, %for.cond94.preheader ], [ %indvars.iv.next234, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit360 ]
  %res.0208 = phi double [ 0.000000e+00, %for.cond94.preheader ], [ %add, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit360 ]
  %77 = load i64, i64* %m_rows.i.i.i, align 8, !tbaa !2
  %cmp2.i327 = icmp sgt i64 %77, %indvars.iv233
  %78 = load i64, i64* %m_cols.i.i.i, align 8
  %cmp7.i334 = icmp sgt i64 %78, %indvars.iv235
  %or.cond3 = and i1 %cmp2.i327, %cmp7.i334
  br i1 %or.cond3, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit343, label %cond.false.i336

cond.false.i336:                                  ; preds = %for.body98
  call void @__assert_fail(i8* getelementptr inbounds ([53 x i8], [53 x i8]* @.str.27, i64 0, i64 0), i8* getelementptr inbounds ([59 x i8], [59 x i8]* @.str.28, i64 0, i64 0), i32 365, i8* getelementptr inbounds ([227 x i8], [227 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll, i64 0, i64 0)) #11
  unreachable

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit343: ; preds = %for.body98
  %cmp2.i352 = icmp sgt i64 2, %indvars.iv233
  br i1 %cmp2.i352, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit360, label %cond.false.i354

cond.false.i354:                                  ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit343
  call void @__assert_fail(i8* getelementptr inbounds ([29 x i8], [29 x i8]* @.str.29, i64 0, i64 0), i8* getelementptr inbounds ([59 x i8], [59 x i8]* @.str.28, i64 0, i64 0), i32 425, i8* getelementptr inbounds ([209 x i8], [209 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl, i64 0, i64 0)) #11
  unreachable

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit360: ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1EEclEll.exit343
  %79 = load double*, double** %53, align 8, !tbaa !9
  %mul.i.i.i340 = mul nsw i64 %77, %indvars.iv235
  %add.i.i.i341 = add nsw i64 %mul.i.i.i340, %indvars.iv233
  %arrayidx.i.i.i342 = getelementptr inbounds double, double* %79, i64 %add.i.i.i341
  %80 = load double, double* %arrayidx.i.i.i342, align 8, !tbaa !21
  %arrayidx.i.i.i359 = getelementptr inbounds double, double* %48, i64 %indvars.iv233
  %81 = load double, double* %arrayidx.i.i.i359, align 8, !tbaa !21
  %mul104 = fmul double %80, %81
  %add = fadd double %res.0208, %mul104
  %indvars.iv.next234 = add nuw nsw i64 %indvars.iv233, 1
  %exitcond = icmp eq i64 %indvars.iv.next234, 2
  br i1 %exitcond, label %land.lhs.true.i297, label %for.body98

if.then112:                                       ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit304
  %call115 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %75, i8* getelementptr inbounds ([68 x i8], [68 x i8]* @.str.3, i64 0, i64 0), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.8, i64 0, i64 0), double %74, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.9, i64 0, i64 0), double %add, double 1.000000e-10, i8* getelementptr inbounds ([65 x i8], [65 x i8]* @.str.6, i64 0, i64 0), i32 72, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @__PRETTY_FUNCTION__.main, i64 0, i64 0)) #12
  call void @abort() #11
  unreachable

if.end116:                                        ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit304
  %82 = trunc i64 %indvars.iv235 to i32
  %call119 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %75, i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.10, i64 0, i64 0), i32 %82, double %74) #12
  %indvars.iv.next236 = add nuw nsw i64 %indvars.iv235, 1
  %cmp90 = icmp ult i64 %indvars.iv.next236, 2
  br i1 %cmp90, label %for.cond94.preheader, label %for.body128

for.cond.cleanup127:                              ; preds = %if.end137
  call void @free(i8* %call.i.i.i26) #10
  call void @free(i8* %call.i.i.i21) #10
  call void @free(i8* %call.i.i.i16) #10
  %83 = bitcast %"class.Eigen::Matrix"* %Wp to i8**
  %84 = load i8*, i8** %83, align 8, !tbaa !9
  call void @free(i8* %84) #10
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %24) #10
  call void @free(i8* %call.i.i.i6) #10
  call void @free(i8* %call.i.i.i) #10
  %85 = bitcast %"class.Eigen::Matrix"* %W to i8**
  %86 = load i8*, i8** %85, align 8, !tbaa !9
  call void @free(i8* %86) #10
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %0) #10
  ret i32 0

for.body128:                                      ; preds = %if.end137, %if.end116
  %indvars.iv = phi i64 [ %indvars.iv.next, %if.end137 ], [ 0, %if.end116 ]
  %cmp2.i384 = icmp sgt i64 2, %indvars.iv
  br i1 %cmp2.i384, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit392, label %cond.false.i386

cond.false.i386:                                  ; preds = %for.body128
  call void @__assert_fail(i8* getelementptr inbounds ([29 x i8], [29 x i8]* @.str.29, i64 0, i64 0), i8* getelementptr inbounds ([59 x i8], [59 x i8]* @.str.28, i64 0, i64 0), i32 425, i8* getelementptr inbounds ([209 x i8], [209 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl, i64 0, i64 0)) #11
  unreachable

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit392: ; preds = %for.body128
  %arrayidx.i.i.i391 = getelementptr inbounds double, double* %46, i64 %indvars.iv
  %87 = load double, double* %arrayidx.i.i.i391, align 8, !tbaa !21
  %88 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !17
  %89 = call double @llvm.fabs.f64(double %87)
  %cmp132 = fcmp ogt double %89, 1.000000e-10
  br i1 %cmp132, label %if.then133, label %if.end137

if.then133:                                       ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit392
  %call136 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %88, i8* getelementptr inbounds ([68 x i8], [68 x i8]* @.str.3, i64 0, i64 0), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.11, i64 0, i64 0), double %87, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.12, i64 0, i64 0), double 0.000000e+00, double 1.000000e-10, i8* getelementptr inbounds ([65 x i8], [65 x i8]* @.str.6, i64 0, i64 0), i32 77, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @__PRETTY_FUNCTION__.main, i64 0, i64 0)) #12
  call void @abort() #11
  unreachable

if.end137:                                        ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEclEl.exit392
  %90 = trunc i64 %indvars.iv to i32
  %call140 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %88, i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.13, i64 0, i64 0), i32 %90, double %87) #12
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %cmp126 = icmp ult i64 %indvars.iv.next, 2
  br i1 %cmp126, label %for.body128, label %for.cond.cleanup127
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

declare dso_local double @__enzyme_autodiff(i8*, double*, double*, double*, double*, double*, double*) local_unnamed_addr #2

; Function Attrs: noinline nounwind uwtable
define internal void @matvec(double* %W, double* noalias %b, double* noalias %output) #3 {
entry:
  %tmp.i.i.i.i = alloca double*, align 8
  call void @subfn(double** %tmp.i.i.i.i, double* %W, i64 2, double* %b) #10
  %0 = load double*, double** %tmp.i.i.i.i, align 8, !tbaa !23
  call void @copydouble(double* %output, double* %0, i64 16) #10
  ret void
}

define dso_local void @copydouble(double* %dst, double* %src, i64 %len) {
entry:
  %dstp = bitcast double* %dst to i8*
  %srcp = bitcast double* %src to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %dstp, i8* align 1 %srcp, i64 %len, i1 false)
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

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local void @subfn(double** %dst, double* noalias %W, i64 %dd, double* %b) local_unnamed_addr #7 {
entry:
  %call.i.i3 = tail call double* @inneralloc() #10
  store double* %call.i.i3, double** %dst
  call void @vprod(i64 2, i64 2, double* nonnull %W, double* %b, double* %call.i.i3) #10
  ret void
}

; Function Attrs: inlinehint nounwind uwtable
define double* @inneralloc() local_unnamed_addr #7 {
entry:
  %call.i.i = tail call noalias i8* @malloc(i64 16) #10
  %0 = bitcast i8* %call.i.i to double*
  ret double* %0
}

; Function Attrs: nobuiltin
declare dso_local noalias nonnull i8* @_Znwm(i64) local_unnamed_addr #8

; Function Attrs: nounwind
declare dso_local void @free(i8* nocapture) local_unnamed_addr #4

; Function Attrs: nounwind
declare dso_local noalias i8* @malloc(i64) local_unnamed_addr #4

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local void @noop(double* %a1) local_unnamed_addr #9 {
entry:
  ret void
}

; Function Attrs: noinline nounwind uwtable
define linkonce_odr dso_local void @vprod(i64 %rows, i64 %cols, double* %W, double* %B, double* %res) local_unnamed_addr #3 {
entry:
  %W12p = bitcast double* %W to <2 x double>*

  %W12 = load <2 x double>, <2 x double>* %W12p, align 16

  %W34p = getelementptr inbounds <2 x double>, <2 x double>* %W12p, i64 1

  %W34 = load <2 x double>, <2 x double>* %W34p, align 16

  %B1 = load double, double* %B
  %preb1 = insertelement <2 x double> undef, double %B1, i32 0
  %B11 = shufflevector <2 x double> %preb1, <2 x double> undef, <2 x i32> zeroinitializer

  %B2p = getelementptr inbounds double, double* %B, i64 1
  %B2 = load double, double* %B2p
  %preb2 = insertelement <2 x double> undef, double %B2, i32 0
  %B22 = shufflevector <2 x double> %preb2, <2 x double> undef, <2 x i32> zeroinitializer

  %za5 = bitcast double* %res to <2 x double>*
  %za6 = load <2 x double>, <2 x double>* %za5
  %zmul.i.i.i.i3 = fmul <2 x double> %W12, %B11
  %zadd.i.i.i.i4 = fadd <2 x double> %zmul.i.i.i.i3, %za6


  %mul.i.i.i.i3 = fmul <2 x double> %W34, %B22
  %add.i.i.i.i4 = fadd <2 x double> %mul.i.i.i.i3, %zadd.i.i.i.i4
  store <2 x double> %add.i.i.i.i4, <2 x double>* %za5
  ret void
}

define void @nothing(double* %W) {
entry:
  ret void
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local void @kern(%"class.Eigen::internal::generic_dense_assignment_kernel.42"* dereferenceable(32) %kernel, i64 %start, i64 %end) local_unnamed_addr #7 {
entry:
  %cmp4 = icmp slt i64 %start, %end
  br i1 %cmp4, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body, %entry
  %index.05 = phi i64 [ %inc, %for.body ], [ %start, %entry ]
  %0 = bitcast %"class.Eigen::internal::generic_dense_assignment_kernel.42"* %kernel to %"struct.Eigen::internal::evaluator.40"**
  %1 = load %"struct.Eigen::internal::evaluator.40"*, %"struct.Eigen::internal::evaluator.40"** %0, align 8, !tbaa !18
  %m_data.i.i = getelementptr inbounds %"struct.Eigen::internal::evaluator.40", %"struct.Eigen::internal::evaluator.40"* %1, i64 0, i32 0
  %2 = load double*, double** %m_data.i.i, align 8, !tbaa !13
  %arrayidx.i.i = getelementptr inbounds double, double* %2, i64 %index.05
  %m_src.i = getelementptr inbounds %"class.Eigen::internal::generic_dense_assignment_kernel.42", %"class.Eigen::internal::generic_dense_assignment_kernel.42"* %kernel, i64 0, i32 1
  %3 = bitcast %"struct.Eigen::internal::evaluator.36"** %m_src.i to i64**
  %4 = load i64*, i64** %3, align 8, !tbaa !20
  %5 = load i64, i64* %4, align 8, !tbaa !10
  %6 = bitcast double* %arrayidx.i.i to i64*
  store i64 %5, i64* %6, align 8, !tbaa !21
  %inc = add nsw i64 %index.05, 1
  %exitcond = icmp eq i64 %inc, %end
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1) #1

attributes #0 = { norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="128" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="128" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { nounwind readnone speculatable }
attributes #6 = { noreturn nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #7 = { inlinehint nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #8 = { nobuiltin "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #9 = { inlinehint nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="128" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #10 = { nounwind }
attributes #11 = { noreturn nounwind }
attributes #12 = { cold }

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
!10 = !{!11, !12, i64 0}
!11 = !{!"_ZTSN5Eigen8internal18scalar_constant_opIdEE", !12, i64 0}
!12 = !{!"double", !5, i64 0}
!13 = !{!14, !4, i64 0}
!14 = !{!"_ZTSN5Eigen8internal9evaluatorINS_15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEEEEE", !4, i64 0, !15, i64 8}
!15 = !{!"_ZTSN5Eigen8internal19variable_if_dynamicIlLin1EEE", !7, i64 0}
!16 = !{!15, !7, i64 0}
!17 = !{!4, !4, i64 0}
!18 = !{!19, !4, i64 0}
!19 = !{!"_ZTSN5Eigen8internal31generic_dense_assignment_kernelINS0_9evaluatorINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEEENS2_INS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES4_EEEENS0_9assign_opIddEELi0EEE", !4, i64 0, !4, i64 8, !4, i64 16, !4, i64 24}
!20 = !{!19, !4, i64 8}
!21 = !{!12, !12, i64 0}
!22 = !{!5, !5, i64 0}
!23 = !{!24, !4, i64 0}
!24 = !{!"_ZTSN5Eigen12DenseStorageIdLin1ELin1ELi1ELi0EEE", !4, i64 0, !7, i64 8}

; CHECK: define internal { { i8* }, double*, double* } @augmented_inneralloc()
; CHECK-NEXT: entry:
; CHECK-NEXT:   %call.i.i = tail call noalias i8* @malloc(i64 16) #10
; CHECK-NEXT:   %"call.i.i'mi" = tail call noalias nonnull i8* @malloc(i64 16) #10
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull align 1 %"call.i.i'mi", i8 0, i64 16, i1 false)
; CHECK-NEXT:   %0 = bitcast i8* %call.i.i to double*
; CHECK-NEXT:   %"'ipc" = bitcast i8* %"call.i.i'mi" to double*
; CHECK-NEXT:   %.fca.0.0.insert = insertvalue { { i8* }, double*, double* } undef, i8* %"call.i.i'mi", 0, 0
; CHECK-NEXT:   %.fca.1.insert = insertvalue { { i8* }, double*, double* } %.fca.0.0.insert, double* %0, 1
; CHECK-NEXT:   %.fca.2.insert = insertvalue { { i8* }, double*, double* } %.fca.1.insert, double* %"'ipc", 2
; CHECK-NEXT:   ret { { i8* }, double*, double* } %.fca.2.insert
; CHECK-NEXT: }
