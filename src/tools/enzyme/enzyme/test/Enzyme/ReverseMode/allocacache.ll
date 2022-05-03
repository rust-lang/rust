; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -simplifycfg -instcombine -adce -S | FileCheck %s

source_filename = "/home/enzyme/Enzyme/enzyme/test/Integration/simpleeigenstatic-made.cpp"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }

@stderr = external dso_local local_unnamed_addr global %struct._IO_FILE*, align 8
@.str = private unnamed_addr constant [18 x i8] c"W(o=%d, i=%d)=%f\0A\00", align 1
@.str.1 = private unnamed_addr constant [12 x i8] c"M(o=%d)=%f\0A\00", align 1
@.str.2 = private unnamed_addr constant [12 x i8] c"O(i=%d)=%f\0A\00", align 1
@.str.3 = private unnamed_addr constant [68 x i8] c"Assertion Failed: fabs( [%s = %g] - [%s = %g] ) > %g at %s:%d (%s)\0A\00", align 1
@.str.4 = private unnamed_addr constant [9 x i8] c"Wp(i, o)\00", align 1
@.str.5 = private unnamed_addr constant [18 x i8] c"M(o) * Op_orig(i)\00", align 1
@.str.6 = private unnamed_addr constant [71 x i8] c"/home/enzyme/Enzyme/enzyme/test/Integration/simpleeigenstatic-made.cpp\00", align 1
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
@__PRETTY_FUNCTION__._ZN5Eigen7ProductINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEENS1_IdLi2ELi1ELi0ELi2ELi1EEELi0EEC2ERKS2_RKS3_ = private unnamed_addr constant [262 x i8] c"Eigen::Product<Eigen::Matrix<double, 2, 2, 0, 2, 2>, Eigen::Matrix<double, 2, 1, 0, 2, 1>, 0>::Product(const Eigen::Product::Lhs &, const Eigen::Product::Rhs &) [Lhs = Eigen::Matrix<double, 2, 2, 0, 2, 2>, Rhs = Eigen::Matrix<double, 2, 1, 0, 2, 1>, Option = 0]\00", align 1
@.str.16 = private unnamed_addr constant [192 x i8] c"(internal::UIntPtr(array) & (15)) == 0 && \22this assertion is explained here: \22 \22http://eigen.tuxfamily.org/dox-devel/group__TopicUnalignedArrayAssert.html\22 \22 **** READ THIS WEB PAGE !!! ****\22\00", align 1
@.str.17 = private unnamed_addr constant [56 x i8] c"/usr/local/include/eigen3/Eigen/src/Core/DenseStorage.h\00", align 1
@__PRETTY_FUNCTION__._ZN5Eigen8internal11plain_arrayIdLi2ELi0ELi16EEC2Ev = private unnamed_addr constant [127 x i8] c"Eigen::internal::plain_array<double, 2, 0, 16>::plain_array() [T = double, Size = 2, MatrixOrArrayOptions = 0, Alignment = 16]\00", align 1
@.str.18 = private unnamed_addr constant [399 x i8] c"(!(RowsAtCompileTime!=Dynamic) || (rows==RowsAtCompileTime)) && (!(ColsAtCompileTime!=Dynamic) || (cols==ColsAtCompileTime)) && (!(RowsAtCompileTime==Dynamic && MaxRowsAtCompileTime!=Dynamic) || (rows<=MaxRowsAtCompileTime)) && (!(ColsAtCompileTime==Dynamic && MaxColsAtCompileTime!=Dynamic) || (cols<=MaxColsAtCompileTime)) && rows>=0 && cols>=0 && \22Invalid sizes when resizing a matrix or array.\22\00", align 1
@.str.19 = private unnamed_addr constant [59 x i8] c"/usr/local/include/eigen3/Eigen/src/Core/PlainObjectBase.h\00", align 1
@__PRETTY_FUNCTION__._ZN5Eigen15PlainObjectBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEEE6resizeEll = private unnamed_addr constant [152 x i8] c"void Eigen::PlainObjectBase<Eigen::Matrix<double, 2, 1, 0, 2, 1> >::resize(Eigen::Index, Eigen::Index) [Derived = Eigen::Matrix<double, 2, 1, 0, 2, 1>]\00", align 1
@.str.20 = private unnamed_addr constant [14 x i8] c"v == T(Value)\00", align 1
@.str.21 = private unnamed_addr constant [58 x i8] c"/usr/local/include/eigen3/Eigen/src/Core/util/XprHelper.h\00", align 1
@__PRETTY_FUNCTION__._ZN5Eigen8internal19variable_if_dynamicIlLi2EEC2El = private unnamed_addr constant [92 x i8] c"Eigen::internal::variable_if_dynamic<long, 2>::variable_if_dynamic(T) [T = long, Value = 2]\00", align 1
@.str.22 = private unnamed_addr constant [47 x i8] c"dst.rows() == dstRows && dst.cols() == dstCols\00", align 1
@.str.23 = private unnamed_addr constant [59 x i8] c"/usr/local/include/eigen3/Eigen/src/Core/AssignEvaluator.h\00", align 1
@__PRETTY_FUNCTION__._ZN5Eigen8internal17resize_if_allowedINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEENS_7ProductINS2_IdLi2ELi2ELi0ELi2ELi2EEES3_Li1EEEddEEvRT_RKT0_RKNS0_9assign_opIT1_T2_EE = private unnamed_addr constant [297 x i8] c"void Eigen::internal::resize_if_allowed(DstXprType &, const SrcXprType &, const internal::assign_op<T1, T2> &) [DstXprType = Eigen::Matrix<double, 2, 1, 0, 2, 1>, SrcXprType = Eigen::Product<Eigen::Matrix<double, 2, 2, 0, 2, 2>, Eigen::Matrix<double, 2, 1, 0, 2, 1>, 1>, T1 = double, T2 = double]\00", align 1
@__PRETTY_FUNCTION__._ZN5Eigen8internal19variable_if_dynamicIlLi0EEC2El = private unnamed_addr constant [92 x i8] c"Eigen::internal::variable_if_dynamic<long, 0>::variable_if_dynamic(T) [T = long, Value = 0]\00", align 1
@__PRETTY_FUNCTION__._ZN5Eigen7ProductINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEENS1_IdLi2ELi1ELi0ELi2ELi1EEELi1EEC2ERKS2_RKS3_ = private unnamed_addr constant [262 x i8] c"Eigen::Product<Eigen::Matrix<double, 2, 2, 0, 2, 2>, Eigen::Matrix<double, 2, 1, 0, 2, 1>, 1>::Product(const Eigen::Product::Lhs &, const Eigen::Product::Rhs &) [Lhs = Eigen::Matrix<double, 2, 2, 0, 2, 2>, Rhs = Eigen::Matrix<double, 2, 1, 0, 2, 1>, Option = 1]\00", align 1
@__PRETTY_FUNCTION__._ZN5Eigen8internal17resize_if_allowedINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEES3_ddEEvRT_RKT0_RKNS0_9assign_opIT1_T2_EE = private unnamed_addr constant [240 x i8] c"void Eigen::internal::resize_if_allowed(DstXprType &, const SrcXprType &, const internal::assign_op<T1, T2> &) [DstXprType = Eigen::Matrix<double, 2, 1, 0, 2, 1>, SrcXprType = Eigen::Matrix<double, 2, 1, 0, 2, 1>, T1 = double, T2 = double]\00", align 1
@.str.24 = private unnamed_addr constant [149 x i8] c"rows >= 0 && (RowsAtCompileTime == Dynamic || RowsAtCompileTime == rows) && cols >= 0 && (ColsAtCompileTime == Dynamic || ColsAtCompileTime == cols)\00", align 1
@.str.25 = private unnamed_addr constant [58 x i8] c"/usr/local/include/eigen3/Eigen/src/Core/CwiseNullaryOp.h\00", align 1
@__PRETTY_FUNCTION__._ZN5Eigen14CwiseNullaryOpINS_8internal18scalar_constant_opIdEENS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEC2EllRKS3_ = private unnamed_addr constant [278 x i8] c"Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<double>, Eigen::Matrix<double, 2, 2, 0, 2, 2> >::CwiseNullaryOp(Eigen::Index, Eigen::Index, const NullaryOp &) [NullaryOp = Eigen::internal::scalar_constant_op<double>, MatrixType = Eigen::Matrix<double, 2, 2, 0, 2, 2>]\00", align 1
@__PRETTY_FUNCTION__._ZN5Eigen8internal11plain_arrayIdLi4ELi0ELi16EEC2Ev = private unnamed_addr constant [127 x i8] c"Eigen::internal::plain_array<double, 4, 0, 16>::plain_array() [T = double, Size = 4, MatrixOrArrayOptions = 0, Alignment = 16]\00", align 1
@__PRETTY_FUNCTION__._ZN5Eigen15PlainObjectBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE6resizeEll = private unnamed_addr constant [152 x i8] c"void Eigen::PlainObjectBase<Eigen::Matrix<double, 2, 2, 0, 2, 2> >::resize(Eigen::Index, Eigen::Index) [Derived = Eigen::Matrix<double, 2, 2, 0, 2, 2>]\00", align 1
@__PRETTY_FUNCTION__._ZN5Eigen8internal17resize_if_allowedINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EEddEEvRT_RKT0_RKNS0_9assign_opIT1_T2_EE = private unnamed_addr constant [309 x i8] c"void Eigen::internal::resize_if_allowed(DstXprType &, const SrcXprType &, const internal::assign_op<T1, T2> &) [DstXprType = Eigen::Matrix<double, 2, 2, 0, 2, 2>, SrcXprType = Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<double>, Eigen::Matrix<double, 2, 2, 0, 2, 2> >, T1 = double, T2 = double]\00", align 1
@__PRETTY_FUNCTION__._ZN5Eigen14CwiseNullaryOpINS_8internal18scalar_constant_opIdEENS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEEEC2EllRKS3_ = private unnamed_addr constant [278 x i8] c"Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<double>, Eigen::Matrix<double, 2, 1, 0, 2, 1> >::CwiseNullaryOp(Eigen::Index, Eigen::Index, const NullaryOp &) [NullaryOp = Eigen::internal::scalar_constant_op<double>, MatrixType = Eigen::Matrix<double, 2, 1, 0, 2, 1>]\00", align 1
@__PRETTY_FUNCTION__._ZN5Eigen8internal19variable_if_dynamicIlLi1EEC2El = private unnamed_addr constant [92 x i8] c"Eigen::internal::variable_if_dynamic<long, 1>::variable_if_dynamic(T) [T = long, Value = 1]\00", align 1
@.str.26 = private unnamed_addr constant [39 x i8] c"other.rows() == 1 || other.cols() == 1\00", align 1
@__PRETTY_FUNCTION__._ZN5Eigen15PlainObjectBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEEE10resizeLikeINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEEvRKNS_9EigenBaseIT_EE = private unnamed_addr constant [283 x i8] c"void Eigen::PlainObjectBase<Eigen::Matrix<double, 2, 1, 0, 2, 1> >::resizeLike(const EigenBase<OtherDerived> &) [Derived = Eigen::Matrix<double, 2, 1, 0, 2, 1>, OtherDerived = Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<double>, Eigen::Matrix<double, 2, 1, 0, 2, 1> >]\00", align 1
@__PRETTY_FUNCTION__._ZN5Eigen8internal17resize_if_allowedINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EEddEEvRT_RKT0_RKNS0_9assign_opIT1_T2_EE = private unnamed_addr constant [309 x i8] c"void Eigen::internal::resize_if_allowed(DstXprType &, const SrcXprType &, const internal::assign_op<T1, T2> &) [DstXprType = Eigen::Matrix<double, 2, 1, 0, 2, 1>, SrcXprType = Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<double>, Eigen::Matrix<double, 2, 1, 0, 2, 1> >, T1 = double, T2 = double]\00", align 1
@.str.27 = private unnamed_addr constant [53 x i8] c"row >= 0 && row < rows() && col >= 0 && col < cols()\00", align 1
@.str.28 = private unnamed_addr constant [59 x i8] c"/usr/local/include/eigen3/Eigen/src/Core/DenseCoeffsBase.h\00", align 1
@__PRETTY_FUNCTION__._ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEELi1EEclEll = private unnamed_addr constant [219 x i8] c"Eigen::DenseCoeffsBase<type-parameter-0-0, 1>::Scalar &Eigen::DenseCoeffsBase<Eigen::Matrix<double, 2, 2, 0, 2, 2>, 1>::operator()(Eigen::Index, Eigen::Index) [Derived = Eigen::Matrix<double, 2, 2, 0, 2, 2>, Level = 1]\00", align 1
@.str.29 = private unnamed_addr constant [29 x i8] c"index >= 0 && index < size()\00", align 1
@__PRETTY_FUNCTION__._ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl = private unnamed_addr constant [205 x i8] c"Eigen::DenseCoeffsBase<type-parameter-0-0, 1>::Scalar &Eigen::DenseCoeffsBase<Eigen::Matrix<double, 2, 1, 0, 2, 1>, 1>::operator()(Eigen::Index) [Derived = Eigen::Matrix<double, 2, 1, 0, 2, 1>, Level = 1]\00", align 1

; Function Attrs: alwaysinline norecurse nounwind uwtable
define dso_local i32 @main(i32 %argc, i8** nocapture readnone %argv) local_unnamed_addr #0 {
entry:
  %W = alloca [4 x double], align 16
  %M = alloca <2 x double>, align 16
  %O = alloca <2 x double>, align 16
  %Wp = alloca [4 x double], align 16
  %Mp = alloca <2 x double>, align 16
  %Op = alloca <2 x double>, align 16
  %Op_orig = alloca [2 x double], align 16
  %0 = bitcast [4 x double]* %W to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %0) #9
  %1 = bitcast [4 x double]* %W to <2 x double>*
  store <2 x double> <double 3.000000e+00, double 3.000000e+00>, <2 x double>* %1, align 16, !tbaa !2
  %arrayidx.i16 = getelementptr inbounds [4 x double], [4 x double]* %W, i64 0, i64 2
  %2 = bitcast double* %arrayidx.i16 to <2 x double>*
  store <2 x double> <double 3.000000e+00, double 3.000000e+00>, <2 x double>* %2, align 16, !tbaa !2
  %3 = bitcast <2 x double>* %M to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %3) #9
  store <2 x double> <double 2.000000e+00, double 2.000000e+00>, <2 x double>* %M, align 16, !tbaa !2
  %4 = bitcast <2 x double>* %O to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %4) #9
  store <2 x double> zeroinitializer, <2 x double>* %O, align 16, !tbaa !2
  %5 = bitcast [4 x double]* %Wp to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %5) #9
  %6 = bitcast [4 x double]* %Wp to <2 x double>*
  store <2 x double> zeroinitializer, <2 x double>* %6, align 16, !tbaa !2
  %arrayidx.i67 = getelementptr inbounds [4 x double], [4 x double]* %Wp, i64 0, i64 2
  %7 = bitcast double* %arrayidx.i67 to <2 x double>*
  store <2 x double> zeroinitializer, <2 x double>* %7, align 16, !tbaa !2
  %8 = bitcast <2 x double>* %Mp to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %8) #9
  store <2 x double> zeroinitializer, <2 x double>* %Mp, align 16, !tbaa !2
  %9 = bitcast <2 x double>* %Op to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %9) #9
  store <2 x double> <double 1.000000e+00, double 1.000000e+00>, <2 x double>* %Op, align 16, !tbaa !2
  %10 = bitcast [2 x double]* %Op_orig to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %10) #9
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %10, i8* nonnull align 16 %9, i64 16, i1 false) #9, !tbaa.struct !5
  %call = call double @__enzyme_autodiff(i8* bitcast (void (<2 x double>*, double*, <2 x double>*)* @matvec to i8*), i8* nonnull %0, i8* nonnull %5, i8* nonnull %3, i8* nonnull %8, i8* nonnull %4, i8* nonnull %9) #9
  br label %for.cond12.preheader

for.cond12.preheader:                             ; preds = %for.cond.cleanup15, %entry
  %indvars.iv250 = phi i64 [ 0, %entry ], [ %indvars.iv.next251, %for.cond.cleanup15 ]
  %11 = trunc i64 %indvars.iv250 to i32
  br label %for.body16

for.cond.cleanup15:                               ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEELi1EEclEll.exit
  %indvars.iv.next251 = add nuw nsw i64 %indvars.iv250, 1
  %exitcond252 = icmp eq i64 %indvars.iv.next251, 2
  br i1 %exitcond252, label %for.body29, label %for.cond12.preheader

for.body16:                                       ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEELi1EEclEll.exit, %for.cond12.preheader
  %indvars.iv247 = phi i64 [ 0, %for.cond12.preheader ], [ %indvars.iv.next248, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEELi1EEclEll.exit ]
  %12 = or i64 %indvars.iv247, %indvars.iv250
  %13 = and i64 %12, 9223372036854775806
  %14 = icmp eq i64 %13, 0
  br i1 %14, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEELi1EEclEll.exit, label %cond.false.i

cond.false.i:                                     ; preds = %for.body16
  call void @__assert_fail(i8* getelementptr inbounds ([53 x i8], [53 x i8]* @.str.27, i64 0, i64 0), i8* getelementptr inbounds ([59 x i8], [59 x i8]* @.str.28, i64 0, i64 0), i32 365, i8* getelementptr inbounds ([219 x i8], [219 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEELi1EEclEll, i64 0, i64 0)) #10
  unreachable

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEELi1EEclEll.exit: ; preds = %for.body16
  %15 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !6
  %mul.i96 = shl nuw nsw i64 %indvars.iv250, 1
  %add.i97 = add nuw nsw i64 %mul.i96, %indvars.iv247
  %arrayidx.i98 = getelementptr inbounds [4 x double], [4 x double]* %W, i64 0, i64 %add.i97
  %16 = load double, double* %arrayidx.i98, align 8, !tbaa !8
  %17 = trunc i64 %indvars.iv247 to i32
  %call20 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %15, i8* getelementptr inbounds ([18 x i8], [18 x i8]* @.str, i64 0, i64 0), i32 %17, i32 %11, double %16) #11
  %indvars.iv.next248 = add nuw nsw i64 %indvars.iv247, 1
  %exitcond249 = icmp eq i64 %indvars.iv.next248, 2
  br i1 %exitcond249, label %for.cond.cleanup15, label %for.body16

for.body29:                                       ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl.exit, %for.cond.cleanup15
  %indvars.iv244 = phi i64 [ %indvars.iv.next245, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl.exit ], [ 0, %for.cond.cleanup15 ]
  %cmp2.i290 = icmp ult i64 %indvars.iv244, 2
  br i1 %cmp2.i290, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl.exit, label %cond.false.i292

cond.false.i292:                                  ; preds = %for.body29
  call void @__assert_fail(i8* getelementptr inbounds ([29 x i8], [29 x i8]* @.str.29, i64 0, i64 0), i8* getelementptr inbounds ([59 x i8], [59 x i8]* @.str.28, i64 0, i64 0), i32 425, i8* getelementptr inbounds ([205 x i8], [205 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl, i64 0, i64 0)) #10
  unreachable

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl.exit: ; preds = %for.body29
  %18 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !6
  %arrayidx.i102 = getelementptr inbounds <2 x double>, <2 x double>* %M, i64 0, i64 %indvars.iv244
  %19 = load double, double* %arrayidx.i102, align 8, !tbaa !8
  %20 = trunc i64 %indvars.iv244 to i32
  %call32 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %18, i8* getelementptr inbounds ([12 x i8], [12 x i8]* @.str.1, i64 0, i64 0), i32 %20, double %19) #11
  %indvars.iv.next245 = add nuw nsw i64 %indvars.iv244, 1
  %exitcond246 = icmp eq i64 %indvars.iv.next245, 2
  br i1 %exitcond246, label %for.body41, label %for.body29

for.body41:                                       ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl.exit301, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl.exit
  %indvars.iv241 = phi i64 [ %indvars.iv.next242, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl.exit301 ], [ 0, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl.exit ]
  %cmp2.i297 = icmp ult i64 %indvars.iv241, 2
  br i1 %cmp2.i297, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl.exit301, label %cond.false.i299

cond.false.i299:                                  ; preds = %for.body41
  call void @__assert_fail(i8* getelementptr inbounds ([29 x i8], [29 x i8]* @.str.29, i64 0, i64 0), i8* getelementptr inbounds ([59 x i8], [59 x i8]* @.str.28, i64 0, i64 0), i32 425, i8* getelementptr inbounds ([205 x i8], [205 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl, i64 0, i64 0)) #10
  unreachable

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl.exit301: ; preds = %for.body41
  %21 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !6
  %arrayidx.i107 = getelementptr inbounds <2 x double>, <2 x double>* %O, i64 0, i64 %indvars.iv241
  %22 = load double, double* %arrayidx.i107, align 8, !tbaa !8
  %23 = trunc i64 %indvars.iv241 to i32
  %call44 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %21, i8* getelementptr inbounds ([12 x i8], [12 x i8]* @.str.2, i64 0, i64 0), i32 %23, double %22) #11
  %indvars.iv.next242 = add nuw nsw i64 %indvars.iv241, 1
  %exitcond243 = icmp eq i64 %indvars.iv.next242, 2
  br i1 %exitcond243, label %for.cond55.preheader, label %for.body41

for.cond55.preheader:                             ; preds = %for.cond.cleanup58, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl.exit301
  %indvars.iv239 = phi i64 [ %indvars.iv.next240, %for.cond.cleanup58 ], [ 0, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl.exit301 ]
  %24 = trunc i64 %indvars.iv239 to i32
  br label %for.body59

for.cond.cleanup58:                               ; preds = %if.end
  %indvars.iv.next240 = add nuw nsw i64 %indvars.iv239, 1
  %cmp51 = icmp ult i64 %indvars.iv.next240, 2
  br i1 %cmp51, label %for.cond55.preheader, label %for.cond94.preheader

for.body59:                                       ; preds = %if.end, %for.cond55.preheader
  %indvars.iv237 = phi i64 [ 0, %for.cond55.preheader ], [ %indvars.iv.next238, %if.end ]
  %25 = or i64 %indvars.iv237, %indvars.iv239
  %26 = and i64 %25, 9223372036854775806
  %27 = icmp eq i64 %26, 0
  br i1 %27, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEELi1EEclEll.exit314, label %cond.false.i312

cond.false.i312:                                  ; preds = %for.body59
  call void @__assert_fail(i8* getelementptr inbounds ([53 x i8], [53 x i8]* @.str.27, i64 0, i64 0), i8* getelementptr inbounds ([59 x i8], [59 x i8]* @.str.28, i64 0, i64 0), i32 365, i8* getelementptr inbounds ([219 x i8], [219 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEELi1EEclEll, i64 0, i64 0)) #10
  unreachable

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEELi1EEclEll.exit314: ; preds = %for.body59
  %28 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !6
  %arrayidx.i118 = getelementptr inbounds <2 x double>, <2 x double>* %M, i64 0, i64 %indvars.iv239
  %29 = load double, double* %arrayidx.i118, align 8, !tbaa !8
  %mul.i111 = shl nuw nsw i64 %indvars.iv239, 1
  %add.i112 = add nuw nsw i64 %mul.i111, %indvars.iv237
  %arrayidx.i113 = getelementptr inbounds [4 x double], [4 x double]* %Wp, i64 0, i64 %add.i112
  %30 = load double, double* %arrayidx.i113, align 8, !tbaa !8
  %arrayidx.i123 = getelementptr inbounds [2 x double], [2 x double]* %Op_orig, i64 0, i64 %indvars.iv237
  %31 = load double, double* %arrayidx.i123, align 8, !tbaa !8
  %mul = fmul double %29, %31
  %sub = fsub double %30, %mul
  %32 = call double @llvm.fabs.f64(double %sub)
  %cmp67 = fcmp ogt double %32, 1.000000e-10
  br i1 %cmp67, label %land.lhs.true.i348, label %if.end

land.lhs.true.i348:                               ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEELi1EEclEll.exit314
  %cmp2.i347 = icmp ult i64 %indvars.iv239, 2
  br i1 %cmp2.i347, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl.exit351, label %cond.false.i349

cond.false.i349:                                  ; preds = %land.lhs.true.i348
  call void @__assert_fail(i8* getelementptr inbounds ([29 x i8], [29 x i8]* @.str.29, i64 0, i64 0), i8* getelementptr inbounds ([59 x i8], [59 x i8]* @.str.28, i64 0, i64 0), i32 425, i8* getelementptr inbounds ([205 x i8], [205 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl, i64 0, i64 0)) #10
  unreachable

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl.exit351: ; preds = %land.lhs.true.i348
  %cmp2.i355 = icmp ult i64 %indvars.iv237, 2
  br i1 %cmp2.i355, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl.exit359, label %cond.false.i357

cond.false.i357:                                  ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl.exit351
  call void @__assert_fail(i8* getelementptr inbounds ([29 x i8], [29 x i8]* @.str.29, i64 0, i64 0), i8* getelementptr inbounds ([59 x i8], [59 x i8]* @.str.28, i64 0, i64 0), i32 425, i8* getelementptr inbounds ([205 x i8], [205 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl, i64 0, i64 0)) #10
  unreachable

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl.exit359: ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl.exit351
  %call76 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %28, i8* getelementptr inbounds ([68 x i8], [68 x i8]* @.str.3, i64 0, i64 0), i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str.4, i64 0, i64 0), double %30, i8* getelementptr inbounds ([18 x i8], [18 x i8]* @.str.5, i64 0, i64 0), double %mul, double 1.000000e-10, i8* getelementptr inbounds ([71 x i8], [71 x i8]* @.str.6, i64 0, i64 0), i32 64, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @__PRETTY_FUNCTION__.main, i64 0, i64 0)) #11
  call void @abort() #10
  unreachable

if.end:                                           ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEELi1EEclEll.exit314
  %33 = trunc i64 %indvars.iv237 to i32
  %call80 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %28, i8* getelementptr inbounds ([19 x i8], [19 x i8]* @.str.7, i64 0, i64 0), i32 %33, i32 %24, double %30) #11
  %indvars.iv.next238 = add nuw nsw i64 %indvars.iv237, 1
  %cmp57 = icmp ult i64 %indvars.iv.next238, 2
  br i1 %cmp57, label %for.body59, label %for.cond.cleanup58

for.cond94.preheader:                             ; preds = %if.end116, %for.cond.cleanup58
  %indvars.iv235 = phi i64 [ %indvars.iv.next236, %if.end116 ], [ 0, %for.cond.cleanup58 ]
  br label %for.body98

land.lhs.true.i364:                               ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEELi1EEclEll.exit388
  %cmp2.i363 = icmp ult i64 %indvars.iv235, 2
  br i1 %cmp2.i363, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl.exit367, label %cond.false.i365

cond.false.i365:                                  ; preds = %land.lhs.true.i364
  call void @__assert_fail(i8* getelementptr inbounds ([29 x i8], [29 x i8]* @.str.29, i64 0, i64 0), i8* getelementptr inbounds ([59 x i8], [59 x i8]* @.str.28, i64 0, i64 0), i32 425, i8* getelementptr inbounds ([205 x i8], [205 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl, i64 0, i64 0)) #10
  unreachable

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl.exit367: ; preds = %land.lhs.true.i364
  %arrayidx.i144 = getelementptr inbounds <2 x double>, <2 x double>* %Mp, i64 0, i64 %indvars.iv235
  %34 = load double, double* %arrayidx.i144, align 8, !tbaa !8
  %sub110 = fsub double %34, %add
  %35 = call double @llvm.fabs.f64(double %sub110)
  %cmp111 = fcmp ogt double %35, 1.000000e-10
  %36 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !6
  br i1 %cmp111, label %if.then112, label %if.end116

for.body98:                                       ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEELi1EEclEll.exit388, %for.cond94.preheader
  %indvars.iv233 = phi i64 [ 0, %for.cond94.preheader ], [ %indvars.iv.next234, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEELi1EEclEll.exit388 ]
  %res.0208 = phi double [ 0.000000e+00, %for.cond94.preheader ], [ %add, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEELi1EEclEll.exit388 ]
  %37 = or i64 %indvars.iv233, %indvars.iv235
  %38 = and i64 %37, 9223372036854775806
  %39 = icmp eq i64 %38, 0
  br i1 %39, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEELi1EEclEll.exit388, label %cond.false.i386

cond.false.i386:                                  ; preds = %for.body98
  call void @__assert_fail(i8* getelementptr inbounds ([53 x i8], [53 x i8]* @.str.27, i64 0, i64 0), i8* getelementptr inbounds ([59 x i8], [59 x i8]* @.str.28, i64 0, i64 0), i32 365, i8* getelementptr inbounds ([219 x i8], [219 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEELi1EEclEll, i64 0, i64 0)) #10
  unreachable

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEELi1EEclEll.exit388: ; preds = %for.body98
  %mul.i153 = shl nuw nsw i64 %indvars.iv235, 1
  %add.i154 = add nuw nsw i64 %mul.i153, %indvars.iv233
  %arrayidx.i155 = getelementptr inbounds [4 x double], [4 x double]* %W, i64 0, i64 %add.i154
  %40 = load double, double* %arrayidx.i155, align 8, !tbaa !8
  %arrayidx.i160 = getelementptr inbounds [2 x double], [2 x double]* %Op_orig, i64 0, i64 %indvars.iv233
  %41 = load double, double* %arrayidx.i160, align 8, !tbaa !8
  %mul104 = fmul double %40, %41
  %add = fadd double %res.0208, %mul104
  %indvars.iv.next234 = add nuw nsw i64 %indvars.iv233, 1
  %exitcond = icmp eq i64 %indvars.iv.next234, 2
  br i1 %exitcond, label %land.lhs.true.i364, label %for.body98

if.then112:                                       ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl.exit367
  %call115 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %36, i8* getelementptr inbounds ([68 x i8], [68 x i8]* @.str.3, i64 0, i64 0), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.8, i64 0, i64 0), double %34, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.9, i64 0, i64 0), double %add, double 1.000000e-10, i8* getelementptr inbounds ([71 x i8], [71 x i8]* @.str.6, i64 0, i64 0), i32 71, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @__PRETTY_FUNCTION__.main, i64 0, i64 0)) #11
  call void @abort() #10
  unreachable

if.end116:                                        ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl.exit367
  %42 = trunc i64 %indvars.iv235 to i32
  %call119 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %36, i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.10, i64 0, i64 0), i32 %42, double %34) #11
  %indvars.iv.next236 = add nuw nsw i64 %indvars.iv235, 1
  %cmp90 = icmp ult i64 %indvars.iv.next236, 2
  br i1 %cmp90, label %for.cond94.preheader, label %for.body128

for.cond.cleanup127:                              ; preds = %if.end137
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %10) #9
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %9) #9
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %8) #9
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %5) #9
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %4) #9
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %3) #9
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %0) #9
  ret i32 0

for.body128:                                      ; preds = %if.end137, %if.end116
  %indvars.iv = phi i64 [ %indvars.iv.next, %if.end137 ], [ 0, %if.end116 ]
  %cmp2.i400 = icmp ult i64 %indvars.iv, 2
  br i1 %cmp2.i400, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl.exit404, label %cond.false.i402

cond.false.i402:                                  ; preds = %for.body128
  call void @__assert_fail(i8* getelementptr inbounds ([29 x i8], [29 x i8]* @.str.29, i64 0, i64 0), i8* getelementptr inbounds ([59 x i8], [59 x i8]* @.str.28, i64 0, i64 0), i32 425, i8* getelementptr inbounds ([205 x i8], [205 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl, i64 0, i64 0)) #10
  unreachable

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl.exit404: ; preds = %for.body128
  %arrayidx.i51 = getelementptr inbounds <2 x double>, <2 x double>* %Op, i64 0, i64 %indvars.iv
  %43 = load double, double* %arrayidx.i51, align 8, !tbaa !8
  %44 = call double @llvm.fabs.f64(double %43)
  %cmp132 = fcmp ogt double %44, 1.000000e-10
  %45 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !6
  br i1 %cmp132, label %if.then133, label %if.end137

if.then133:                                       ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl.exit404
  %call136 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %45, i8* getelementptr inbounds ([68 x i8], [68 x i8]* @.str.3, i64 0, i64 0), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.11, i64 0, i64 0), double %43, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.12, i64 0, i64 0), double 0.000000e+00, double 1.000000e-10, i8* getelementptr inbounds ([71 x i8], [71 x i8]* @.str.6, i64 0, i64 0), i32 76, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @__PRETTY_FUNCTION__.main, i64 0, i64 0)) #11
  call void @abort() #10
  unreachable

if.end137:                                        ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl.exit404
  %46 = trunc i64 %indvars.iv to i32
  %call140 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %45, i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.13, i64 0, i64 0), i32 %46, double %43) #11
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %cmp126 = icmp ult i64 %indvars.iv.next, 2
  br i1 %cmp126, label %for.body128, label %for.cond.cleanup127
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

; Function Attrs: alwaysinline
declare dso_local double @__enzyme_autodiff(i8*, i8*, i8*, i8*, i8*, i8*, i8*) local_unnamed_addr #2

; Function Attrs: noinline nounwind uwtable
define void @matvec(<2 x double>* %Wptr, double* %B, <2 x double>* %outvec) #3 {
entry:
  %B1 = load double, double* %B
  %B2p = getelementptr inbounds double, double* %B, i64 1
  %B2 = load double, double* %B2p

  %call = call <2 x double> @subfn(<2 x double>* %Wptr, double %B1, double %B2, i64 0) #9
  call void @copy(<2 x double>* %outvec, <2 x double> %call) #9
  ret void
}

; Function Attrs: inlinehint norecurse nounwind uwtable
define void @copy(<2 x double>* %to, <2 x double> %from) #8 {
entry:
  store <2 x double> %from, <2 x double>* %to
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

; Function Attrs: nounwind uwtable
define <2 x double> @subfn(<2 x double>* %W, double %B1, double %B2, i64 %row) #7 {
entry:
  %Bref = alloca <2 x double>, align 16

  %W34p = getelementptr inbounds <2 x double>, <2 x double>* %W, i64 1
  %W34 = load <2 x double>, <2 x double>* %W34p

  %preb1 = insertelement <2 x double> undef, double %B1, i32 0
  %B11 = insertelement <2 x double> %preb1, double %B1, i32 1

  %preb2 = insertelement <2 x double> undef, double %B2, i32 0
  %B22 = insertelement <2 x double> %preb2, double %B2, i32 1

  store <2 x double> %B11, <2 x double>* %Bref

  ; %Bref2 = bitcast double* %B to <2 x double>*

  %call = call <2 x double> @loadmul(<2 x double>* %W, <2 x double>* %Bref) #9

  %mul = fmul <2 x double> %W34, %B22
  %add = fadd <2 x double> %mul, %call
  ret <2 x double> %add
}

; Function Attrs: inlinehint norecurse nounwind uwtable
define <2 x double> @loadmul(<2 x double>* %a, <2 x double>* %b) #8 {
entry:
  %0 = load <2 x double>, <2 x double>* %a
  %1 = load <2 x double>, <2 x double>* %b
  %mul.i = fmul <2 x double> %0, %1
  ret <2 x double> %mul.i
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1) #1

attributes #0 = { norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="128" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="128" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { nounwind readnone speculatable }
attributes #6 = { noreturn nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #7 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="128" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #8 = { inlinehint norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="128" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #9 = { nounwind }
attributes #10 = { noreturn nounwind }
attributes #11 = { cold }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C++ TBAA"}
!5 = !{i64 0, i64 16, !2}
!6 = !{!7, !7, i64 0}
!7 = !{!"any pointer", !3, i64 0}
!8 = !{!9, !9, i64 0}
!9 = !{!"double", !3, i64 0}

; CHECK: define internal void @diffematvec(<2 x double>* %Wptr, <2 x double>* %"Wptr'", double* %B, double* %"B'", <2 x double>* %outvec, <2 x double>* %"outvec'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %B1 = load double, double* %B, align 8
; CHECK-NEXT:   %[[B2pprime:.+]] = getelementptr inbounds double, double* %"B'", i64 1
; CHECK-NEXT:   %B2p = getelementptr inbounds double, double* %B, i64 1
; CHECK-NEXT:   %B2 = load double, double* %B2p, align 8
; CHECK-NEXT:   %call_augmented = call {{.*}} @augmented_subfn(<2 x double>* %Wptr, <2 x double>* %"Wptr'", double %B1, double %B2, i64 0)
; CHECK-NEXT:   %[[calltape:.+]] = extractvalue {{.*}} %call_augmented, 0
; CHECK-NEXT:   %call = extractvalue { {{.*}}, <2 x double> } %call_augmented, 1
; CHECK-NEXT:   %[[copyret:.+]] = call { <2 x double> } @diffecopy(<2 x double>* %outvec, <2 x double>* %"outvec'", <2 x double> %call)
; CHECK-NEXT:   %[[copyext:.+]] = extractvalue { <2 x double> } %[[copyret]], 0
; CHECK-NEXT:   %[[subfnret:.+]] = call { double, double } @diffesubfn(<2 x double>* %Wptr, <2 x double>* %"Wptr'", double %B1, double %B2, i64 0, <2 x double> %[[copyext]], {{.*}} %[[calltape]])
; CHECK-NEXT:   %[[sub0:.+]] = extractvalue { double, double } %[[subfnret]], 0
; CHECK-NEXT:   %[[sub1:.+]] = extractvalue { double, double } %[[subfnret]], 1
; CHECK-NEXT:   %[[preb2:.+]] = load double, double* %[[B2pprime]], align 8
; CHECK-NEXT:   %[[addb2:.+]] = fadd fast double %[[preb2]], %[[sub1]]
; CHECK-NEXT:   store double %[[addb2]], double* %[[B2pprime]], align 8
; CHECK-NEXT:   %[[preb:.+]] = load double, double* %"B'", align 8
; CHECK-NEXT:   %[[addb:.+]] = fadd fast double %[[preb]], %[[sub0]]
; CHECK-NEXT:   store double %[[addb]], double* %"B'", align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal { <2 x double> } @diffecopy(<2 x double>* %to, <2 x double>* %"to'", <2 x double> %from)
; CHECK-NEXT: entry:
; CHECK-NEXT:   store <2 x double> %from, <2 x double>* %to, align 16
; CHECK-NEXT:   %0 = load <2 x double>, <2 x double>* %"to'", align 16
; CHECK-NEXT:   store <2 x double> zeroinitializer, <2 x double>* %"to'", align 16
; CHECK-NEXT:   %1 = insertvalue { <2 x double> } undef, <2 x double> %0, 0
; CHECK-NEXT:   ret { <2 x double> } %1
; CHECK-NEXT: }

; CHECK: define internal { <2 x double>, <2 x double> } @augmented_loadmul(<2 x double>* %a, <2 x double>* %"a'", <2 x double>* %b, <2 x double>* %"b'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = load <2 x double>, <2 x double>* %a, align 16
; CHECK-NEXT:   %1 = load <2 x double>, <2 x double>* %b, align 16
; CHECK-NEXT:   %mul.i = fmul <2 x double> %0, %1
; CHECK-NEXT:   %.fca.0.insert = insertvalue { <2 x double>, <2 x double> } undef, <2 x double> %0, 0
; CHECK-NEXT:   %.fca.1.insert = insertvalue { <2 x double>, <2 x double> } %.fca.0.insert, <2 x double> %mul.i, 1
; CHECK-NEXT:   ret { <2 x double>, <2 x double> } %.fca.1.insert
; CHECK-NEXT: }

; CHECK: define internal { { <2 x double>, i8*, i8*, <2 x double> }, <2 x double> } @augmented_subfn(<2 x double>* %W, <2 x double>* %"W'", double %B1, double %B2, i64 %row)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %malloccall = tail call noalias nonnull dereferenceable(16) dereferenceable_or_null(16) i8* @malloc(i64 16)
; CHECK-NEXT:   %"malloccall'mi" = tail call noalias nonnull dereferenceable(16) dereferenceable_or_null(16) i8* @malloc(i64 16)
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* {{(noundef )?}}nonnull align 1 dereferenceable(16) dereferenceable_or_null(16) %"malloccall'mi", i8 0, i64 16, i1 false)
; CHECK-NEXT:   %"Bref'ipc" = bitcast i8* %"malloccall'mi" to <2 x double>*
; CHECK-NEXT:   %Bref = bitcast i8* %malloccall to <2 x double>*
; CHECK-NEXT:   %W34p = getelementptr inbounds <2 x double>, <2 x double>* %W, i64 1
; CHECK-NEXT:   %W34 = load <2 x double>, <2 x double>* %W34p, align 16
; CHECK-NEXT:   %preb1 = insertelement <2 x double> undef, double %B1, i32 0
; CHECK-NEXT:   %B11 = shufflevector <2 x double> %preb1, <2 x double> {{(undef|poison)}}, <2 x i32> zeroinitializer
; CHECK-NEXT:   %preb2 = insertelement <2 x double> undef, double %B2, i32 0
; CHECK-NEXT:   %B22 = shufflevector <2 x double> %preb2, <2 x double> {{(undef|poison)}}, <2 x i32> zeroinitializer
; CHECK-NEXT:   store <2 x double> %B11, <2 x double>* %Bref, align 16
; CHECK-NEXT:   %call_augmented = call { <2 x double>, <2 x double> } @augmented_loadmul(<2 x double>* %W, <2 x double>* %"W'", <2 x double>*{{( nonnull)?}} %Bref, <2 x double>*{{( nonnull)?}} %"Bref'ipc")
; CHECK-NEXT:   %subcache = extractvalue { <2 x double>, <2 x double> } %call_augmented, 0
; CHECK-NEXT:   %call = extractvalue { <2 x double>, <2 x double> } %call_augmented, 1
; CHECK-NEXT:   %mul = fmul <2 x double> %W34, %B22
; CHECK-NEXT:   %add = fadd <2 x double> %mul, %call

; CHECK-NEXT:   %.fca.0.0.insert = insertvalue { { <2 x double>, i8*, i8*, <2 x double> }, <2 x double> } undef, <2 x double> %subcache, 0, 0
; CHECK-NEXT:   %.fca.0.1.insert = insertvalue { { <2 x double>, i8*, i8*, <2 x double> }, <2 x double> } %.fca.0.0.insert, i8* %"malloccall'mi", 0, 1
; CHECK-NEXT:   %.fca.0.2.insert = insertvalue { { <2 x double>, i8*, i8*, <2 x double> }, <2 x double> } %.fca.0.1.insert, i8* %malloccall, 0, 2
; CHECK-NEXT:   %.fca.0.3.insert = insertvalue { { <2 x double>, i8*, i8*, <2 x double> }, <2 x double> } %.fca.0.2.insert, <2 x double> %W34, 0, 3
; CHECK-NEXT:   %.fca.1.insert = insertvalue { { <2 x double>, i8*, i8*, <2 x double> }, <2 x double> } %.fca.0.3.insert, <2 x double> %add, 1
; CHECK-NEXT:   ret { { <2 x double>, i8*, i8*, <2 x double> }, <2 x double> } %.fca.1.insert
; CHECK-NEXT: }

; CHECK: define internal { double, double } @diffesubfn(<2 x double>* %W, <2 x double>* %"W'", double %B1, double %B2, i64 %row, <2 x double> %differeturn, { <2 x double>, i8*, i8*, <2 x double> } %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[malloccall:.+]] = extractvalue { <2 x double>, i8*, i8*, <2 x double> } %tapeArg, 2
; CHECK-NEXT:   %[[malloccallmi:.+]] = extractvalue { <2 x double>, i8*, i8*, <2 x double> } %tapeArg, 1
; CHECK-NEXT:   %[[Brefipc:.+]] = bitcast i8* %[[malloccallmi]] to <2 x double>*
; CHECK-NEXT:   %[[Bref:.+]] = bitcast i8* %[[malloccall]] to <2 x double>*
; CHECK-NEXT:   %[[W34pipge:.+]] = getelementptr inbounds <2 x double>, <2 x double>* %"W'", i64 1

; CHECK-NEXT:   %[[W34:.+]] = extractvalue { <2 x double>, i8*, i8*, <2 x double> } %tapeArg, 3

; CHECK-NEXT:   %preb2 = insertelement <2 x double> undef, double %B2, i32 0
; CHECK-NEXT:   %B22 = shufflevector <2 x double> %preb2, <2 x double> {{(undef|poison)}}, <2 x i32> zeroinitializer
; CHECK-NEXT:   %[[loadmultape:.+]] = extractvalue { <2 x double>, i8*, i8*, <2 x double> } %tapeArg, 0
; CHECK-NEXT:   %m0diffeW34 = fmul fast <2 x double> %B22, %differeturn
; CHECK-NEXT:   %m1diffeB22 = fmul fast <2 x double> %[[W34]], %differeturn


; CHECK-NEXT:   call void @diffeloadmul(<2 x double>* %W, <2 x double>* %"W'", <2 x double>* %[[Bref]], <2 x double>* %[[Brefipc]], <2 x double> %differeturn, <2 x double> %[[loadmultape]])
; CHECK-NEXT:   %[[lbref:.+]] = load <2 x double>, <2 x double>* %[[Brefipc]], align 16
; CHECK-NEXT:   store <2 x double> zeroinitializer, <2 x double>* %[[Brefipc]], align 16
; CHECK-NEXT:   %[[lb221:.+]] = extractelement <2 x double> %m1diffeB22, i32 1
; CHECK-NEXT:   %[[lb220:.+]] = extractelement <2 x double> %m1diffeB22, i32 0
; CHECK-NEXT:   %[[addb22:.+]] = fadd fast double %[[lb221]], %[[lb220]]
; CHECK-NEXT:   %[[bref1:.+]] = extractelement <2 x double> %[[lbref]], i32 1
; CHECK-NEXT:   %[[bref0:.+]] = extractelement <2 x double> %[[lbref]], i32 0
; CHECK-NEXT:   %[[addbref:.+]] = fadd fast double %[[bref1]], %[[bref0]]
; CHECK-NEXT:   %[[lW34:.+]] = load <2 x double>, <2 x double>* %[[W34pipge]], align 16
; CHECK-NEXT:   %[[addW34:.+]] = fadd fast <2 x double> %[[lW34]], %m0diffeW34
; CHECK-NEXT:   store <2 x double> %[[addW34]], <2 x double>* %[[W34pipge]], align 16
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[malloccallmi]])
; CHECK-NEXT:   tail call void @free(i8* %[[malloccall]])
; CHECK-NEXT:   %[[inserted0:.+]] = insertvalue { double, double } undef, double %[[addbref]], 0
; CHECK-NEXT:   %[[inserted1:.+]] = insertvalue { double, double } %[[inserted0]], double %[[addb22]], 1
; CHECK-NEXT:   ret { double, double } %[[inserted1]]
; CHECK-NEXT: }

; CHECK: define internal void @diffeloadmul(<2 x double>* %a, <2 x double>* %"a'", <2 x double>* %b, <2 x double>* %"b'", <2 x double> %differeturn, <2 x double>
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[realb:.+]] = load <2 x double>, <2 x double>* %b, align 16
; CHECK-NEXT:   %m0diffe = fmul fast <2 x double> %[[realb]], %differeturn
; CHECK-NEXT:   %m1diffe = fmul fast <2 x double> %differeturn, %0
; CHECK-NEXT:   %[[lb:.+]] = load <2 x double>, <2 x double>* %"b'", align 16
; CHECK-NEXT:   %[[addB:.+]] = fadd fast <2 x double> %[[lb]], %m1diffe
; CHECK-NEXT:   store <2 x double> %[[addB]], <2 x double>* %"b'", align 16
; CHECK-NEXT:   %[[la:.+]] = load <2 x double>, <2 x double>* %"a'", align 16
; CHECK-NEXT:   %[[addA:.+]] = fadd fast <2 x double> %[[la]], %m0diffe
; CHECK-NEXT:   store <2 x double> %[[addA]], <2 x double>* %"a'", align 16
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
