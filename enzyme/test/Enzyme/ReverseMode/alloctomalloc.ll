; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -simplifycfg -instcombine -gvn -adce -S | FileCheck %s
source_filename = "/home/enzyme/Enzyme/enzyme/test/Integration/simpleeigenstatic-made.cpp"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }
%"class.Eigen::Matrix" = type { %"class.Eigen::PlainObjectBase" }
%"class.Eigen::PlainObjectBase" = type { %"class.Eigen::DenseStorage" }
%"class.Eigen::DenseStorage" = type { %"struct.Eigen::internal::plain_array" }
%"struct.Eigen::internal::plain_array" = type { [4 x double] }
%"class.Eigen::Matrix.6" = type { %"class.Eigen::PlainObjectBase.7" }
%"class.Eigen::PlainObjectBase.7" = type { %"class.Eigen::DenseStorage.14" }
%"class.Eigen::DenseStorage.14" = type { %"struct.Eigen::internal::plain_array.15" }
%"struct.Eigen::internal::plain_array.15" = type { [2 x double] }
%"struct.Eigen::EigenBase.13" = type { i8 }
%"class.Eigen::DenseBase" = type { i8 }

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
  %W = alloca %"class.Eigen::Matrix", align 16
  %M = alloca <2 x double>, align 16
  %tmpcast = bitcast <2 x double>* %M to %"class.Eigen::Matrix.6"*
  %O = alloca <2 x double>, align 16
  %tmpcast1 = bitcast <2 x double>* %O to %"class.Eigen::Matrix.6"*
  %Wp = alloca %"class.Eigen::Matrix", align 16
  %Mp = alloca <2 x double>, align 16
  %tmpcast2 = bitcast <2 x double>* %Mp to %"class.Eigen::Matrix.6"*
  %Op = alloca <2 x double>, align 16
  %tmpcast3 = bitcast <2 x double>* %Op to %"class.Eigen::Matrix.6"*
  %Op_orig = alloca %"class.Eigen::Matrix.6", align 16
  %0 = bitcast %"class.Eigen::Matrix"* %W to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %0) #7
  %1 = bitcast %"class.Eigen::Matrix"* %W to <2 x double>*
  store <2 x double> <double 3.000000e+00, double 3.000000e+00>, <2 x double>* %1, align 16, !tbaa !2
  %2 = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %W, i64 0, i32 0, i32 0, i32 0, i32 0, i64 2
  %3 = bitcast double* %2 to <2 x double>*
  store <2 x double> <double 3.000000e+00, double 3.000000e+00>, <2 x double>* %3, align 16, !tbaa !2
  %4 = bitcast <2 x double>* %M to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %4) #7
  store <2 x double> <double 2.000000e+00, double 2.000000e+00>, <2 x double>* %M, align 16, !tbaa !2
  %5 = bitcast <2 x double>* %O to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %5) #7
  store <2 x double> zeroinitializer, <2 x double>* %O, align 16, !tbaa !2
  %6 = bitcast %"class.Eigen::Matrix"* %Wp to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %6) #7
  %7 = bitcast %"class.Eigen::Matrix"* %Wp to <2 x double>*
  store <2 x double> zeroinitializer, <2 x double>* %7, align 16, !tbaa !2
  %8 = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %Wp, i64 0, i32 0, i32 0, i32 0, i32 0, i64 2
  %9 = bitcast double* %8 to <2 x double>*
  store <2 x double> zeroinitializer, <2 x double>* %9, align 16, !tbaa !2
  %10 = bitcast <2 x double>* %Mp to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %10) #7
  store <2 x double> zeroinitializer, <2 x double>* %Mp, align 16, !tbaa !2
  %11 = bitcast <2 x double>* %Op to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %11) #7
  store <2 x double> <double 1.000000e+00, double 1.000000e+00>, <2 x double>* %Op, align 16, !tbaa !2
  %12 = bitcast %"class.Eigen::Matrix.6"* %Op_orig to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %12) #7
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %12, i8* nonnull align 16 %11, i64 16, i1 false) #7, !tbaa.struct !5
  %call = call double @__enzyme_autodiff(i8* bitcast (void (%"class.Eigen::Matrix"*, %"class.Eigen::Matrix.6"*, %"class.Eigen::Matrix.6"*)* @matvec to i8*), i8* nonnull %0, i8* nonnull %6, i8* nonnull %4, i8* nonnull %10, i8* nonnull %5, i8* nonnull %11) #7
  br label %for.cond12.preheader

for.cond12.preheader:                             ; preds = %for.cond.cleanup15, %entry
  %indvars.iv250 = phi i64 [ 0, %entry ], [ %indvars.iv.next251, %for.cond.cleanup15 ]
  %13 = trunc i64 %indvars.iv250 to i32
  br label %for.body16

for.cond.cleanup15:                               ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEELi1EE8coeffRefEll.exit
  %indvars.iv.next251 = add nuw nsw i64 %indvars.iv250, 1
  %exitcond252 = icmp eq i64 %indvars.iv.next251, 2
  br i1 %exitcond252, label %for.body29, label %for.cond12.preheader

for.body16:                                       ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEELi1EE8coeffRefEll.exit, %for.cond12.preheader
  %indvars.iv247 = phi i64 [ 0, %for.cond12.preheader ], [ %indvars.iv.next248, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEELi1EE8coeffRefEll.exit ]
  %14 = or i64 %indvars.iv247, %indvars.iv250
  %15 = and i64 %14, 9223372036854775806
  %16 = icmp eq i64 %15, 0
  br i1 %16, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEELi1EE8coeffRefEll.exit, label %cond.false.i20

cond.false.i20:                                   ; preds = %for.body16
  call void @__assert_fail(i8* getelementptr inbounds ([53 x i8], [53 x i8]* @.str.27, i64 0, i64 0), i8* getelementptr inbounds ([59 x i8], [59 x i8]* @.str.28, i64 0, i64 0), i32 365, i8* getelementptr inbounds ([219 x i8], [219 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEELi1EEclEll, i64 0, i64 0)) #8
  unreachable

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEELi1EE8coeffRefEll.exit: ; preds = %for.body16
  %17 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !6
  %mul.i.i = shl nuw nsw i64 %indvars.iv250, 1
  %add.i.i = add nuw nsw i64 %mul.i.i, %indvars.iv247
  %arrayidx.i.i15 = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %W, i64 0, i32 0, i32 0, i32 0, i32 0, i64 %add.i.i
  %18 = load double, double* %arrayidx.i.i15, align 8, !tbaa !8
  %19 = trunc i64 %indvars.iv247 to i32
  %call20 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %17, i8* getelementptr inbounds ([18 x i8], [18 x i8]* @.str, i64 0, i64 0), i32 %19, i32 %13, double %18) #9
  %indvars.iv.next248 = add nuw nsw i64 %indvars.iv247, 1
  %exitcond249 = icmp eq i64 %indvars.iv.next248, 2
  br i1 %exitcond249, label %for.cond.cleanup15, label %for.body16

for.body29:                                       ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl.exit27, %for.cond.cleanup15
  %indvars.iv244 = phi i64 [ %indvars.iv.next245, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl.exit27 ], [ 0, %for.cond.cleanup15 ]
  %cmp2.i23 = icmp ult i64 %indvars.iv244, 2
  br i1 %cmp2.i23, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl.exit27, label %cond.false.i25

cond.false.i25:                                   ; preds = %for.body29
  call void @__assert_fail(i8* getelementptr inbounds ([29 x i8], [29 x i8]* @.str.29, i64 0, i64 0), i8* getelementptr inbounds ([59 x i8], [59 x i8]* @.str.28, i64 0, i64 0), i32 425, i8* getelementptr inbounds ([205 x i8], [205 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl, i64 0, i64 0)) #8
  unreachable

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl.exit27: ; preds = %for.body29
  %20 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !6
  %arrayidx.i.i17 = getelementptr inbounds %"class.Eigen::Matrix.6", %"class.Eigen::Matrix.6"* %tmpcast, i64 0, i32 0, i32 0, i32 0, i32 0, i64 %indvars.iv244
  %21 = load double, double* %arrayidx.i.i17, align 8, !tbaa !8
  %22 = trunc i64 %indvars.iv244 to i32
  %call32 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %20, i8* getelementptr inbounds ([12 x i8], [12 x i8]* @.str.1, i64 0, i64 0), i32 %22, double %21) #9
  %indvars.iv.next245 = add nuw nsw i64 %indvars.iv244, 1
  %exitcond246 = icmp eq i64 %indvars.iv.next245, 2
  br i1 %exitcond246, label %for.body41, label %for.body29

for.body41:                                       ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl.exit34, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl.exit27
  %indvars.iv241 = phi i64 [ %indvars.iv.next242, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl.exit34 ], [ 0, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl.exit27 ]
  %cmp2.i30 = icmp ult i64 %indvars.iv241, 2
  br i1 %cmp2.i30, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl.exit34, label %cond.false.i32

cond.false.i32:                                   ; preds = %for.body41
  call void @__assert_fail(i8* getelementptr inbounds ([29 x i8], [29 x i8]* @.str.29, i64 0, i64 0), i8* getelementptr inbounds ([59 x i8], [59 x i8]* @.str.28, i64 0, i64 0), i32 425, i8* getelementptr inbounds ([205 x i8], [205 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl, i64 0, i64 0)) #8
  unreachable

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl.exit34: ; preds = %for.body41
  %23 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !6
  %arrayidx.i.i21 = getelementptr inbounds %"class.Eigen::Matrix.6", %"class.Eigen::Matrix.6"* %tmpcast1, i64 0, i32 0, i32 0, i32 0, i32 0, i64 %indvars.iv241
  %24 = load double, double* %arrayidx.i.i21, align 8, !tbaa !8
  %25 = trunc i64 %indvars.iv241 to i32
  %call44 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %23, i8* getelementptr inbounds ([12 x i8], [12 x i8]* @.str.2, i64 0, i64 0), i32 %25, double %24) #9
  %indvars.iv.next242 = add nuw nsw i64 %indvars.iv241, 1
  %exitcond243 = icmp eq i64 %indvars.iv.next242, 2
  br i1 %exitcond243, label %for.cond55.preheader, label %for.body41

for.cond55.preheader:                             ; preds = %for.cond.cleanup58, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl.exit34
  %indvars.iv239 = phi i64 [ %indvars.iv.next240, %for.cond.cleanup58 ], [ 0, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl.exit34 ]
  %26 = trunc i64 %indvars.iv239 to i32
  br label %for.body59

for.cond.cleanup58:                               ; preds = %if.end
  %indvars.iv.next240 = add nuw nsw i64 %indvars.iv239, 1
  %cmp51 = icmp ult i64 %indvars.iv.next240, 2
  br i1 %cmp51, label %for.cond55.preheader, label %for.cond94.preheader

for.body59:                                       ; preds = %if.end, %for.cond55.preheader
  %indvars.iv237 = phi i64 [ 0, %for.cond55.preheader ], [ %indvars.iv.next238, %if.end ]
  %27 = or i64 %indvars.iv237, %indvars.iv239
  %28 = and i64 %27, 9223372036854775806
  %29 = icmp eq i64 %28, 0
  br i1 %29, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEELi1EE8coeffRefEll.exit37, label %cond.false.i43

cond.false.i43:                                   ; preds = %for.body59
  call void @__assert_fail(i8* getelementptr inbounds ([53 x i8], [53 x i8]* @.str.27, i64 0, i64 0), i8* getelementptr inbounds ([59 x i8], [59 x i8]* @.str.28, i64 0, i64 0), i32 365, i8* getelementptr inbounds ([219 x i8], [219 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEELi1EEclEll, i64 0, i64 0)) #8
  unreachable

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEELi1EE8coeffRefEll.exit37: ; preds = %for.body59
  %mul.i.i34 = shl nuw nsw i64 %indvars.iv239, 1
  %add.i.i35 = add nuw nsw i64 %mul.i.i34, %indvars.iv237
  %arrayidx.i.i36 = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %Wp, i64 0, i32 0, i32 0, i32 0, i32 0, i64 %add.i.i35
  %30 = load double, double* %arrayidx.i.i36, align 8, !tbaa !8
  %cmp2.i55 = icmp ult i64 %indvars.iv237, 2
  br i1 %cmp2.i55, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl.exit59, label %cond.false.i57

cond.false.i57:                                   ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEELi1EE8coeffRefEll.exit37
  call void @__assert_fail(i8* getelementptr inbounds ([29 x i8], [29 x i8]* @.str.29, i64 0, i64 0), i8* getelementptr inbounds ([59 x i8], [59 x i8]* @.str.28, i64 0, i64 0), i32 425, i8* getelementptr inbounds ([205 x i8], [205 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl, i64 0, i64 0)) #8
  unreachable

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl.exit59: ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEELi1EE8coeffRefEll.exit37
  %31 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !6
  %arrayidx.i.i41 = getelementptr inbounds %"class.Eigen::Matrix.6", %"class.Eigen::Matrix.6"* %Op_orig, i64 0, i32 0, i32 0, i32 0, i32 0, i64 %indvars.iv237
  %32 = load double, double* %arrayidx.i.i41, align 8, !tbaa !8
  %arrayidx.i.i39 = getelementptr inbounds %"class.Eigen::Matrix.6", %"class.Eigen::Matrix.6"* %tmpcast, i64 0, i32 0, i32 0, i32 0, i32 0, i64 %indvars.iv239
  %33 = load double, double* %arrayidx.i.i39, align 8, !tbaa !8
  %mul = fmul double %33, %32
  %sub = fsub double %30, %mul
  %34 = call double @llvm.fabs.f64(double %sub)
  %cmp67 = fcmp ogt double %34, 1.000000e-10
  br i1 %cmp67, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl.exit77, label %if.end

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl.exit77: ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl.exit59
  %call76 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %31, i8* getelementptr inbounds ([68 x i8], [68 x i8]* @.str.3, i64 0, i64 0), i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str.4, i64 0, i64 0), double %30, i8* getelementptr inbounds ([18 x i8], [18 x i8]* @.str.5, i64 0, i64 0), double %mul, double 1.000000e-10, i8* getelementptr inbounds ([71 x i8], [71 x i8]* @.str.6, i64 0, i64 0), i32 64, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @__PRETTY_FUNCTION__.main, i64 0, i64 0)) #9
  call void @abort() #8
  unreachable

if.end:                                           ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl.exit59
  %35 = trunc i64 %indvars.iv237 to i32
  %call80 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %31, i8* getelementptr inbounds ([19 x i8], [19 x i8]* @.str.7, i64 0, i64 0), i32 %35, i32 %26, double %30) #9
  %indvars.iv.next238 = add nuw nsw i64 %indvars.iv237, 1
  %cmp57 = icmp ult i64 %indvars.iv.next238, 2
  br i1 %cmp57, label %for.body59, label %for.cond.cleanup58

for.cond94.preheader:                             ; preds = %if.end116, %for.cond.cleanup58
  %indvars.iv235 = phi i64 [ %indvars.iv.next236, %if.end116 ], [ 0, %for.cond.cleanup58 ]
  br label %for.body98

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl.exit91: ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl.exit116
  %arrayidx.i.i55 = getelementptr inbounds %"class.Eigen::Matrix.6", %"class.Eigen::Matrix.6"* %tmpcast2, i64 0, i32 0, i32 0, i32 0, i32 0, i64 %indvars.iv235
  %36 = load double, double* %arrayidx.i.i55, align 8, !tbaa !8
  %sub110 = fsub double %36, %add
  %37 = call double @llvm.fabs.f64(double %sub110)
  %cmp111 = fcmp ogt double %37, 1.000000e-10
  %38 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !6
  br i1 %cmp111, label %if.then112, label %if.end116

for.body98:                                       ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl.exit116, %for.cond94.preheader
  %indvars.iv233 = phi i64 [ 0, %for.cond94.preheader ], [ %indvars.iv.next234, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl.exit116 ]
  %res.0208 = phi double [ 0.000000e+00, %for.cond94.preheader ], [ %add, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl.exit116 ]
  %39 = or i64 %indvars.iv233, %indvars.iv235
  %40 = and i64 %39, 9223372036854775806
  %41 = icmp eq i64 %40, 0
  br i1 %41, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEELi1EE8coeffRefEll.exit29, label %cond.false.i107

cond.false.i107:                                  ; preds = %for.body98
  call void @__assert_fail(i8* getelementptr inbounds ([53 x i8], [53 x i8]* @.str.27, i64 0, i64 0), i8* getelementptr inbounds ([59 x i8], [59 x i8]* @.str.28, i64 0, i64 0), i32 365, i8* getelementptr inbounds ([219 x i8], [219 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEELi1EEclEll, i64 0, i64 0)) #8
  unreachable

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEELi1EE8coeffRefEll.exit29: ; preds = %for.body98
  %cmp2.i112 = icmp ult i64 %indvars.iv233, 2
  br i1 %cmp2.i112, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl.exit116, label %cond.false.i114

cond.false.i114:                                  ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEELi1EE8coeffRefEll.exit29
  call void @__assert_fail(i8* getelementptr inbounds ([29 x i8], [29 x i8]* @.str.29, i64 0, i64 0), i8* getelementptr inbounds ([59 x i8], [59 x i8]* @.str.28, i64 0, i64 0), i32 425, i8* getelementptr inbounds ([205 x i8], [205 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl, i64 0, i64 0)) #8
  unreachable

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl.exit116: ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEELi1EE8coeffRefEll.exit29
  %mul.i.i26 = shl nuw nsw i64 %indvars.iv235, 1
  %add.i.i27 = add nuw nsw i64 %mul.i.i26, %indvars.iv233
  %arrayidx.i.i28 = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %W, i64 0, i32 0, i32 0, i32 0, i32 0, i64 %add.i.i27
  %42 = load double, double* %arrayidx.i.i28, align 8, !tbaa !8
  %arrayidx.i.i19 = getelementptr inbounds %"class.Eigen::Matrix.6", %"class.Eigen::Matrix.6"* %Op_orig, i64 0, i32 0, i32 0, i32 0, i32 0, i64 %indvars.iv233
  %43 = load double, double* %arrayidx.i.i19, align 8, !tbaa !8
  %mul104 = fmul double %42, %43
  %add = fadd double %res.0208, %mul104
  %indvars.iv.next234 = add nuw nsw i64 %indvars.iv233, 1
  %exitcond = icmp eq i64 %indvars.iv.next234, 2
  br i1 %exitcond, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl.exit91, label %for.body98

if.then112:                                       ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl.exit91
  %call115 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %38, i8* getelementptr inbounds ([68 x i8], [68 x i8]* @.str.3, i64 0, i64 0), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.8, i64 0, i64 0), double %36, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.9, i64 0, i64 0), double %add, double 1.000000e-10, i8* getelementptr inbounds ([71 x i8], [71 x i8]* @.str.6, i64 0, i64 0), i32 71, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @__PRETTY_FUNCTION__.main, i64 0, i64 0)) #9
  call void @abort() #8
  unreachable

if.end116:                                        ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl.exit91
  %44 = trunc i64 %indvars.iv235 to i32
  %call119 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %38, i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.10, i64 0, i64 0), i32 %44, double %36) #9
  %indvars.iv.next236 = add nuw nsw i64 %indvars.iv235, 1
  %cmp90 = icmp ult i64 %indvars.iv.next236, 2
  br i1 %cmp90, label %for.cond94.preheader, label %for.body128

for.cond.cleanup127:                              ; preds = %if.end137
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %12) #7
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %11) #7
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %10) #7
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %6) #7
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %5) #7
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %4) #7
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %0) #7
  ret i32 0

for.body128:                                      ; preds = %if.end137, %if.end116
  %indvars.iv = phi i64 [ %indvars.iv.next, %if.end137 ], [ 0, %if.end116 ]
  %cmp2.i119 = icmp ult i64 %indvars.iv, 2
  br i1 %cmp2.i119, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl.exit123, label %cond.false.i121

cond.false.i121:                                  ; preds = %for.body128
  call void @__assert_fail(i8* getelementptr inbounds ([29 x i8], [29 x i8]* @.str.29, i64 0, i64 0), i8* getelementptr inbounds ([59 x i8], [59 x i8]* @.str.28, i64 0, i64 0), i32 425, i8* getelementptr inbounds ([205 x i8], [205 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl, i64 0, i64 0)) #8
  unreachable

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl.exit123: ; preds = %for.body128
  %arrayidx.i.i2 = getelementptr inbounds %"class.Eigen::Matrix.6", %"class.Eigen::Matrix.6"* %tmpcast3, i64 0, i32 0, i32 0, i32 0, i32 0, i64 %indvars.iv
  %45 = load double, double* %arrayidx.i.i2, align 8, !tbaa !8
  %46 = call double @llvm.fabs.f64(double %45)
  %cmp132 = fcmp ogt double %46, 1.000000e-10
  %47 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !6
  br i1 %cmp132, label %if.then133, label %if.end137

if.then133:                                       ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl.exit123
  %call136 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %47, i8* getelementptr inbounds ([68 x i8], [68 x i8]* @.str.3, i64 0, i64 0), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.11, i64 0, i64 0), double %45, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.12, i64 0, i64 0), double 0.000000e+00, double 1.000000e-10, i8* getelementptr inbounds ([71 x i8], [71 x i8]* @.str.6, i64 0, i64 0), i32 76, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @__PRETTY_FUNCTION__.main, i64 0, i64 0)) #9
  call void @abort() #8
  unreachable

if.end137:                                        ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi1ELi0ELi2ELi1EEELi1EEclEl.exit123
  %48 = trunc i64 %indvars.iv to i32
  %call140 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %47, i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.13, i64 0, i64 0), i32 %48, double %45) #9
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %cmp126 = icmp ult i64 %indvars.iv.next, 2
  br i1 %cmp126, label %for.body128, label %for.cond.cleanup127
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

declare dso_local double @__enzyme_autodiff(i8*, i8*, i8*, i8*, i8*, i8*, i8*) local_unnamed_addr #2

; Function Attrs: noinline nounwind uwtable
define internal void @matvec(%"class.Eigen::Matrix"* noalias %W, %"class.Eigen::Matrix.6"* noalias %b, %"class.Eigen::Matrix.6"* noalias %output) #3 {
entry:
  %0 = bitcast %"class.Eigen::Matrix.6"* %output to %"struct.Eigen::EigenBase.13"*
  %a0 = bitcast %"class.Eigen::Matrix.6"* %output to <2 x double>*
  %Bdouble = getelementptr inbounds %"class.Eigen::Matrix.6", %"class.Eigen::Matrix.6"* %b, i64 0, i32 0, i32 0, i32 0, i32 0, i64 0
  ; %B2p = getelementptr inbounds double, double* %Bdouble, i64 1
  ; %B2 = load double, double* %B2p, align 8, !tbaa !8
  ; %B1 = load double, double* %Bdouble, align 8, !tbaa !8

  ; %W12p = bitcast %"class.Eigen::Matrix"* %W to <2 x double>*
  ; %W34p = getelementptr inbounds <2 x double>, <2 x double>* %W12p, i64 1
  ; %W34 = load <2 x double>, <2 x double>* %W34p, align 16, !tbaa !2

  call void @subfn(<2 x double>* %a0, %"class.Eigen::Matrix"* %W, double* %Bdouble) #7
  %call3.i.i = call dereferenceable(16) %"class.Eigen::Matrix.6"* @cast(%"struct.Eigen::EigenBase.13"* %0) #7
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

define void @subfn(<2 x double>* %dst, %"class.Eigen::Matrix"* %W, double* %B) {
entry:
  %tmp.i = alloca <2 x double>

  %subcast = call <2 x double>* @subcast(<2 x double>* %tmp.i) #7

  %unused = call i64 @get2(%"class.Eigen::Matrix"* %W) #7

  %W12p = bitcast %"class.Eigen::Matrix"* %W to <2 x double>*
  %W12 = load <2 x double>, <2 x double>* %W12p

  %B1 = load double, double* %B

  %preb1 = insertelement <2 x double> undef, double %B1, i32 0
  %B11 = insertelement <2 x double> %preb1, double %B1, i32 1

  %mul = fmul <2 x double> %W12, %B11
  %W34p = getelementptr inbounds <2 x double>, <2 x double>* %W12p, i64 1
  %W34 = load <2 x double>, <2 x double>* %W34p

  %B2p = getelementptr inbounds double, double* %B, i64 1
  %B2 = load double, double* %B2p

  %preb2 = insertelement <2 x double> undef, double %B2, i32 0
  %B22 = insertelement <2 x double> %preb2, double %B2, i32 1

  %mul2 = fmul <2 x double> %W34, %B22
  %result = fadd <2 x double> %mul2, %mul

  store <2 x double> %result, <2 x double>* %subcast
  %a13 = load <2 x double>, <2 x double>* %tmp.i
  store <2 x double> %a13, <2 x double>* %dst

  ret void
}


define <2 x double>* @subcast(<2 x double>* %tmp.i) #70 {
entry:
  ; %0 = bitcast <2 x double>* %tmp.i to %"class.Eigen::Matrix.6"*
  ret <2 x double>* %tmp.i
}

define linkonce_odr %"class.Eigen::Matrix.6"* @cast(%"struct.Eigen::EigenBase.13"* %this) {
entry:
  %0 = bitcast %"struct.Eigen::EigenBase.13"* %this to %"class.Eigen::Matrix.6"*
  ret %"class.Eigen::Matrix.6"* %0
}

define i64 @get2(%"class.Eigen::Matrix"* %this) {
entry:
  ret i64 2
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1) #1

attributes #70 = { noinline }
attributes #0 = { alwaysinline norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { nounwind readnone speculatable }
attributes #6 = { noreturn nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #7 = { nounwind }
attributes #8 = { noreturn nounwind }
attributes #9 = { cold }

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

; CHECK: define internal void @diffematvec(%"class.Eigen::Matrix"* noalias %W, %"class.Eigen::Matrix"* %"W'", %"class.Eigen::Matrix.6"* noalias %b, %"class.Eigen::Matrix.6"* %"b'", %"class.Eigen::Matrix.6"* noalias %output, %"class.Eigen::Matrix.6"* %"output'") 
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[ipc:.+]] = bitcast %"class.Eigen::Matrix.6"* %"output'" to %"struct.Eigen::EigenBase.13"*
; CHECK-NEXT:   %[[uw:.+]] = bitcast %"class.Eigen::Matrix.6"* %output to %"struct.Eigen::EigenBase.13"*
; CHECK-NEXT:   %"a0'ipc" = bitcast %"class.Eigen::Matrix.6"* %"output'" to <2 x double>*
; CHECK-NEXT:   %a0 = bitcast %"class.Eigen::Matrix.6"* %output to <2 x double>*
; CHECK-NEXT:   %[[Bdoubleipge:.+]] = getelementptr inbounds %"class.Eigen::Matrix.6", %"class.Eigen::Matrix.6"* %"b'", i64 0, i32 0, i32 0, i32 0, i32 0, i64 0
; CHECK-NEXT:   %Bdouble = getelementptr inbounds %"class.Eigen::Matrix.6", %"class.Eigen::Matrix.6"* %b, i64 0, i32 0, i32 0, i32 0, i32 0, i64 0
; CHECK-NEXT:   %_augmented = call {{.*}} @augmented_subfn(<2 x double>* %a0, <2 x double>* %"a0'ipc", %"class.Eigen::Matrix"* %W, %"class.Eigen::Matrix"* %"W'", double* %Bdouble, double* %[[Bdoubleipge]])
; CHECK-NEXT:   call void @diffecast(%"struct.Eigen::EigenBase.13"* %[[uw]], %"struct.Eigen::EigenBase.13"* %[[ipc]])
; CHECK-NEXT:   call void @diffesubfn(<2 x double>* %a0, <2 x double>* %"a0'ipc", %"class.Eigen::Matrix"* %W, %"class.Eigen::Matrix"* %"W'", double* %Bdouble, double* %[[Bdoubleipge]], {{.*}} %_augmented)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @diffecast(%"struct.Eigen::EigenBase.13"* %this, %"struct.Eigen::EigenBase.13"* %"this'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @augmented_get2(%"class.Eigen::Matrix"* %this, %"class.Eigen::Matrix"* %"this'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal { <2 x double>*, <2 x double>* } @augmented_subcast(<2 x double>* %tmp.i, <2 x double>* %"tmp.i'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %.fca.0.insert = insertvalue { <2 x double>*, <2 x double>* } undef, <2 x double>* %tmp.i, 0
; CHECK-NEXT:   %.fca.1.insert = insertvalue { <2 x double>*, <2 x double>* } %.fca.0.insert, <2 x double>* %"tmp.i'", 1
; CHECK-NEXT:   ret { <2 x double>*, <2 x double>* } %.fca.1.insert
; CHECK-NEXT: }

; CHECK: define internal { <2 x double>*, i8*, i8*, <2 x double>, double, <2 x double>, double } @augmented_subfn(<2 x double>* %dst, <2 x double>* %"dst'", %"class.Eigen::Matrix"* %W, %"class.Eigen::Matrix"* %"W'", double* %B, double* %"B'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %malloccall = tail call noalias nonnull dereferenceable(16) dereferenceable_or_null(16) i8* @malloc(i64 16)
; CHECK-NEXT:   %"malloccall'mi" = tail call noalias nonnull dereferenceable(16) dereferenceable_or_null(16) i8* @malloc(i64 16)
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* {{(noundef )?}}nonnull align 1 dereferenceable(16) dereferenceable_or_null(16) %"malloccall'mi", i8 0, i64 16, i1 false)
; CHECK-NEXT:   %"tmp.i'ipc" = bitcast i8* %"malloccall'mi" to <2 x double>*
; CHECK-NEXT:   %tmp.i = bitcast i8* %malloccall to <2 x double>*
; CHECK-NEXT:   %subcast_augmented = call { <2 x double>*, <2 x double>* } @augmented_subcast(<2 x double>*{{( nonnull)?}} %tmp.i, <2 x double>*{{( nonnull)?}} %"tmp.i'ipc")
; CHECK-NEXT:   %subcast = extractvalue { <2 x double>*, <2 x double>* } %subcast_augmented, 0
; CHECK-NEXT:   %[[antisubcast:.+]] = extractvalue { <2 x double>*, <2 x double>* } %subcast_augmented, 1
; CHECK-NEXT:   call void @augmented_get2(%"class.Eigen::Matrix"* %W, %"class.Eigen::Matrix"* %"W'")
; CHECK-NEXT:   %W12p = bitcast %"class.Eigen::Matrix"* %W to <2 x double>*
; CHECK-NEXT:   %W12 = load <2 x double>, <2 x double>* %W12p, align 16
; CHECK-NEXT:   %B1 = load double, double* %B, align 8
; CHECK-NEXT:   %preb1 = insertelement <2 x double> undef, double %B1, i32 0
; CHECK-NEXT:   %B11 = shufflevector <2 x double> %preb1, <2 x double> {{(undef|poison)?}}, <2 x i32> zeroinitializer
; CHECK-NEXT:   %mul = fmul <2 x double> %W12, %B11
; CHECK-NEXT:   %W34p = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %W, i64 0, i32 0, i32 0, i32 0, i32 0, i64 2
; CHECK-NEXT:   %0 = bitcast double* %W34p to <2 x double>*
; CHECK-NEXT:   %W34 = load <2 x double>, <2 x double>* %0, align 16
; CHECK-NEXT:   %B2p = getelementptr inbounds double, double* %B, i64 1
; CHECK-NEXT:   %B2 = load double, double* %B2p, align 8
; CHECK-NEXT:   %preb2 = insertelement <2 x double> undef, double %B2, i32 0
; CHECK-NEXT:   %B22 = shufflevector <2 x double> %preb2, <2 x double> {{(undef|poison)?}}, <2 x i32> zeroinitializer
; CHECK-NEXT:   %mul2 = fmul <2 x double> %W34, %B22
; CHECK-NEXT:   %result = fadd <2 x double> %mul2, %mul
; CHECK-NEXT:   store <2 x double> %result, <2 x double>* %subcast, align 16
; CHECK-NEXT:   %a13 = load <2 x double>, <2 x double>* %tmp.i, align 16
; CHECK-NEXT:   store <2 x double> %a13, <2 x double>* %dst, align 16
; CHECK-NEXT:   %.fca.0.insert = insertvalue { <2 x double>*, i8*, i8*, <2 x double>, double, <2 x double>, double } undef, <2 x double>* %"subcast'ac", 0
; CHECK-NEXT:   %.fca.1.insert = insertvalue { <2 x double>*, i8*, i8*, <2 x double>, double, <2 x double>, double } %.fca.0.insert, i8* %"malloccall'mi", 1
; CHECK-NEXT:   %.fca.2.insert = insertvalue { <2 x double>*, i8*, i8*, <2 x double>, double, <2 x double>, double } %.fca.1.insert, i8* %malloccall, 2
; CHECK-NEXT:   %.fca.3.insert = insertvalue { <2 x double>*, i8*, i8*, <2 x double>, double, <2 x double>, double } %.fca.2.insert, <2 x double> %W12, 3
; CHECK-NEXT:   %.fca.4.insert = insertvalue { <2 x double>*, i8*, i8*, <2 x double>, double, <2 x double>, double } %.fca.3.insert, double %B1, 4
; CHECK-NEXT:   %.fca.5.insert = insertvalue { <2 x double>*, i8*, i8*, <2 x double>, double, <2 x double>, double } %.fca.4.insert, <2 x double> %W34, 5
; CHECK-NEXT:   %.fca.6.insert = insertvalue { <2 x double>*, i8*, i8*, <2 x double>, double, <2 x double>, double } %.fca.5.insert, double %B2, 6
; CHECK-NEXT:   ret { <2 x double>*, i8*, i8*, <2 x double>, double, <2 x double>, double } %.fca.6.insert
; CHECK-NEXT: }

; CHECK: define internal void @diffesubfn(<2 x double>* %dst, <2 x double>* %"dst'", %"class.Eigen::Matrix"* %W, %"class.Eigen::Matrix"* %"W'", double* %B, double* %"B'", { <2 x double>*, i8*, i8*, <2 x double>, double, <2 x double>, double } %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[malloccall:.+]] = extractvalue { <2 x double>*, i8*, i8*, <2 x double>, double, <2 x double>, double } %tapeArg, 2
; CHECK-NEXT:   %[[malloccallmi:.+]] = extractvalue { <2 x double>*, i8*, i8*, <2 x double>, double, <2 x double>, double } %tapeArg, 1

; CHECK-NEXT:   %[[tmpiipc:.+]] = bitcast i8* %[[malloccallmi]] to <2 x double>*
; CHECK-NEXT:   %[[tmpi:.+]] = bitcast i8* %[[malloccall]] to <2 x double>*

; CHECK-NEXT:   %[[dsubcast:.+]] = extractvalue { <2 x double>*, i8*, i8*, <2 x double>, double, <2 x double>, double } %tapeArg, 0
; CHECK-NEXT:   %[[W12p_ipc1:.+]] = bitcast %"class.Eigen::Matrix"* %"W'" to <2 x double>*

; CHECK-NEXT:   %[[W12:.+]] = extractvalue { <2 x double>*, i8*, i8*, <2 x double>, double, <2 x double>, double } %tapeArg, 3




; CHECK-NEXT:   %B1 = extractvalue { <2 x double>*, i8*, i8*, <2 x double>, double, <2 x double>, double } %tapeArg, 4
; CHECK-NEXT:   %preb1 = insertelement <2 x double> undef, double %B1, i32 0
; CHECK-NEXT:   %B11 = shufflevector <2 x double> %preb1, <2 x double> {{(undef|poison)?}}, <2 x i32> zeroinitializer

; CHECK-NEXT:   %[[W34p_ipge:.+]] = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %"W'", i64 0, i32 0, i32 0, i32 0, i32 0, i64 2
; CHECK-NEXT:   %[[vW34:.+]] = bitcast double* %[[W34p_ipge]] to <2 x double>*

; CHECK-NEXT:   %[[W34:.+]] = extractvalue { <2 x double>*, i8*, i8*, <2 x double>, double, <2 x double>, double } %tapeArg, 5

; CHECK-NEXT:   %[[B2p_ipge:.+]] = getelementptr inbounds double, double* %"B'", i64 1

; CHECK-NEXT:   %B2 = extractvalue { <2 x double>*, i8*, i8*, <2 x double>, double, <2 x double>, double } %tapeArg, 6
; CHECK-NEXT:   %preb2 = insertelement <2 x double> undef, double %B2, i32 0
; CHECK-NEXT:   %B22 = shufflevector <2 x double> %preb2, <2 x double> {{(undef|poison)?}}, <2 x i32> zeroinitializer

; CHECK-NEXT:   %[[dstload:.+]] = load <2 x double>, <2 x double>* %"dst'", align 16
; CHECK-NEXT:   store <2 x double> zeroinitializer, <2 x double>* %"dst'", align 16


; CHECK-NEXT:   %[[oldtmp:.+]] = load <2 x double>, <2 x double>* %[[tmpiipc]], align 16
; CHECK-NEXT:   %[[newdst:.+]] = fadd fast <2 x double> %[[oldtmp]], %[[dstload]]
; CHECK-NEXT:   store <2 x double> %[[newdst]], <2 x double>* %[[tmpiipc]], align 16

; CHECK-NEXT:   %[[loadsc:.+]] = load <2 x double>, <2 x double>* %[[dsubcast]], align 16
; CHECK-NEXT:   store <2 x double> zeroinitializer, <2 x double>* %[[dsubcast]], align 16
; CHECK-NEXT:   %m0diffeW34 = fmul fast <2 x double> %[[loadsc]], %B22
; CHECK-NEXT:   %m1diffeB22 = fmul fast <2 x double> %[[loadsc]], %[[W34]]
; CHECK-NEXT:   %[[b221:.+]] = extractelement <2 x double> %m1diffeB22, i32 1
; CHECK-NEXT:   %[[b220:.+]] = extractelement <2 x double> %m1diffeB22, i32 0
; CHECK-NEXT:   %[[added:.+]] = fadd fast double %[[b221]], %[[b220]]

; CHECK-NEXT:   %[[lb2p:.+]] = load double, double* %[[B2p_ipge]], align 8
; CHECK-NEXT:   %[[lbadd:.+]] = fadd fast double %[[lb2p]], %[[added]]
; CHECK-NEXT:   store double %[[lbadd]], double* %[[B2p_ipge]], align 8


; CHECK-NEXT:   %[[lvW34:.+]] = load <2 x double>, <2 x double>* %[[vW34]], align 16
; CHECK-NEXT:   %[[addDW34:.+]] = fadd fast <2 x double> %[[lvW34]], %m0diffeW34
; CHECK-NEXT:   store <2 x double> %[[addDW34]], <2 x double>* %[[vW34]], align 16

; CHECK-NEXT:   %m0diffeW12 = fmul fast <2 x double> %[[loadsc]], %B11
; CHECK-NEXT:   %m1diffeB11 = fmul fast <2 x double> %[[loadsc]], %[[W12]]
; CHECK-NEXT:   %12 = extractelement <2 x double> %m1diffeB11, i32 1
; CHECK-NEXT:   %13 = extractelement <2 x double> %m1diffeB11, i32 0
; CHECK-NEXT:   %14 = fadd fast double %12, %13
; CHECK-NEXT:   %15 = load double, double* %"B'", align 8
; CHECK-NEXT:   %16 = fadd fast double %15, %14
; CHECK-NEXT:   store double %16, double* %"B'", align 8
; CHECK-NEXT:   %17 = load <2 x double>, <2 x double>* %[[W12p_ipc1]], align 16
; CHECK-NEXT:   %18 = fadd fast <2 x double> %17, %m0diffeW12
; CHECK-NEXT:   store <2 x double> %18, <2 x double>* %[[W12p_ipc1]], align 16
; CHECK-NEXT:   call void @diffeget2(%"class.Eigen::Matrix"* %W, %"class.Eigen::Matrix"* %"W'")

; CHECK-NEXT:   call void @diffesubcast(<2 x double>* %[[tmpi]], <2 x double>* {{(nonnull )?}}%[[tmpiipc]])
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[malloccallmi]])
; CHECK-NEXT:   tail call void @free(i8* %[[malloccall]])
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @diffeget2(%"class.Eigen::Matrix"* %this, %"class.Eigen::Matrix"* %"this'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @diffesubcast(<2 x double>* %tmp.i, <2 x double>* %"tmp.i'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
