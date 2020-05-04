; RUN: %opt < %s %loadEnzyme -enzyme -enzyme_preopt=false -mem2reg -early-cse -simplifycfg -correlated-propagation -instcombine -adce -S | FileCheck %s
source_filename = "/mnt/Data/git/Enzyme/enzyme/test/Integration/eigensumsqdyn.cpp"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }
%"struct.Eigen::internal::CacheSizes" = type { i64, i64, i64 }
%"struct.Eigen::internal::evaluator.73" = type <{ %"struct.Eigen::internal::scalar_constant_op", %"struct.Eigen::internal::nullary_wrapper", [7 x i8] }>
%"struct.Eigen::internal::scalar_constant_op" = type { double }
%"struct.Eigen::internal::nullary_wrapper" = type { i8 }
%"struct.Eigen::internal::evaluator.15" = type { %"struct.Eigen::internal::evaluator.16" }
%"struct.Eigen::internal::evaluator.16" = type { double*, %"class.Eigen::internal::variable_if_dynamic" }
%"class.Eigen::internal::variable_if_dynamic" = type { i64 }
%"class.Eigen::internal::generic_dense_assignment_kernel.76" = type { %"struct.Eigen::internal::evaluator.15"*, %"struct.Eigen::internal::evaluator.73"*, %"struct.Eigen::internal::assign_op"*, %"class.Eigen::Matrix"* }
%"struct.Eigen::internal::assign_op" = type { i8 }
%"class.Eigen::Matrix" = type { %"class.Eigen::PlainObjectBase" }
%"class.Eigen::PlainObjectBase" = type { %"class.Eigen::DenseStorage" }
%"class.Eigen::DenseStorage" = type { double*, i64, i64 }
%"class.Eigen::CwiseNullaryOp" = type { %"class.Eigen::internal::variable_if_dynamic", %"class.Eigen::internal::variable_if_dynamic", %"struct.Eigen::internal::scalar_constant_op" }
%"class.Eigen::CwiseBinaryOp.30" = type <{ %"class.Eigen::Transpose", %"class.Eigen::Block.47", %"struct.Eigen::internal::scalar_product_op", [7 x i8] }>
%"class.Eigen::Transpose" = type { %"class.Eigen::Block" }
%"class.Eigen::Block" = type { %"class.Eigen::BlockImpl" }
%"class.Eigen::BlockImpl" = type { %"class.Eigen::internal::BlockImpl_dense" }
%"class.Eigen::internal::BlockImpl_dense" = type { %"class.Eigen::MapBase", %"class.Eigen::Matrix"*, %"class.Eigen::internal::variable_if_dynamic", %"class.Eigen::internal::variable_if_dynamic", i64 }
%"class.Eigen::MapBase" = type { double*, %"class.Eigen::internal::variable_if_dynamic.46", %"class.Eigen::internal::variable_if_dynamic" }
%"class.Eigen::internal::variable_if_dynamic.46" = type { i8 }
%"class.Eigen::Block.47" = type { %"class.Eigen::BlockImpl.48" }
%"class.Eigen::BlockImpl.48" = type { %"class.Eigen::internal::BlockImpl_dense.49" }
%"class.Eigen::internal::BlockImpl_dense.49" = type { %"class.Eigen::MapBase.base", %"class.Eigen::Matrix"*, %"class.Eigen::internal::variable_if_dynamic", %"class.Eigen::internal::variable_if_dynamic", i64 }
%"class.Eigen::MapBase.base" = type <{ double*, %"class.Eigen::internal::variable_if_dynamic", %"class.Eigen::internal::variable_if_dynamic.46" }>
%"struct.Eigen::internal::scalar_product_op" = type { i8 }
%"struct.Eigen::internal::evaluator.26" = type { %"struct.Eigen::internal::product_evaluator.27" }
%"struct.Eigen::internal::product_evaluator.27" = type { %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"*, %"struct.Eigen::internal::evaluator.15", %"struct.Eigen::internal::evaluator.15", i64 }
%"class.Eigen::internal::generic_dense_assignment_kernel.29" = type { %"struct.Eigen::internal::evaluator.15"*, %"struct.Eigen::internal::evaluator.26"*, %"struct.Eigen::internal::assign_op"*, %"class.Eigen::Matrix"* }
%"class.Eigen::Product.19" = type { %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }
%"class.Eigen::internal::redux_evaluator" = type { %"struct.Eigen::internal::evaluator.18", %"class.Eigen::Product"* }
%"struct.Eigen::internal::evaluator.18" = type { %"struct.Eigen::internal::product_evaluator" }
%"struct.Eigen::internal::product_evaluator" = type { %"struct.Eigen::internal::evaluator.15", %"class.Eigen::Matrix" }
%"class.Eigen::Product" = type { %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }
%"struct.Eigen::internal::evaluator" = type { %"struct.Eigen::internal::binary_evaluator" }
%"struct.Eigen::internal::binary_evaluator" = type { %"struct.Eigen::internal::scalar_difference_op", %"struct.Eigen::internal::evaluator.14", %"struct.Eigen::internal::evaluator.14" }
%"struct.Eigen::internal::scalar_difference_op" = type { i8 }
%"struct.Eigen::internal::evaluator.14" = type { %"struct.Eigen::internal::evaluator.15" }
%"class.Eigen::internal::generic_dense_assignment_kernel" = type { %"struct.Eigen::internal::evaluator.15"*, %"struct.Eigen::internal::evaluator"*, %"struct.Eigen::internal::assign_op"*, %"class.Eigen::Matrix"* }
%"class.Eigen::CwiseBinaryOp" = type <{ %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"*, %"struct.Eigen::internal::scalar_difference_op", [7 x i8] }>

$_ZZN5Eigen8internal20manage_caching_sizesENS_6ActionEPlS2_S2_E12m_cacheSizes = comdat any

$_ZGVZN5Eigen8internal20manage_caching_sizesENS_6ActionEPlS2_S2_E12m_cacheSizes = comdat any

@stderr = external dso_local local_unnamed_addr global %struct._IO_FILE*, align 8
@.str = private unnamed_addr constant [68 x i8] c"Assertion Failed: fabs( [%s = %g] - [%s = %g] ) > %g at %s:%d (%s)\0A\00", align 1
@.str.1 = private unnamed_addr constant [9 x i8] c"Wp(i, o)\00", align 1
@.str.2 = private unnamed_addr constant [4 x i8] c"-8.\00", align 1
@.str.3 = private unnamed_addr constant [63 x i8] c"/mnt/Data/git/Enzyme/enzyme/test/Integration/eigensumsqdyn.cpp\00", align 1
@__PRETTY_FUNCTION__.main = private unnamed_addr constant [23 x i8] c"int main(int, char **)\00", align 1
@.str.4 = private unnamed_addr constant [19 x i8] c"Wp(o=%d, i=%d)=%f\0A\00", align 1
@.str.5 = private unnamed_addr constant [9 x i8] c"Mp(i, o)\00", align 1
@.str.6 = private unnamed_addr constant [3 x i8] c"8.\00", align 1
@.str.7 = private unnamed_addr constant [19 x i8] c"Mp(o=%d, i=%d)=%f\0A\00", align 1
@_ZZN5Eigen8internal20manage_caching_sizesENS_6ActionEPlS2_S2_E12m_cacheSizes = linkonce_odr dso_local global %"struct.Eigen::internal::CacheSizes" zeroinitializer, comdat, align 8
@_ZGVZN5Eigen8internal20manage_caching_sizesENS_6ActionEPlS2_S2_E12m_cacheSizes = linkonce_odr dso_local global i64 0, comdat, align 8
@_ZZN5Eigen8internal15queryCacheSizesERiS1_S1_E12GenuineIntel = private unnamed_addr constant [3 x i32] [i32 1970169159, i32 1231384169, i32 1818588270], align 4
@_ZZN5Eigen8internal15queryCacheSizesERiS1_S1_E12AuthenticAMD = private unnamed_addr constant [3 x i32] [i32 1752462657, i32 1769238117, i32 1145913699], align 4
@_ZZN5Eigen8internal15queryCacheSizesERiS1_S1_E12AMDisbetter_ = private unnamed_addr constant [3 x i32] [i32 1766083905, i32 1952801395, i32 561145204], align 4

; Function Attrs: alwaysinline nounwind uwtable
define dso_local i32 @main(i32 %argc, i8** nocapture readnone %argv) local_unnamed_addr #0 {
entry:
  %srcEvaluator.i.i.i.i.i.i158 = alloca %"struct.Eigen::internal::evaluator.73", align 8
  %dstEvaluator.i.i.i.i.i.i159 = alloca %"struct.Eigen::internal::evaluator.15", align 8
  %kernel.i.i.i.i.i.i160 = alloca %"class.Eigen::internal::generic_dense_assignment_kernel.76", align 8
  %ref.tmp.i.i.i161 = alloca %"struct.Eigen::internal::assign_op", align 1
  %srcEvaluator.i.i.i.i.i.i80 = alloca %"struct.Eigen::internal::evaluator.73", align 8
  %dstEvaluator.i.i.i.i.i.i81 = alloca %"struct.Eigen::internal::evaluator.15", align 8
  %kernel.i.i.i.i.i.i82 = alloca %"class.Eigen::internal::generic_dense_assignment_kernel.76", align 8
  %ref.tmp.i.i.i83 = alloca %"struct.Eigen::internal::assign_op", align 1
  %srcEvaluator.i.i.i.i.i.i2 = alloca %"struct.Eigen::internal::evaluator.73", align 8
  %dstEvaluator.i.i.i.i.i.i3 = alloca %"struct.Eigen::internal::evaluator.15", align 8
  %kernel.i.i.i.i.i.i4 = alloca %"class.Eigen::internal::generic_dense_assignment_kernel.76", align 8
  %ref.tmp.i.i.i5 = alloca %"struct.Eigen::internal::assign_op", align 1
  %srcEvaluator.i.i.i.i.i.i = alloca %"struct.Eigen::internal::evaluator.73", align 8
  %dstEvaluator.i.i.i.i.i.i = alloca %"struct.Eigen::internal::evaluator.15", align 8
  %kernel.i.i.i.i.i.i = alloca %"class.Eigen::internal::generic_dense_assignment_kernel.76", align 8
  %ref.tmp.i.i.i = alloca %"struct.Eigen::internal::assign_op", align 1
  %W = alloca %"class.Eigen::Matrix", align 8
  %ref.tmp = alloca %"class.Eigen::CwiseNullaryOp", align 8
  %M = alloca %"class.Eigen::Matrix", align 8
  %ref.tmp2 = alloca %"class.Eigen::CwiseNullaryOp", align 8
  %Wp = alloca %"class.Eigen::Matrix", align 8
  %ref.tmp4 = alloca %"class.Eigen::CwiseNullaryOp", align 8
  %Mp = alloca %"class.Eigen::Matrix", align 8
  %ref.tmp6 = alloca %"class.Eigen::CwiseNullaryOp", align 8
  %0 = bitcast %"class.Eigen::Matrix"* %W to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %0) #7
  %1 = bitcast %"class.Eigen::CwiseNullaryOp"* %ref.tmp to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %1) #7
  %m_value.i.i.i.i1 = getelementptr inbounds %"class.Eigen::CwiseNullaryOp", %"class.Eigen::CwiseNullaryOp"* %ref.tmp, i64 0, i32 0, i32 0
  store i64 4, i64* %m_value.i.i.i.i1, align 8, !tbaa !2, !alias.scope !7
  %m_value.i1.i.i.i = getelementptr inbounds %"class.Eigen::CwiseNullaryOp", %"class.Eigen::CwiseNullaryOp"* %ref.tmp, i64 0, i32 1, i32 0
  store i64 4, i64* %m_value.i1.i.i.i, align 8, !tbaa !2, !alias.scope !7
  %m_functor.i.i.i = getelementptr inbounds %"class.Eigen::CwiseNullaryOp", %"class.Eigen::CwiseNullaryOp"* %ref.tmp, i64 0, i32 2
  %2 = bitcast %"struct.Eigen::internal::scalar_constant_op"* %m_functor.i.i.i to i64*
  store i64 4607182418800017408, i64* %2, align 8, !tbaa !12, !alias.scope !7
  call void @llvm.memset.p0i8.i64(i8* align 8 %0, i8 0, i64 24, i1 false) #7
  %3 = load i64, i64* %m_value.i.i.i.i1, align 8, !tbaa !2
  %mul.i.i.i.i = shl nsw i64 %3, 2
  %m_rows.i4 = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %W, i64 0, i32 0, i32 0, i32 1
  %4 = load i64, i64* %m_rows.i4, align 8, !tbaa !15
  %m_cols.i5 = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %W, i64 0, i32 0, i32 0, i32 2
  %5 = load i64, i64* %m_cols.i5, align 8, !tbaa !18
  %mul.i6 = mul nsw i64 %5, %4
  %cmp.i = icmp eq i64 %mul.i6, %mul.i.i.i.i
  br i1 %cmp.i, label %_ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EE6resizeElll.exit, label %if.then.i

if.then.i:                                        ; preds = %entry
  %m_data.i7 = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %W, i64 0, i32 0, i32 0, i32 0
  %6 = bitcast %"class.Eigen::Matrix"* %W to i8**
  %7 = load i8*, i8** %6, align 8, !tbaa !19
  call void @free(i8* %7) #7
  %tobool.i = icmp eq i64 %3, 0
  br i1 %tobool.i, label %if.end8.sink.split.i, label %if.end.i.i

if.end.i.i:                                       ; preds = %if.then.i
  %mul.i.i8 = shl i64 %3, 5
  %call.i.i.i.i = call noalias i8* @malloc(i64 %mul.i.i8) #7
  %8 = bitcast i8* %call.i.i.i.i to double*
  br label %if.end8.sink.split.i

if.end8.sink.split.i:                             ; preds = %if.end.i.i, %if.then.i
  %call.sink.i = phi double* [ %8, %if.end.i.i ], [ null, %if.then.i ]
  store double* %call.sink.i, double** %m_data.i7, align 8, !tbaa !19
  br label %_ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EE6resizeElll.exit

_ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EE6resizeElll.exit: ; preds = %if.end8.sink.split.i, %entry
  store i64 %3, i64* %m_rows.i4, align 8, !tbaa !15
  store i64 4, i64* %m_cols.i5, align 8, !tbaa !18
  %9 = getelementptr inbounds %"struct.Eigen::internal::assign_op", %"struct.Eigen::internal::assign_op"* %ref.tmp.i.i.i, i64 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %9) #7
  %10 = bitcast %"struct.Eigen::internal::evaluator.73"* %srcEvaluator.i.i.i.i.i.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %10) #7
  %11 = load i64, i64* %2, align 8, !tbaa !12
  %12 = bitcast %"struct.Eigen::internal::evaluator.73"* %srcEvaluator.i.i.i.i.i.i to i64*
  store i64 %11, i64* %12, align 8, !tbaa !12
  %13 = load i64, i64* %m_value.i.i.i.i1, align 8, !tbaa !2
  %14 = load i64, i64* %m_value.i1.i.i.i, align 8, !tbaa !2
  %15 = load i64, i64* %m_rows.i4, align 8, !tbaa !15
  %cmp.i.i.i.i.i.i.i = icmp eq i64 %15, %13
  %16 = load i64, i64* %m_cols.i5, align 8
  %cmp4.i.i.i.i.i.i.i = icmp eq i64 %16, %14
  %or.cond = and i1 %cmp.i.i.i.i.i.i.i, %cmp4.i.i.i.i.i.i.i
  br i1 %or.cond, label %_ZN5Eigen8internal17resize_if_allowedINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EEddEEvRT_RKT0_RKNS0_9assign_opIT1_T2_EE.exit.i.i.i.i.i.i, label %if.then.i.i.i.i.i.i.i

if.then.i.i.i.i.i.i.i:                            ; preds = %_ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EE6resizeElll.exit
  %mul.i.i.i.i.i.i.i.i = mul nsw i64 %14, %13
  %mul.i13 = mul nsw i64 %16, %15
  %cmp.i14 = icmp eq i64 %mul.i13, %mul.i.i.i.i.i.i.i.i
  br i1 %cmp.i14, label %_ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EE6resizeElll.exit35, label %if.then.i17

if.then.i17:                                      ; preds = %if.then.i.i.i.i.i.i.i
  %m_data.i15 = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %W, i64 0, i32 0, i32 0, i32 0
  %17 = bitcast %"class.Eigen::Matrix"* %W to i8**
  %18 = load i8*, i8** %17, align 8, !tbaa !19
  call void @free(i8* %18) #7
  %tobool.i16 = icmp eq i64 %mul.i.i.i.i.i.i.i.i, 0
  br i1 %tobool.i16, label %if.end8.sink.split.i34, label %if.end.i.i21

if.end.i.i21:                                     ; preds = %if.then.i17
  %mul.i.i23 = shl i64 %mul.i.i.i.i.i.i.i.i, 3
  %call.i.i.i.i24 = call noalias i8* @malloc(i64 %mul.i.i23) #7
  %19 = bitcast i8* %call.i.i.i.i24 to double*
  br label %if.end8.sink.split.i34

if.end8.sink.split.i34:                           ; preds = %if.end.i.i21, %if.then.i17
  %call.sink.i33 = phi double* [ %19, %if.end.i.i21 ], [ null, %if.then.i17 ]
  store double* %call.sink.i33, double** %m_data.i15, align 8, !tbaa !19
  br label %_ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EE6resizeElll.exit35

_ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EE6resizeElll.exit35: ; preds = %if.end8.sink.split.i34, %if.then.i.i.i.i.i.i.i
  store i64 %13, i64* %m_rows.i4, align 8, !tbaa !15
  store i64 %14, i64* %m_cols.i5, align 8, !tbaa !18
  br label %_ZN5Eigen8internal17resize_if_allowedINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EEddEEvRT_RKT0_RKNS0_9assign_opIT1_T2_EE.exit.i.i.i.i.i.i

_ZN5Eigen8internal17resize_if_allowedINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EEddEEvRT_RKT0_RKNS0_9assign_opIT1_T2_EE.exit.i.i.i.i.i.i: ; preds = %_ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EE6resizeElll.exit35, %_ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EE6resizeElll.exit
  %20 = bitcast %"struct.Eigen::internal::evaluator.15"* %dstEvaluator.i.i.i.i.i.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %20) #7
  %21 = bitcast %"class.Eigen::Matrix"* %W to i64*
  %22 = load i64, i64* %21, align 8, !tbaa !19
  %23 = bitcast %"struct.Eigen::internal::evaluator.15"* %dstEvaluator.i.i.i.i.i.i to i64*
  store i64 %22, i64* %23, align 8, !tbaa !20
  %24 = load i64, i64* %m_rows.i4, align 8, !tbaa !15
  %m_value.i.i.i65 = getelementptr inbounds %"struct.Eigen::internal::evaluator.15", %"struct.Eigen::internal::evaluator.15"* %dstEvaluator.i.i.i.i.i.i, i64 0, i32 0, i32 1, i32 0
  store i64 %24, i64* %m_value.i.i.i65, align 8, !tbaa !2
  %25 = bitcast %"class.Eigen::internal::generic_dense_assignment_kernel.76"* %kernel.i.i.i.i.i.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %25) #7
  %m_dst.i66 = getelementptr inbounds %"class.Eigen::internal::generic_dense_assignment_kernel.76", %"class.Eigen::internal::generic_dense_assignment_kernel.76"* %kernel.i.i.i.i.i.i, i64 0, i32 0
  store %"struct.Eigen::internal::evaluator.15"* %dstEvaluator.i.i.i.i.i.i, %"struct.Eigen::internal::evaluator.15"** %m_dst.i66, align 8, !tbaa !22
  %m_src.i67 = getelementptr inbounds %"class.Eigen::internal::generic_dense_assignment_kernel.76", %"class.Eigen::internal::generic_dense_assignment_kernel.76"* %kernel.i.i.i.i.i.i, i64 0, i32 1
  store %"struct.Eigen::internal::evaluator.73"* %srcEvaluator.i.i.i.i.i.i, %"struct.Eigen::internal::evaluator.73"** %m_src.i67, align 8, !tbaa !22
  %m_functor.i68 = getelementptr inbounds %"class.Eigen::internal::generic_dense_assignment_kernel.76", %"class.Eigen::internal::generic_dense_assignment_kernel.76"* %kernel.i.i.i.i.i.i, i64 0, i32 2
  store %"struct.Eigen::internal::assign_op"* %ref.tmp.i.i.i, %"struct.Eigen::internal::assign_op"** %m_functor.i68, align 8, !tbaa !22
  %m_dstExpr.i69 = getelementptr inbounds %"class.Eigen::internal::generic_dense_assignment_kernel.76", %"class.Eigen::internal::generic_dense_assignment_kernel.76"* %kernel.i.i.i.i.i.i, i64 0, i32 3
  store %"class.Eigen::Matrix"* %W, %"class.Eigen::Matrix"** %m_dstExpr.i69, align 8, !tbaa !22
  %26 = load i64, i64* %m_rows.i4, align 8, !tbaa !15
  %27 = load i64, i64* %m_cols.i5, align 8, !tbaa !18
  %mul.i.i92 = mul nsw i64 %27, %26
  %cmp6.i.i.i.i.i.i.i = icmp sgt i64 %mul.i.i92, 0
  br i1 %cmp6.i.i.i.i.i.i.i, label %for.body.i.i.i.i.i.i.i.preheader, label %_ZN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEC2INS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES1_EEEERKNS_9EigenBaseIT_EE.exit

for.body.i.i.i.i.i.i.i.preheader:                 ; preds = %_ZN5Eigen8internal17resize_if_allowedINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EEddEEvRT_RKT0_RKNS0_9assign_opIT1_T2_EE.exit.i.i.i.i.i.i
  %28 = bitcast %"class.Eigen::internal::generic_dense_assignment_kernel.76"* %kernel.i.i.i.i.i.i to %"struct.Eigen::internal::evaluator.16"**
  %29 = load %"struct.Eigen::internal::evaluator.16"*, %"struct.Eigen::internal::evaluator.16"** %28, align 8, !tbaa !23
  %m_data.i.i.i.i.i.i.i.i.i = getelementptr inbounds %"struct.Eigen::internal::evaluator.16", %"struct.Eigen::internal::evaluator.16"* %29, i64 0, i32 0
  %30 = bitcast %"struct.Eigen::internal::evaluator.73"** %m_src.i67 to i64**
  %31 = load i64*, i64** %30, align 8, !tbaa !25
  br label %for.body.i.i.i.i.i.i.i

for.body.i.i.i.i.i.i.i:                           ; preds = %for.body.i.i.i.i.i.i.i, %for.body.i.i.i.i.i.i.i.preheader
  %i.07.i.i.i.i.i.i.i = phi i64 [ %inc.i.i.i.i.i.i.i, %for.body.i.i.i.i.i.i.i ], [ 0, %for.body.i.i.i.i.i.i.i.preheader ]
  %32 = load double*, double** %m_data.i.i.i.i.i.i.i.i.i, align 8, !tbaa !20
  %arrayidx.i.i.i.i.i.i.i.i.i = getelementptr inbounds double, double* %32, i64 %i.07.i.i.i.i.i.i.i
  %33 = load i64, i64* %31, align 8, !tbaa !12
  %34 = bitcast double* %arrayidx.i.i.i.i.i.i.i.i.i to i64*
  store i64 %33, i64* %34, align 8, !tbaa !26
  %inc.i.i.i.i.i.i.i = add nuw nsw i64 %i.07.i.i.i.i.i.i.i, 1
  %exitcond.i.i.i.i.i.i.i = icmp eq i64 %inc.i.i.i.i.i.i.i, %mul.i.i92
  br i1 %exitcond.i.i.i.i.i.i.i, label %_ZN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEC2INS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES1_EEEERKNS_9EigenBaseIT_EE.exit, label %for.body.i.i.i.i.i.i.i

_ZN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEC2INS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES1_EEEERKNS_9EigenBaseIT_EE.exit: ; preds = %for.body.i.i.i.i.i.i.i, %_ZN5Eigen8internal17resize_if_allowedINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EEddEEvRT_RKT0_RKNS0_9assign_opIT1_T2_EE.exit.i.i.i.i.i.i
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %25) #7
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %20) #7
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %10) #7
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %9) #7
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %1) #7
  %35 = bitcast %"class.Eigen::Matrix"* %M to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %35) #7
  %36 = bitcast %"class.Eigen::CwiseNullaryOp"* %ref.tmp2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %36) #7
  %m_value.i.i.i.i38 = getelementptr inbounds %"class.Eigen::CwiseNullaryOp", %"class.Eigen::CwiseNullaryOp"* %ref.tmp2, i64 0, i32 0, i32 0
  store i64 4, i64* %m_value.i.i.i.i38, align 8, !tbaa !2, !alias.scope !27
  %m_value.i1.i.i.i40 = getelementptr inbounds %"class.Eigen::CwiseNullaryOp", %"class.Eigen::CwiseNullaryOp"* %ref.tmp2, i64 0, i32 1, i32 0
  store i64 4, i64* %m_value.i1.i.i.i40, align 8, !tbaa !2, !alias.scope !27
  %m_functor.i.i.i41 = getelementptr inbounds %"class.Eigen::CwiseNullaryOp", %"class.Eigen::CwiseNullaryOp"* %ref.tmp2, i64 0, i32 2
  %37 = bitcast %"struct.Eigen::internal::scalar_constant_op"* %m_functor.i.i.i41 to i64*
  store i64 4611686018427387904, i64* %37, align 8, !tbaa !12, !alias.scope !27
  call void @llvm.memset.p0i8.i64(i8* align 8 %35, i8 0, i64 24, i1 false) #7
  %38 = load i64, i64* %m_value.i.i.i.i38, align 8, !tbaa !2
  %mul.i.i.i.i37 = shl nsw i64 %38, 2
  %m_rows.i42 = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %M, i64 0, i32 0, i32 0, i32 1
  %39 = load i64, i64* %m_rows.i42, align 8, !tbaa !15
  %m_cols.i43 = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %M, i64 0, i32 0, i32 0, i32 2
  %40 = load i64, i64* %m_cols.i43, align 8, !tbaa !18
  %mul.i44 = mul nsw i64 %40, %39
  %cmp.i45 = icmp eq i64 %mul.i44, %mul.i.i.i.i37
  br i1 %cmp.i45, label %_ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EE6resizeElll.exit66, label %if.then.i48

if.then.i48:                                      ; preds = %_ZN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEC2INS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES1_EEEERKNS_9EigenBaseIT_EE.exit
  %m_data.i46 = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %M, i64 0, i32 0, i32 0, i32 0
  %41 = bitcast %"class.Eigen::Matrix"* %M to i8**
  %42 = load i8*, i8** %41, align 8, !tbaa !19
  call void @free(i8* %42) #7
  %tobool.i47 = icmp eq i64 %38, 0
  br i1 %tobool.i47, label %if.end8.sink.split.i65, label %if.end.i.i52

if.end.i.i52:                                     ; preds = %if.then.i48
  %mul.i.i54 = shl i64 %38, 5
  %call.i.i.i.i55 = call noalias i8* @malloc(i64 %mul.i.i54) #7
  %43 = bitcast i8* %call.i.i.i.i55 to double*
  br label %if.end8.sink.split.i65

if.end8.sink.split.i65:                           ; preds = %if.end.i.i52, %if.then.i48
  %call.sink.i64 = phi double* [ %43, %if.end.i.i52 ], [ null, %if.then.i48 ]
  store double* %call.sink.i64, double** %m_data.i46, align 8, !tbaa !19
  br label %_ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EE6resizeElll.exit66

_ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EE6resizeElll.exit66: ; preds = %if.end8.sink.split.i65, %_ZN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEC2INS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES1_EEEERKNS_9EigenBaseIT_EE.exit
  store i64 %38, i64* %m_rows.i42, align 8, !tbaa !15
  store i64 4, i64* %m_cols.i43, align 8, !tbaa !18
  %44 = getelementptr inbounds %"struct.Eigen::internal::assign_op", %"struct.Eigen::internal::assign_op"* %ref.tmp.i.i.i5, i64 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %44) #7
  %45 = bitcast %"struct.Eigen::internal::evaluator.73"* %srcEvaluator.i.i.i.i.i.i2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %45) #7
  %46 = load i64, i64* %37, align 8, !tbaa !12
  %47 = bitcast %"struct.Eigen::internal::evaluator.73"* %srcEvaluator.i.i.i.i.i.i2 to i64*
  store i64 %46, i64* %47, align 8, !tbaa !12
  %48 = load i64, i64* %m_value.i.i.i.i38, align 8, !tbaa !2
  %49 = load i64, i64* %m_value.i1.i.i.i40, align 8, !tbaa !2
  %50 = load i64, i64* %m_rows.i42, align 8, !tbaa !15
  %cmp.i.i.i.i.i.i.i46 = icmp eq i64 %50, %48
  %51 = load i64, i64* %m_cols.i43, align 8
  %cmp4.i.i.i.i.i.i.i50 = icmp eq i64 %51, %49
  %or.cond8 = and i1 %cmp.i.i.i.i.i.i.i46, %cmp4.i.i.i.i.i.i.i50
  br i1 %or.cond8, label %_ZN5Eigen8internal17resize_if_allowedINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EEddEEvRT_RKT0_RKNS0_9assign_opIT1_T2_EE.exit.i.i.i.i.i.i65, label %if.then.i.i.i.i.i.i.i55

if.then.i.i.i.i.i.i.i55:                          ; preds = %_ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EE6resizeElll.exit66
  %mul.i.i.i.i.i.i.i.i61 = mul nsw i64 %49, %48
  %mul.i69 = mul nsw i64 %51, %50
  %cmp.i70 = icmp eq i64 %mul.i69, %mul.i.i.i.i.i.i.i.i61
  br i1 %cmp.i70, label %_ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EE6resizeElll.exit91, label %if.then.i73

if.then.i73:                                      ; preds = %if.then.i.i.i.i.i.i.i55
  %m_data.i71 = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %M, i64 0, i32 0, i32 0, i32 0
  %52 = bitcast %"class.Eigen::Matrix"* %M to i8**
  %53 = load i8*, i8** %52, align 8, !tbaa !19
  call void @free(i8* %53) #7
  %tobool.i72 = icmp eq i64 %mul.i.i.i.i.i.i.i.i61, 0
  br i1 %tobool.i72, label %if.end8.sink.split.i90, label %if.end.i.i77

if.end.i.i77:                                     ; preds = %if.then.i73
  %mul.i.i79 = shl i64 %mul.i.i.i.i.i.i.i.i61, 3
  %call.i.i.i.i80 = call noalias i8* @malloc(i64 %mul.i.i79) #7
  %54 = bitcast i8* %call.i.i.i.i80 to double*
  br label %if.end8.sink.split.i90

if.end8.sink.split.i90:                           ; preds = %if.end.i.i77, %if.then.i73
  %call.sink.i89 = phi double* [ %54, %if.end.i.i77 ], [ null, %if.then.i73 ]
  store double* %call.sink.i89, double** %m_data.i71, align 8, !tbaa !19
  br label %_ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EE6resizeElll.exit91

_ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EE6resizeElll.exit91: ; preds = %if.end8.sink.split.i90, %if.then.i.i.i.i.i.i.i55
  store i64 %48, i64* %m_rows.i42, align 8, !tbaa !15
  store i64 %49, i64* %m_cols.i43, align 8, !tbaa !18
  br label %_ZN5Eigen8internal17resize_if_allowedINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EEddEEvRT_RKT0_RKNS0_9assign_opIT1_T2_EE.exit.i.i.i.i.i.i65

_ZN5Eigen8internal17resize_if_allowedINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EEddEEvRT_RKT0_RKNS0_9assign_opIT1_T2_EE.exit.i.i.i.i.i.i65: ; preds = %_ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EE6resizeElll.exit91, %_ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EE6resizeElll.exit66
  %55 = bitcast %"struct.Eigen::internal::evaluator.15"* %dstEvaluator.i.i.i.i.i.i3 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %55) #7
  %56 = bitcast %"class.Eigen::Matrix"* %M to i64*
  %57 = load i64, i64* %56, align 8, !tbaa !19
  %58 = bitcast %"struct.Eigen::internal::evaluator.15"* %dstEvaluator.i.i.i.i.i.i3 to i64*
  store i64 %57, i64* %58, align 8, !tbaa !20
  %59 = load i64, i64* %m_rows.i42, align 8, !tbaa !15
  %m_value.i.i.i86 = getelementptr inbounds %"struct.Eigen::internal::evaluator.15", %"struct.Eigen::internal::evaluator.15"* %dstEvaluator.i.i.i.i.i.i3, i64 0, i32 0, i32 1, i32 0
  store i64 %59, i64* %m_value.i.i.i86, align 8, !tbaa !2
  %60 = bitcast %"class.Eigen::internal::generic_dense_assignment_kernel.76"* %kernel.i.i.i.i.i.i4 to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %60) #7
  %m_dst.i76 = getelementptr inbounds %"class.Eigen::internal::generic_dense_assignment_kernel.76", %"class.Eigen::internal::generic_dense_assignment_kernel.76"* %kernel.i.i.i.i.i.i4, i64 0, i32 0
  store %"struct.Eigen::internal::evaluator.15"* %dstEvaluator.i.i.i.i.i.i3, %"struct.Eigen::internal::evaluator.15"** %m_dst.i76, align 8, !tbaa !22
  %m_src.i77 = getelementptr inbounds %"class.Eigen::internal::generic_dense_assignment_kernel.76", %"class.Eigen::internal::generic_dense_assignment_kernel.76"* %kernel.i.i.i.i.i.i4, i64 0, i32 1
  store %"struct.Eigen::internal::evaluator.73"* %srcEvaluator.i.i.i.i.i.i2, %"struct.Eigen::internal::evaluator.73"** %m_src.i77, align 8, !tbaa !22
  %m_functor.i78 = getelementptr inbounds %"class.Eigen::internal::generic_dense_assignment_kernel.76", %"class.Eigen::internal::generic_dense_assignment_kernel.76"* %kernel.i.i.i.i.i.i4, i64 0, i32 2
  store %"struct.Eigen::internal::assign_op"* %ref.tmp.i.i.i5, %"struct.Eigen::internal::assign_op"** %m_functor.i78, align 8, !tbaa !22
  %m_dstExpr.i79 = getelementptr inbounds %"class.Eigen::internal::generic_dense_assignment_kernel.76", %"class.Eigen::internal::generic_dense_assignment_kernel.76"* %kernel.i.i.i.i.i.i4, i64 0, i32 3
  store %"class.Eigen::Matrix"* %M, %"class.Eigen::Matrix"** %m_dstExpr.i79, align 8, !tbaa !22
  %61 = load i64, i64* %m_rows.i42, align 8, !tbaa !15
  %62 = load i64, i64* %m_cols.i43, align 8, !tbaa !18
  %mul.i.i75 = mul nsw i64 %62, %61
  %cmp6.i.i.i.i.i.i.i64 = icmp sgt i64 %mul.i.i75, 0
  br i1 %cmp6.i.i.i.i.i.i.i64, label %for.body.i.i.i.i.i.i.i76.preheader, label %_ZN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEC2INS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES1_EEEERKNS_9EigenBaseIT_EE.exit78

for.body.i.i.i.i.i.i.i76.preheader:               ; preds = %_ZN5Eigen8internal17resize_if_allowedINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EEddEEvRT_RKT0_RKNS0_9assign_opIT1_T2_EE.exit.i.i.i.i.i.i65
  %63 = bitcast %"class.Eigen::internal::generic_dense_assignment_kernel.76"* %kernel.i.i.i.i.i.i4 to %"struct.Eigen::internal::evaluator.16"**
  %64 = load %"struct.Eigen::internal::evaluator.16"*, %"struct.Eigen::internal::evaluator.16"** %63, align 8, !tbaa !23
  %m_data.i.i.i.i.i.i.i.i.i68 = getelementptr inbounds %"struct.Eigen::internal::evaluator.16", %"struct.Eigen::internal::evaluator.16"* %64, i64 0, i32 0
  %65 = bitcast %"struct.Eigen::internal::evaluator.73"** %m_src.i77 to i64**
  %66 = load i64*, i64** %65, align 8, !tbaa !25
  br label %for.body.i.i.i.i.i.i.i76

for.body.i.i.i.i.i.i.i76:                         ; preds = %for.body.i.i.i.i.i.i.i76, %for.body.i.i.i.i.i.i.i76.preheader
  %i.07.i.i.i.i.i.i.i66 = phi i64 [ %inc.i.i.i.i.i.i.i74, %for.body.i.i.i.i.i.i.i76 ], [ 0, %for.body.i.i.i.i.i.i.i76.preheader ]
  %67 = load double*, double** %m_data.i.i.i.i.i.i.i.i.i68, align 8, !tbaa !20
  %arrayidx.i.i.i.i.i.i.i.i.i69 = getelementptr inbounds double, double* %67, i64 %i.07.i.i.i.i.i.i.i66
  %68 = load i64, i64* %66, align 8, !tbaa !12
  %69 = bitcast double* %arrayidx.i.i.i.i.i.i.i.i.i69 to i64*
  store i64 %68, i64* %69, align 8, !tbaa !26
  %inc.i.i.i.i.i.i.i74 = add nuw nsw i64 %i.07.i.i.i.i.i.i.i66, 1
  %exitcond.i.i.i.i.i.i.i75 = icmp eq i64 %inc.i.i.i.i.i.i.i74, %mul.i.i75
  br i1 %exitcond.i.i.i.i.i.i.i75, label %_ZN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEC2INS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES1_EEEERKNS_9EigenBaseIT_EE.exit78, label %for.body.i.i.i.i.i.i.i76

_ZN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEC2INS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES1_EEEERKNS_9EigenBaseIT_EE.exit78: ; preds = %for.body.i.i.i.i.i.i.i76, %_ZN5Eigen8internal17resize_if_allowedINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EEddEEvRT_RKT0_RKNS0_9assign_opIT1_T2_EE.exit.i.i.i.i.i.i65
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %60) #7
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %55) #7
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %45) #7
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %44) #7
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %36) #7
  %70 = bitcast %"class.Eigen::Matrix"* %Wp to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %70) #7
  %71 = bitcast %"class.Eigen::CwiseNullaryOp"* %ref.tmp4 to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %71) #7
  %m_value.i.i.i.i94 = getelementptr inbounds %"class.Eigen::CwiseNullaryOp", %"class.Eigen::CwiseNullaryOp"* %ref.tmp4, i64 0, i32 0, i32 0
  store i64 4, i64* %m_value.i.i.i.i94, align 8, !tbaa !2, !alias.scope !32
  %m_value.i1.i.i.i96 = getelementptr inbounds %"class.Eigen::CwiseNullaryOp", %"class.Eigen::CwiseNullaryOp"* %ref.tmp4, i64 0, i32 1, i32 0
  store i64 4, i64* %m_value.i1.i.i.i96, align 8, !tbaa !2, !alias.scope !32
  %m_functor.i.i.i97 = getelementptr inbounds %"class.Eigen::CwiseNullaryOp", %"class.Eigen::CwiseNullaryOp"* %ref.tmp4, i64 0, i32 2
  %72 = bitcast %"struct.Eigen::internal::scalar_constant_op"* %m_functor.i.i.i97 to i64*
  store i64 0, i64* %72, align 8, !tbaa !12, !alias.scope !32
  call void @llvm.memset.p0i8.i64(i8* align 8 %70, i8 0, i64 24, i1 false) #7
  %73 = load i64, i64* %m_value.i.i.i.i94, align 8, !tbaa !2
  %mul.i.i.i.i115 = shl nsw i64 %73, 2
  %m_rows.i98 = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %Wp, i64 0, i32 0, i32 0, i32 1
  %74 = load i64, i64* %m_rows.i98, align 8, !tbaa !15
  %m_cols.i99 = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %Wp, i64 0, i32 0, i32 0, i32 2
  %75 = load i64, i64* %m_cols.i99, align 8, !tbaa !18
  %mul.i100 = mul nsw i64 %75, %74
  %cmp.i101 = icmp eq i64 %mul.i100, %mul.i.i.i.i115
  br i1 %cmp.i101, label %_ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EE6resizeElll.exit122, label %if.then.i104

if.then.i104:                                     ; preds = %_ZN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEC2INS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES1_EEEERKNS_9EigenBaseIT_EE.exit78
  %m_data.i102 = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %Wp, i64 0, i32 0, i32 0, i32 0
  %76 = bitcast %"class.Eigen::Matrix"* %Wp to i8**
  %77 = load i8*, i8** %76, align 8, !tbaa !19
  call void @free(i8* %77) #7
  %tobool.i103 = icmp eq i64 %73, 0
  br i1 %tobool.i103, label %if.end8.sink.split.i121, label %if.end.i.i108

if.end.i.i108:                                    ; preds = %if.then.i104
  %mul.i.i110 = shl i64 %73, 5
  %call.i.i.i.i111 = call noalias i8* @malloc(i64 %mul.i.i110) #7
  %78 = bitcast i8* %call.i.i.i.i111 to double*
  br label %if.end8.sink.split.i121

if.end8.sink.split.i121:                          ; preds = %if.end.i.i108, %if.then.i104
  %call.sink.i120 = phi double* [ %78, %if.end.i.i108 ], [ null, %if.then.i104 ]
  store double* %call.sink.i120, double** %m_data.i102, align 8, !tbaa !19
  br label %_ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EE6resizeElll.exit122

_ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EE6resizeElll.exit122: ; preds = %if.end8.sink.split.i121, %_ZN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEC2INS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES1_EEEERKNS_9EigenBaseIT_EE.exit78
  store i64 %73, i64* %m_rows.i98, align 8, !tbaa !15
  store i64 4, i64* %m_cols.i99, align 8, !tbaa !18
  %79 = getelementptr inbounds %"struct.Eigen::internal::assign_op", %"struct.Eigen::internal::assign_op"* %ref.tmp.i.i.i83, i64 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %79) #7
  %80 = bitcast %"struct.Eigen::internal::evaluator.73"* %srcEvaluator.i.i.i.i.i.i80 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %80) #7
  %81 = load i64, i64* %72, align 8, !tbaa !12
  %82 = bitcast %"struct.Eigen::internal::evaluator.73"* %srcEvaluator.i.i.i.i.i.i80 to i64*
  store i64 %81, i64* %82, align 8, !tbaa !12
  %83 = load i64, i64* %m_value.i.i.i.i94, align 8, !tbaa !2
  %84 = load i64, i64* %m_value.i1.i.i.i96, align 8, !tbaa !2
  %85 = load i64, i64* %m_rows.i98, align 8, !tbaa !15
  %cmp.i.i.i.i.i.i.i124 = icmp eq i64 %85, %83
  %86 = load i64, i64* %m_cols.i99, align 8
  %cmp4.i.i.i.i.i.i.i128 = icmp eq i64 %86, %84
  %or.cond9 = and i1 %cmp.i.i.i.i.i.i.i124, %cmp4.i.i.i.i.i.i.i128
  br i1 %or.cond9, label %_ZN5Eigen8internal17resize_if_allowedINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EEddEEvRT_RKT0_RKNS0_9assign_opIT1_T2_EE.exit.i.i.i.i.i.i143, label %if.then.i.i.i.i.i.i.i133

if.then.i.i.i.i.i.i.i133:                         ; preds = %_ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EE6resizeElll.exit122
  %mul.i.i.i.i.i.i.i.i139 = mul nsw i64 %84, %83
  %mul.i125 = mul nsw i64 %86, %85
  %cmp.i126 = icmp eq i64 %mul.i125, %mul.i.i.i.i.i.i.i.i139
  br i1 %cmp.i126, label %_ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EE6resizeElll.exit147, label %if.then.i129

if.then.i129:                                     ; preds = %if.then.i.i.i.i.i.i.i133
  %m_data.i127 = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %Wp, i64 0, i32 0, i32 0, i32 0
  %87 = bitcast %"class.Eigen::Matrix"* %Wp to i8**
  %88 = load i8*, i8** %87, align 8, !tbaa !19
  call void @free(i8* %88) #7
  %tobool.i128 = icmp eq i64 %mul.i.i.i.i.i.i.i.i139, 0
  br i1 %tobool.i128, label %if.end8.sink.split.i146, label %if.end.i.i133

if.end.i.i133:                                    ; preds = %if.then.i129
  %mul.i.i135 = shl i64 %mul.i.i.i.i.i.i.i.i139, 3
  %call.i.i.i.i136 = call noalias i8* @malloc(i64 %mul.i.i135) #7
  %89 = bitcast i8* %call.i.i.i.i136 to double*
  br label %if.end8.sink.split.i146

if.end8.sink.split.i146:                          ; preds = %if.end.i.i133, %if.then.i129
  %call.sink.i145 = phi double* [ %89, %if.end.i.i133 ], [ null, %if.then.i129 ]
  store double* %call.sink.i145, double** %m_data.i127, align 8, !tbaa !19
  br label %_ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EE6resizeElll.exit147

_ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EE6resizeElll.exit147: ; preds = %if.end8.sink.split.i146, %if.then.i.i.i.i.i.i.i133
  store i64 %83, i64* %m_rows.i98, align 8, !tbaa !15
  store i64 %84, i64* %m_cols.i99, align 8, !tbaa !18
  br label %_ZN5Eigen8internal17resize_if_allowedINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EEddEEvRT_RKT0_RKNS0_9assign_opIT1_T2_EE.exit.i.i.i.i.i.i143

_ZN5Eigen8internal17resize_if_allowedINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EEddEEvRT_RKT0_RKNS0_9assign_opIT1_T2_EE.exit.i.i.i.i.i.i143: ; preds = %_ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EE6resizeElll.exit147, %_ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EE6resizeElll.exit122
  %90 = bitcast %"struct.Eigen::internal::evaluator.15"* %dstEvaluator.i.i.i.i.i.i81 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %90) #7
  %91 = bitcast %"class.Eigen::Matrix"* %Wp to i64*
  %92 = load i64, i64* %91, align 8, !tbaa !19
  %93 = bitcast %"struct.Eigen::internal::evaluator.15"* %dstEvaluator.i.i.i.i.i.i81 to i64*
  store i64 %92, i64* %93, align 8, !tbaa !20
  %94 = load i64, i64* %m_rows.i98, align 8, !tbaa !15
  %m_value.i.i.i53 = getelementptr inbounds %"struct.Eigen::internal::evaluator.15", %"struct.Eigen::internal::evaluator.15"* %dstEvaluator.i.i.i.i.i.i81, i64 0, i32 0, i32 1, i32 0
  store i64 %94, i64* %m_value.i.i.i53, align 8, !tbaa !2
  %95 = bitcast %"class.Eigen::internal::generic_dense_assignment_kernel.76"* %kernel.i.i.i.i.i.i82 to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %95) #7
  %m_dst.i43 = getelementptr inbounds %"class.Eigen::internal::generic_dense_assignment_kernel.76", %"class.Eigen::internal::generic_dense_assignment_kernel.76"* %kernel.i.i.i.i.i.i82, i64 0, i32 0
  store %"struct.Eigen::internal::evaluator.15"* %dstEvaluator.i.i.i.i.i.i81, %"struct.Eigen::internal::evaluator.15"** %m_dst.i43, align 8, !tbaa !22
  %m_src.i44 = getelementptr inbounds %"class.Eigen::internal::generic_dense_assignment_kernel.76", %"class.Eigen::internal::generic_dense_assignment_kernel.76"* %kernel.i.i.i.i.i.i82, i64 0, i32 1
  store %"struct.Eigen::internal::evaluator.73"* %srcEvaluator.i.i.i.i.i.i80, %"struct.Eigen::internal::evaluator.73"** %m_src.i44, align 8, !tbaa !22
  %m_functor.i45 = getelementptr inbounds %"class.Eigen::internal::generic_dense_assignment_kernel.76", %"class.Eigen::internal::generic_dense_assignment_kernel.76"* %kernel.i.i.i.i.i.i82, i64 0, i32 2
  store %"struct.Eigen::internal::assign_op"* %ref.tmp.i.i.i83, %"struct.Eigen::internal::assign_op"** %m_functor.i45, align 8, !tbaa !22
  %m_dstExpr.i46 = getelementptr inbounds %"class.Eigen::internal::generic_dense_assignment_kernel.76", %"class.Eigen::internal::generic_dense_assignment_kernel.76"* %kernel.i.i.i.i.i.i82, i64 0, i32 3
  store %"class.Eigen::Matrix"* %Wp, %"class.Eigen::Matrix"** %m_dstExpr.i46, align 8, !tbaa !22
  %96 = load i64, i64* %m_rows.i98, align 8, !tbaa !15
  %97 = load i64, i64* %m_cols.i99, align 8, !tbaa !18
  %mul.i.i42 = mul nsw i64 %97, %96
  %cmp6.i.i.i.i.i.i.i142 = icmp sgt i64 %mul.i.i42, 0
  br i1 %cmp6.i.i.i.i.i.i.i142, label %for.body.i.i.i.i.i.i.i154.preheader, label %_ZN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEC2INS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES1_EEEERKNS_9EigenBaseIT_EE.exit156

for.body.i.i.i.i.i.i.i154.preheader:              ; preds = %_ZN5Eigen8internal17resize_if_allowedINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EEddEEvRT_RKT0_RKNS0_9assign_opIT1_T2_EE.exit.i.i.i.i.i.i143
  %98 = bitcast %"class.Eigen::internal::generic_dense_assignment_kernel.76"* %kernel.i.i.i.i.i.i82 to %"struct.Eigen::internal::evaluator.16"**
  %99 = load %"struct.Eigen::internal::evaluator.16"*, %"struct.Eigen::internal::evaluator.16"** %98, align 8, !tbaa !23
  %m_data.i.i.i.i.i.i.i.i.i146 = getelementptr inbounds %"struct.Eigen::internal::evaluator.16", %"struct.Eigen::internal::evaluator.16"* %99, i64 0, i32 0
  %100 = bitcast %"struct.Eigen::internal::evaluator.73"** %m_src.i44 to i64**
  %101 = load i64*, i64** %100, align 8, !tbaa !25
  br label %for.body.i.i.i.i.i.i.i154

for.body.i.i.i.i.i.i.i154:                        ; preds = %for.body.i.i.i.i.i.i.i154, %for.body.i.i.i.i.i.i.i154.preheader
  %i.07.i.i.i.i.i.i.i144 = phi i64 [ %inc.i.i.i.i.i.i.i152, %for.body.i.i.i.i.i.i.i154 ], [ 0, %for.body.i.i.i.i.i.i.i154.preheader ]
  %102 = load double*, double** %m_data.i.i.i.i.i.i.i.i.i146, align 8, !tbaa !20
  %arrayidx.i.i.i.i.i.i.i.i.i147 = getelementptr inbounds double, double* %102, i64 %i.07.i.i.i.i.i.i.i144
  %103 = load i64, i64* %101, align 8, !tbaa !12
  %104 = bitcast double* %arrayidx.i.i.i.i.i.i.i.i.i147 to i64*
  store i64 %103, i64* %104, align 8, !tbaa !26
  %inc.i.i.i.i.i.i.i152 = add nuw nsw i64 %i.07.i.i.i.i.i.i.i144, 1
  %exitcond.i.i.i.i.i.i.i153 = icmp eq i64 %inc.i.i.i.i.i.i.i152, %mul.i.i42
  br i1 %exitcond.i.i.i.i.i.i.i153, label %_ZN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEC2INS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES1_EEEERKNS_9EigenBaseIT_EE.exit156, label %for.body.i.i.i.i.i.i.i154

_ZN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEC2INS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES1_EEEERKNS_9EigenBaseIT_EE.exit156: ; preds = %for.body.i.i.i.i.i.i.i154, %_ZN5Eigen8internal17resize_if_allowedINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EEddEEvRT_RKT0_RKNS0_9assign_opIT1_T2_EE.exit.i.i.i.i.i.i143
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %95) #7
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %90) #7
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %80) #7
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %79) #7
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %71) #7
  %105 = bitcast %"class.Eigen::Matrix"* %Mp to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %105) #7
  %106 = bitcast %"class.Eigen::CwiseNullaryOp"* %ref.tmp6 to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %106) #7
  %m_value.i.i.i.i150 = getelementptr inbounds %"class.Eigen::CwiseNullaryOp", %"class.Eigen::CwiseNullaryOp"* %ref.tmp6, i64 0, i32 0, i32 0
  store i64 4, i64* %m_value.i.i.i.i150, align 8, !tbaa !2, !alias.scope !37
  %m_value.i1.i.i.i152 = getelementptr inbounds %"class.Eigen::CwiseNullaryOp", %"class.Eigen::CwiseNullaryOp"* %ref.tmp6, i64 0, i32 1, i32 0
  store i64 4, i64* %m_value.i1.i.i.i152, align 8, !tbaa !2, !alias.scope !37
  %m_functor.i.i.i153 = getelementptr inbounds %"class.Eigen::CwiseNullaryOp", %"class.Eigen::CwiseNullaryOp"* %ref.tmp6, i64 0, i32 2
  %107 = bitcast %"struct.Eigen::internal::scalar_constant_op"* %m_functor.i.i.i153 to i64*
  store i64 0, i64* %107, align 8, !tbaa !12, !alias.scope !37
  call void @llvm.memset.p0i8.i64(i8* align 8 %105, i8 0, i64 24, i1 false) #7
  %108 = load i64, i64* %m_value.i.i.i.i150, align 8, !tbaa !2
  %mul.i.i.i.i193 = shl nsw i64 %108, 2
  %m_rows.i154 = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %Mp, i64 0, i32 0, i32 0, i32 1
  %109 = load i64, i64* %m_rows.i154, align 8, !tbaa !15
  %m_cols.i155 = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %Mp, i64 0, i32 0, i32 0, i32 2
  %110 = load i64, i64* %m_cols.i155, align 8, !tbaa !18
  %mul.i156 = mul nsw i64 %110, %109
  %cmp.i157 = icmp eq i64 %mul.i156, %mul.i.i.i.i193
  br i1 %cmp.i157, label %_ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EE6resizeElll.exit178, label %if.then.i160

if.then.i160:                                     ; preds = %_ZN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEC2INS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES1_EEEERKNS_9EigenBaseIT_EE.exit156
  %m_data.i158 = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %Mp, i64 0, i32 0, i32 0, i32 0
  %111 = bitcast %"class.Eigen::Matrix"* %Mp to i8**
  %112 = load i8*, i8** %111, align 8, !tbaa !19
  call void @free(i8* %112) #7
  %tobool.i159 = icmp eq i64 %108, 0
  br i1 %tobool.i159, label %if.end8.sink.split.i177, label %if.end.i.i164

if.end.i.i164:                                    ; preds = %if.then.i160
  %mul.i.i166 = shl i64 %108, 5
  %call.i.i.i.i167 = call noalias i8* @malloc(i64 %mul.i.i166) #7
  %113 = bitcast i8* %call.i.i.i.i167 to double*
  br label %if.end8.sink.split.i177

if.end8.sink.split.i177:                          ; preds = %if.end.i.i164, %if.then.i160
  %call.sink.i176 = phi double* [ %113, %if.end.i.i164 ], [ null, %if.then.i160 ]
  store double* %call.sink.i176, double** %m_data.i158, align 8, !tbaa !19
  br label %_ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EE6resizeElll.exit178

_ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EE6resizeElll.exit178: ; preds = %if.end8.sink.split.i177, %_ZN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEC2INS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES1_EEEERKNS_9EigenBaseIT_EE.exit156
  store i64 %108, i64* %m_rows.i154, align 8, !tbaa !15
  store i64 4, i64* %m_cols.i155, align 8, !tbaa !18
  %114 = getelementptr inbounds %"struct.Eigen::internal::assign_op", %"struct.Eigen::internal::assign_op"* %ref.tmp.i.i.i161, i64 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %114) #7
  %115 = bitcast %"struct.Eigen::internal::evaluator.73"* %srcEvaluator.i.i.i.i.i.i158 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %115) #7
  %116 = load i64, i64* %107, align 8, !tbaa !12
  %117 = bitcast %"struct.Eigen::internal::evaluator.73"* %srcEvaluator.i.i.i.i.i.i158 to i64*
  store i64 %116, i64* %117, align 8, !tbaa !12
  %118 = load i64, i64* %m_value.i.i.i.i150, align 8, !tbaa !2
  %119 = load i64, i64* %m_value.i1.i.i.i152, align 8, !tbaa !2
  %120 = load i64, i64* %m_rows.i154, align 8, !tbaa !15
  %cmp.i.i.i.i.i.i.i202 = icmp eq i64 %120, %118
  %121 = load i64, i64* %m_cols.i155, align 8
  %cmp4.i.i.i.i.i.i.i206 = icmp eq i64 %121, %119
  %or.cond10 = and i1 %cmp.i.i.i.i.i.i.i202, %cmp4.i.i.i.i.i.i.i206
  br i1 %or.cond10, label %_ZN5Eigen8internal17resize_if_allowedINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EEddEEvRT_RKT0_RKNS0_9assign_opIT1_T2_EE.exit.i.i.i.i.i.i221, label %if.then.i.i.i.i.i.i.i211

if.then.i.i.i.i.i.i.i211:                         ; preds = %_ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EE6resizeElll.exit178
  %mul.i.i.i.i.i.i.i.i217 = mul nsw i64 %119, %118
  %mul.i181 = mul nsw i64 %121, %120
  %cmp.i182 = icmp eq i64 %mul.i181, %mul.i.i.i.i.i.i.i.i217
  br i1 %cmp.i182, label %_ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EE6resizeElll.exit203, label %if.then.i185

if.then.i185:                                     ; preds = %if.then.i.i.i.i.i.i.i211
  %m_data.i183 = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %Mp, i64 0, i32 0, i32 0, i32 0
  %122 = bitcast %"class.Eigen::Matrix"* %Mp to i8**
  %123 = load i8*, i8** %122, align 8, !tbaa !19
  call void @free(i8* %123) #7
  %tobool.i184 = icmp eq i64 %mul.i.i.i.i.i.i.i.i217, 0
  br i1 %tobool.i184, label %if.end8.sink.split.i202, label %if.end.i.i189

if.end.i.i189:                                    ; preds = %if.then.i185
  %mul.i.i191 = shl i64 %mul.i.i.i.i.i.i.i.i217, 3
  %call.i.i.i.i192 = call noalias i8* @malloc(i64 %mul.i.i191) #7
  %124 = bitcast i8* %call.i.i.i.i192 to double*
  br label %if.end8.sink.split.i202

if.end8.sink.split.i202:                          ; preds = %if.end.i.i189, %if.then.i185
  %call.sink.i201 = phi double* [ %124, %if.end.i.i189 ], [ null, %if.then.i185 ]
  store double* %call.sink.i201, double** %m_data.i183, align 8, !tbaa !19
  br label %_ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EE6resizeElll.exit203

_ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EE6resizeElll.exit203: ; preds = %if.end8.sink.split.i202, %if.then.i.i.i.i.i.i.i211
  store i64 %118, i64* %m_rows.i154, align 8, !tbaa !15
  store i64 %119, i64* %m_cols.i155, align 8, !tbaa !18
  br label %_ZN5Eigen8internal17resize_if_allowedINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EEddEEvRT_RKT0_RKNS0_9assign_opIT1_T2_EE.exit.i.i.i.i.i.i221

_ZN5Eigen8internal17resize_if_allowedINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EEddEEvRT_RKT0_RKNS0_9assign_opIT1_T2_EE.exit.i.i.i.i.i.i221: ; preds = %_ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EE6resizeElll.exit203, %_ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EE6resizeElll.exit178
  %125 = bitcast %"struct.Eigen::internal::evaluator.15"* %dstEvaluator.i.i.i.i.i.i159 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %125) #7
  %126 = bitcast %"class.Eigen::Matrix"* %Mp to i64*
  %127 = load i64, i64* %126, align 8, !tbaa !19
  %128 = bitcast %"struct.Eigen::internal::evaluator.15"* %dstEvaluator.i.i.i.i.i.i159 to i64*
  store i64 %127, i64* %128, align 8, !tbaa !20
  %129 = load i64, i64* %m_rows.i154, align 8, !tbaa !15
  %m_value.i.i.i32 = getelementptr inbounds %"struct.Eigen::internal::evaluator.15", %"struct.Eigen::internal::evaluator.15"* %dstEvaluator.i.i.i.i.i.i159, i64 0, i32 0, i32 1, i32 0
  store i64 %129, i64* %m_value.i.i.i32, align 8, !tbaa !2
  %130 = bitcast %"class.Eigen::internal::generic_dense_assignment_kernel.76"* %kernel.i.i.i.i.i.i160 to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %130) #7
  %m_dst.i = getelementptr inbounds %"class.Eigen::internal::generic_dense_assignment_kernel.76", %"class.Eigen::internal::generic_dense_assignment_kernel.76"* %kernel.i.i.i.i.i.i160, i64 0, i32 0
  store %"struct.Eigen::internal::evaluator.15"* %dstEvaluator.i.i.i.i.i.i159, %"struct.Eigen::internal::evaluator.15"** %m_dst.i, align 8, !tbaa !22
  %m_src.i = getelementptr inbounds %"class.Eigen::internal::generic_dense_assignment_kernel.76", %"class.Eigen::internal::generic_dense_assignment_kernel.76"* %kernel.i.i.i.i.i.i160, i64 0, i32 1
  store %"struct.Eigen::internal::evaluator.73"* %srcEvaluator.i.i.i.i.i.i158, %"struct.Eigen::internal::evaluator.73"** %m_src.i, align 8, !tbaa !22
  %m_functor.i = getelementptr inbounds %"class.Eigen::internal::generic_dense_assignment_kernel.76", %"class.Eigen::internal::generic_dense_assignment_kernel.76"* %kernel.i.i.i.i.i.i160, i64 0, i32 2
  store %"struct.Eigen::internal::assign_op"* %ref.tmp.i.i.i161, %"struct.Eigen::internal::assign_op"** %m_functor.i, align 8, !tbaa !22
  %m_dstExpr.i25 = getelementptr inbounds %"class.Eigen::internal::generic_dense_assignment_kernel.76", %"class.Eigen::internal::generic_dense_assignment_kernel.76"* %kernel.i.i.i.i.i.i160, i64 0, i32 3
  store %"class.Eigen::Matrix"* %Mp, %"class.Eigen::Matrix"** %m_dstExpr.i25, align 8, !tbaa !22
  %131 = load i64, i64* %m_rows.i154, align 8, !tbaa !15
  %132 = load i64, i64* %m_cols.i155, align 8, !tbaa !18
  %mul.i.i = mul nsw i64 %132, %131
  %cmp6.i.i.i.i.i.i.i220 = icmp sgt i64 %mul.i.i, 0
  br i1 %cmp6.i.i.i.i.i.i.i220, label %for.body.i.i.i.i.i.i.i232.preheader, label %_ZN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEC2INS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES1_EEEERKNS_9EigenBaseIT_EE.exit234

for.body.i.i.i.i.i.i.i232.preheader:              ; preds = %_ZN5Eigen8internal17resize_if_allowedINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EEddEEvRT_RKT0_RKNS0_9assign_opIT1_T2_EE.exit.i.i.i.i.i.i221
  %133 = bitcast %"class.Eigen::internal::generic_dense_assignment_kernel.76"* %kernel.i.i.i.i.i.i160 to %"struct.Eigen::internal::evaluator.16"**
  %134 = load %"struct.Eigen::internal::evaluator.16"*, %"struct.Eigen::internal::evaluator.16"** %133, align 8, !tbaa !23
  %m_data.i.i.i.i.i.i.i.i.i224 = getelementptr inbounds %"struct.Eigen::internal::evaluator.16", %"struct.Eigen::internal::evaluator.16"* %134, i64 0, i32 0
  %135 = bitcast %"struct.Eigen::internal::evaluator.73"** %m_src.i to i64**
  %136 = load i64*, i64** %135, align 8, !tbaa !25
  br label %for.body.i.i.i.i.i.i.i232

for.body.i.i.i.i.i.i.i232:                        ; preds = %for.body.i.i.i.i.i.i.i232, %for.body.i.i.i.i.i.i.i232.preheader
  %i.07.i.i.i.i.i.i.i222 = phi i64 [ %inc.i.i.i.i.i.i.i230, %for.body.i.i.i.i.i.i.i232 ], [ 0, %for.body.i.i.i.i.i.i.i232.preheader ]
  %137 = load double*, double** %m_data.i.i.i.i.i.i.i.i.i224, align 8, !tbaa !20
  %arrayidx.i.i.i.i.i.i.i.i.i225 = getelementptr inbounds double, double* %137, i64 %i.07.i.i.i.i.i.i.i222
  %138 = load i64, i64* %136, align 8, !tbaa !12
  %139 = bitcast double* %arrayidx.i.i.i.i.i.i.i.i.i225 to i64*
  store i64 %138, i64* %139, align 8, !tbaa !26
  %inc.i.i.i.i.i.i.i230 = add nuw nsw i64 %i.07.i.i.i.i.i.i.i222, 1
  %exitcond.i.i.i.i.i.i.i231 = icmp eq i64 %inc.i.i.i.i.i.i.i230, %mul.i.i
  br i1 %exitcond.i.i.i.i.i.i.i231, label %_ZN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEC2INS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES1_EEEERKNS_9EigenBaseIT_EE.exit234, label %for.body.i.i.i.i.i.i.i232

_ZN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEC2INS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES1_EEEERKNS_9EigenBaseIT_EE.exit234: ; preds = %for.body.i.i.i.i.i.i.i232, %_ZN5Eigen8internal17resize_if_allowedINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EEddEEvRT_RKT0_RKNS0_9assign_opIT1_T2_EE.exit.i.i.i.i.i.i221
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %130) #7
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %125) #7
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %115) #7
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %114) #7
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %106) #7
  %m_data.i.i.i.i.i.i.i.i.i.i = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %W, i64 0, i32 0, i32 0, i32 0
  %m_data.i1.i.i.i.i.i.i.i.i.i = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %M, i64 0, i32 0, i32 0, i32 0
  %ptrW = load double*, double** %m_data.i.i.i.i.i.i.i.i.i.i, align 8, !tbaa !20
  %ptrM = load double*, double** %m_data.i1.i.i.i.i.i.i.i.i.i, align 8, !tbaa !20
  %dm_data.i.i.i.i.i.i.i.i.i.i = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %Wp, i64 0, i32 0, i32 0, i32 0
  %dm_data.i1.i.i.i.i.i.i.i.i.i = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %Mp, i64 0, i32 0, i32 0, i32 0
  %dptrW = load double*, double** %dm_data.i.i.i.i.i.i.i.i.i.i, align 8, !tbaa !20
  %dptrM = load double*, double** %dm_data.i1.i.i.i.i.i.i.i.i.i, align 8, !tbaa !20
  %call = call double @__enzyme_autodiff(i8* bitcast (double (double*, double*)* @_ZL6matvecPKN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEES3_ to i8*), double* nonnull %ptrW, double* nonnull %dptrW, double* nonnull %ptrM, double* nonnull %dptrM) #7
  %140 = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %Wp, i64 0, i32 0, i32 0, i32 0
  br label %for.cond8.preheader

for.cond8.preheader:                              ; preds = %for.cond.cleanup11, %_ZN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEC2INS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES1_EEEERKNS_9EigenBaseIT_EE.exit234
  %indvars.iv103 = phi i64 [ 0, %_ZN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEC2INS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES1_EEEERKNS_9EigenBaseIT_EE.exit234 ], [ %indvars.iv.next104, %for.cond.cleanup11 ]
  %141 = trunc i64 %indvars.iv103 to i32
  br label %for.body12

for.cond29.preheader:                             ; preds = %for.cond.cleanup11
  %142 = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %Mp, i64 0, i32 0, i32 0, i32 0
  br label %for.cond35.preheader

for.cond.cleanup11:                               ; preds = %if.end
  %indvars.iv.next104 = add nuw nsw i64 %indvars.iv103, 1
  %cmp = icmp ult i64 %indvars.iv.next104, 4
  br i1 %cmp, label %for.cond8.preheader, label %for.cond29.preheader

for.body12:                                       ; preds = %if.end, %for.cond8.preheader
  %indvars.iv101 = phi i64 [ 0, %for.cond8.preheader ], [ %indvars.iv.next102, %if.end ]
  %143 = load double*, double** %140, align 8, !tbaa !19
  %144 = load i64, i64* %m_rows.i98, align 8, !tbaa !15
  %mul.i.i.i = mul nsw i64 %144, %indvars.iv103
  %add.i.i.i = add nsw i64 %mul.i.i.i, %indvars.iv101
  %arrayidx.i.i.i = getelementptr inbounds double, double* %143, i64 %add.i.i.i
  %145 = load double, double* %arrayidx.i.i.i, align 8, !tbaa !26
  %sub = fadd double %145, 8.000000e+00
  %146 = call double @llvm.fabs.f64(double %sub)
  %cmp16 = fcmp ogt double %146, 1.000000e-10
  %147 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !22
  br i1 %cmp16, label %if.then, label %if.end

if.then:                                          ; preds = %for.body12
  %call20 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %147, i8* getelementptr inbounds ([68 x i8], [68 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str.1, i64 0, i64 0), double %145, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.2, i64 0, i64 0), double -8.000000e+00, double 1.000000e-10, i8* getelementptr inbounds ([63 x i8], [63 x i8]* @.str.3, i64 0, i64 0), i32 61, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @__PRETTY_FUNCTION__.main, i64 0, i64 0)) #9
  call void @abort() #10
  unreachable

if.end:                                           ; preds = %for.body12
  %148 = trunc i64 %indvars.iv101 to i32
  %call24 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %147, i8* getelementptr inbounds ([19 x i8], [19 x i8]* @.str.4, i64 0, i64 0), i32 %148, i32 %141, double %145) #9
  %indvars.iv.next102 = add nuw nsw i64 %indvars.iv101, 1
  %cmp10 = icmp ult i64 %indvars.iv.next102, 4
  br i1 %cmp10, label %for.body12, label %for.cond.cleanup11

for.cond35.preheader:                             ; preds = %for.cond.cleanup38, %for.cond29.preheader
  %indvars.iv99 = phi i64 [ 0, %for.cond29.preheader ], [ %indvars.iv.next100, %for.cond.cleanup38 ]
  %149 = trunc i64 %indvars.iv99 to i32
  br label %for.body39

for.cond.cleanup32:                               ; preds = %for.cond.cleanup38
  %150 = bitcast %"class.Eigen::Matrix"* %Mp to i8**
  %151 = load i8*, i8** %150, align 8, !tbaa !19
  call void @free(i8* %151) #7
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %105) #7
  %152 = bitcast %"class.Eigen::Matrix"* %Wp to i8**
  %153 = load i8*, i8** %152, align 8, !tbaa !19
  call void @free(i8* %153) #7
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %70) #7
  %154 = bitcast %"class.Eigen::Matrix"* %M to i8**
  %155 = load i8*, i8** %154, align 8, !tbaa !19
  call void @free(i8* %155) #7
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %35) #7
  %156 = bitcast %"class.Eigen::Matrix"* %W to i8**
  %157 = load i8*, i8** %156, align 8, !tbaa !19
  call void @free(i8* %157) #7
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %0) #7
  ret i32 0

for.cond.cleanup38:                               ; preds = %if.end50
  %indvars.iv.next100 = add nuw nsw i64 %indvars.iv99, 1
  %cmp31 = icmp ult i64 %indvars.iv.next100, 4
  br i1 %cmp31, label %for.cond35.preheader, label %for.cond.cleanup32

for.body39:                                       ; preds = %if.end50, %for.cond35.preheader
  %indvars.iv = phi i64 [ 0, %for.cond35.preheader ], [ %indvars.iv.next, %if.end50 ]
  %158 = load double*, double** %142, align 8, !tbaa !19
  %159 = load i64, i64* %m_rows.i154, align 8, !tbaa !15
  %mul.i.i.i251 = mul nsw i64 %159, %indvars.iv99
  %add.i.i.i252 = add nsw i64 %mul.i.i.i251, %indvars.iv
  %arrayidx.i.i.i253 = getelementptr inbounds double, double* %158, i64 %add.i.i.i252
  %160 = load double, double* %arrayidx.i.i.i253, align 8, !tbaa !26
  %sub43 = fadd double %160, -8.000000e+00
  %161 = call double @llvm.fabs.f64(double %sub43)
  %cmp44 = fcmp ogt double %161, 1.000000e-10
  %162 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !22
  br i1 %cmp44, label %if.then45, label %if.end50

if.then45:                                        ; preds = %for.body39
  %call49 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %162, i8* getelementptr inbounds ([68 x i8], [68 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str.5, i64 0, i64 0), double %160, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.6, i64 0, i64 0), double 8.000000e+00, double 1.000000e-10, i8* getelementptr inbounds ([63 x i8], [63 x i8]* @.str.3, i64 0, i64 0), i32 67, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @__PRETTY_FUNCTION__.main, i64 0, i64 0)) #9
  call void @abort() #10
  unreachable

if.end50:                                         ; preds = %for.body39
  %163 = trunc i64 %indvars.iv to i32
  %call54 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %162, i8* getelementptr inbounds ([19 x i8], [19 x i8]* @.str.7, i64 0, i64 0), i32 %163, i32 %149, double %160) #9
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %cmp37 = icmp ult i64 %indvars.iv.next, 4
  br i1 %cmp37, label %for.body39, label %for.cond.cleanup38
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

declare dso_local double @__enzyme_autodiff(i8*, double*, double*, double*, double*)

; Function Attrs: alwaysinline uwtable
define internal double @_ZL6matvecPKN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEES3_(double* noalias %ptrW, double* noalias %ptrM) #3 {
entry:
  %transp = alloca i64
  %call.i.i.i.i = call noalias i8* @malloc(i64 128) #7
  %doubles = bitcast i8* %call.i.i.i.i to double*
  br label %dfor

dfor:                           ; preds = %for.body.i.i.i.i.i.i.i, %entry
  %i.07.i.i.i.i.i.i.i = phi i64 [ %inc.i.i.i.i.i.i.i, %dfor ], [ 0, %entry ]
  %arrayidx.i.i.i.i.i.i.i.i.i = getelementptr inbounds double, double* %doubles, i64 %i.07.i.i.i.i.i.i.i
  %Wi = getelementptr inbounds double, double* %ptrW, i64 %i.07.i.i.i.i.i.i.i
  %Mi = getelementptr inbounds double, double* %ptrM, i64 %i.07.i.i.i.i.i.i.i
  %lwi = load double, double* %Wi, align 8, !tbaa !26
  %lmi = load double, double* %Mi, align 8, !tbaa !26
  %sub.i.i.i.i.i.i.i.i.i.i = fsub double %lwi, %lmi
  store double %sub.i.i.i.i.i.i.i.i.i.i, double* %arrayidx.i.i.i.i.i.i.i.i.i, align 8, !tbaa !26
  %inc.i.i.i.i.i.i.i = add nuw nsw i64 %i.07.i.i.i.i.i.i.i, 1
  %exitcond.i.i.i.i.i.i.i = icmp eq i64 %inc.i.i.i.i.i.i.i, 16
  br i1 %exitcond.i.i.i.i.i.i.i, label %mid, label %dfor

mid: ; preds = %for.body.i.i.i.i.i.i.i
  br label %for.cond1.preheader.i.i.i.i.i.i.i.i.i.i

for.cond1.preheader.i.i.i.i.i.i.i.i.i.i:          ; preds = %for.cond.cleanup4.i.i.i.i.i.i.i.i.i.i, %_ZN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEC2INS_13CwiseBinaryOpINS_8internal20scalar_difference_opIddEEKS1_S7_EEEERKNS_9EigenBaseIT_EE.exit
  %total.0 = phi double [ 0.000000e+00, %mid ], [ %addtotal, %for.cond.cleanup4.i.i.i.i.i.i.i.i.i.i ]
  %outer.022.i.i.i.i.i.i.i.i.i.i = phi i64 [ %inc7.i.i.i.i.i.i.i.i.i.i, %for.cond.cleanup4.i.i.i.i.i.i.i.i.i.i ], [ 0, %mid ]
  br label %for.body5.i.i.i.i.i.i.i.i.i.i

for.body5.i.i.i.i.i.i.i.i.i.i:                    ; preds = %_ZNK5Eigen9DenseBaseINS_13CwiseBinaryOpINS_8internal17scalar_product_opIddEEKNS_9TransposeIKNS_5BlockIKNS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1ELin1ELb0EEEEEKNS6_IS9_Lin1ELi1ELb1EEEEEE5reduxINS2_13scalar_sum_opIddEEEEdRKT_.exit.i.i.i.i.i.i, %for.cond1.preheader.i.i.i.i.i.i.i.i.i.i
  %total.1 = phi double [ %total.0, %for.cond1.preheader.i.i.i.i.i.i.i.i.i.i ], [ %addtotal, %_ZNK5Eigen9DenseBaseINS_13CwiseBinaryOpINS_8internal17scalar_product_opIddEEKNS_9TransposeIKNS_5BlockIKNS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1ELin1ELb0EEEEEKNS6_IS9_Lin1ELi1ELb1EEEEEE5reduxINS2_13scalar_sum_opIddEEEEdRKT_.exit.i.i.i.i.i.i ]
  %inner.019.i.i.i.i.i.i.i.i.i.i = phi i64 [ %inc.i.i.i.i.i.i.i.i.i.i, %_ZNK5Eigen9DenseBaseINS_13CwiseBinaryOpINS_8internal17scalar_product_opIddEEKNS_9TransposeIKNS_5BlockIKNS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1ELin1ELb0EEEEEKNS6_IS9_Lin1ELi1ELb1EEEEEE5reduxINS2_13scalar_sum_opIddEEEEdRKT_.exit.i.i.i.i.i.i ], [ 0, %for.cond1.preheader.i.i.i.i.i.i.i.i.i.i ]
  %add.ptr = getelementptr inbounds double, double* %doubles, i64 %inner.019.i.i.i.i.i.i.i.i.i.i
  %.cast = ptrtoint double* %add.ptr to i64
  store i64 %.cast, i64* %transp, align 8, !tbaa !62, !alias.scope !65
  %tdoub = load i64, i64* %transp, align 8, !tbaa !62
  %.cast1 = inttoptr i64 %tdoub to double*
  br label %for.body.i.i.i.i.i.i.i.i

for.body.i.i.i.i.i.i.i.i:                         ; preds = %for.body.i.i.i.i.i.i.i.i, %for.body5.i.i.i.i.i.i.i.i.i.i
  %res.i.i.i.i.i.i.i.i.0 = phi double [ 0.000000e+00, %for.body5.i.i.i.i.i.i.i.i.i.i ], [ %add.i.i.i.i.i.i.i.i.i, %for.body.i.i.i.i.i.i.i.i ]
  %i.047.i.i.i.i.i.i.i.i = phi i64 [ 0, %for.body5.i.i.i.i.i.i.i.i.i.i ], [ %inc.i.i.i.i.i.i.i.i, %for.body.i.i.i.i.i.i.i.i ]
  %mul.i.i.i.i20.i.i.i.i.i.i.i = shl nsw i64 %i.047.i.i.i.i.i.i.i.i, 2
  %valptr = getelementptr inbounds double, double* %.cast1, i64 %mul.i.i.i.i20.i.i.i.i.i.i.i
  %val = load double, double* %valptr, align 8, !tbaa !26
  %mul26 = fmul double %val, %val
  %add.i.i.i.i.i.i.i.i.i = fadd double %res.i.i.i.i.i.i.i.i.0, %mul26
  %inc.i.i.i.i.i.i.i.i = add nuw nsw i64 %i.047.i.i.i.i.i.i.i.i, 1
  %cmp.i.i.i15.i.i.i.i.i = icmp slt i64 %inc.i.i.i.i.i.i.i.i, 4
  br i1 %cmp.i.i.i15.i.i.i.i.i, label %for.body.i.i.i.i.i.i.i.i, label %_ZNK5Eigen9DenseBaseINS_13CwiseBinaryOpINS_8internal17scalar_product_opIddEEKNS_9TransposeIKNS_5BlockIKNS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1ELin1ELb0EEEEEKNS6_IS9_Lin1ELi1ELb1EEEEEE5reduxINS2_13scalar_sum_opIddEEEEdRKT_.exit.i.i.i.i.i.i

_ZNK5Eigen9DenseBaseINS_13CwiseBinaryOpINS_8internal17scalar_product_opIddEEKNS_9TransposeIKNS_5BlockIKNS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1ELin1ELb0EEEEEKNS6_IS9_Lin1ELi1ELb1EEEEEE5reduxINS2_13scalar_sum_opIddEEEEdRKT_.exit.i.i.i.i.i.i: ; preds = %for.body.i.i.i.i.i.i.i.i
  %addtotal = fadd double %total.1, %add.i.i.i.i.i.i.i.i.i
  %inc.i.i.i.i.i.i.i.i.i.i = add nuw nsw i64 %inner.019.i.i.i.i.i.i.i.i.i.i, 1
  %cmp3.i.i.i.i.i.i.i.i.i.i = icmp ult i64 %inc.i.i.i.i.i.i.i.i.i.i, 4
  br i1 %cmp3.i.i.i.i.i.i.i.i.i.i, label %for.body5.i.i.i.i.i.i.i.i.i.i, label %for.cond.cleanup4.i.i.i.i.i.i.i.i.i.i

for.cond.cleanup4.i.i.i.i.i.i.i.i.i.i:            ; preds = %_ZNK5Eigen9DenseBaseINS_13CwiseBinaryOpINS_8internal17scalar_product_opIddEEKNS_9TransposeIKNS_5BlockIKNS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1ELin1ELb0EEEEEKNS6_IS9_Lin1ELi1ELb1EEEEEE5reduxINS2_13scalar_sum_opIddEEEEdRKT_.exit.i.i.i.i.i.i
  %inc7.i.i.i.i.i.i.i.i.i.i = add nuw nsw i64 %outer.022.i.i.i.i.i.i.i.i.i.i, 1
  %cmp.i1.i.i.i.i.i.i.i.i.i = icmp ult i64 %inc7.i.i.i.i.i.i.i.i.i.i, 4
  br i1 %cmp.i1.i.i.i.i.i.i.i.i.i, label %for.cond1.preheader.i.i.i.i.i.i.i.i.i.i, label %exit

exit: ; preds = %for.cond.cleanup4.i.i.i.i.i.i.i.i.i.i
  ret double %addtotal
}

; Function Attrs: nounwind readnone speculatable
declare double @llvm.fabs.f64(double) #4

; Function Attrs: nounwind
declare dso_local i32 @fprintf(%struct._IO_FILE* nocapture, i8* nocapture readonly, ...) local_unnamed_addr #5

; Function Attrs: noreturn nounwind
declare dso_local void @abort() local_unnamed_addr #6

; Function Attrs: nounwind
declare dso_local void @free(i8* nocapture) local_unnamed_addr #5

; Function Attrs: nounwind
declare dso_local noalias i8* @malloc(i64) local_unnamed_addr #5

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1) #1

; Function Attrs: nounwind
declare dso_local i32 @__cxa_guard_acquire(i64*) local_unnamed_addr #7

; Function Attrs: nounwind
declare dso_local void @__cxa_guard_release(i64*) local_unnamed_addr #7

; Function Attrs: inaccessiblemem_or_argmemonly nounwind
declare void @llvm.prefetch(i8* nocapture readonly, i32, i32, i32) #8

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1) #1

; Function Attrs: nounwind
declare i8* @llvm.stacksave() #7

; Function Attrs: nounwind
declare void @llvm.stackrestore(i8*) #7

attributes #0 = { alwaysinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { alwaysinline uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind readnone speculatable }
attributes #5 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { noreturn nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #7 = { nounwind }
attributes #8 = { inaccessiblemem_or_argmemonly nounwind }
attributes #9 = { cold }
attributes #10 = { noreturn nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !4, i64 0}
!3 = !{!"_ZTSN5Eigen8internal19variable_if_dynamicIlLin1EEE", !4, i64 0}
!4 = !{!"long", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}
!7 = !{!8, !10}
!8 = distinct !{!8, !9, !"_ZN5Eigen9DenseBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEE11NullaryExprINS_8internal18scalar_constant_opIdEEEEKNS_14CwiseNullaryOpIT_S2_EEllRKS9_: %agg.result"}
!9 = distinct !{!9, !"_ZN5Eigen9DenseBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEE11NullaryExprINS_8internal18scalar_constant_opIdEEEEKNS_14CwiseNullaryOpIT_S2_EEllRKS9_"}
!10 = distinct !{!10, !11, !"_ZN5Eigen9DenseBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEE8ConstantEllRKd: %agg.result"}
!11 = distinct !{!11, !"_ZN5Eigen9DenseBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEE8ConstantEllRKd"}
!12 = !{!13, !14, i64 0}
!13 = !{!"_ZTSN5Eigen8internal18scalar_constant_opIdEE", !14, i64 0}
!14 = !{!"double", !5, i64 0}
!15 = !{!16, !4, i64 8}
!16 = !{!"_ZTSN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EEE", !17, i64 0, !4, i64 8, !4, i64 16}
!17 = !{!"any pointer", !5, i64 0}
!18 = !{!16, !4, i64 16}
!19 = !{!16, !17, i64 0}
!20 = !{!21, !17, i64 0}
!21 = !{!"_ZTSN5Eigen8internal9evaluatorINS_15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEEEEE", !17, i64 0, !3, i64 8}
!22 = !{!17, !17, i64 0}
!23 = !{!24, !17, i64 0}
!24 = !{!"_ZTSN5Eigen8internal31generic_dense_assignment_kernelINS0_9evaluatorINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEEENS2_INS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES4_EEEENS0_9assign_opIddEELi0EEE", !17, i64 0, !17, i64 8, !17, i64 16, !17, i64 24}
!25 = !{!24, !17, i64 8}
!26 = !{!14, !14, i64 0}
!27 = !{!28, !30}
!28 = distinct !{!28, !29, !"_ZN5Eigen9DenseBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEE11NullaryExprINS_8internal18scalar_constant_opIdEEEEKNS_14CwiseNullaryOpIT_S2_EEllRKS9_: %agg.result"}
!29 = distinct !{!29, !"_ZN5Eigen9DenseBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEE11NullaryExprINS_8internal18scalar_constant_opIdEEEEKNS_14CwiseNullaryOpIT_S2_EEllRKS9_"}
!30 = distinct !{!30, !31, !"_ZN5Eigen9DenseBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEE8ConstantEllRKd: %agg.result"}
!31 = distinct !{!31, !"_ZN5Eigen9DenseBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEE8ConstantEllRKd"}
!32 = !{!33, !35}
!33 = distinct !{!33, !34, !"_ZN5Eigen9DenseBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEE11NullaryExprINS_8internal18scalar_constant_opIdEEEEKNS_14CwiseNullaryOpIT_S2_EEllRKS9_: %agg.result"}
!34 = distinct !{!34, !"_ZN5Eigen9DenseBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEE11NullaryExprINS_8internal18scalar_constant_opIdEEEEKNS_14CwiseNullaryOpIT_S2_EEllRKS9_"}
!35 = distinct !{!35, !36, !"_ZN5Eigen9DenseBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEE8ConstantEllRKd: %agg.result"}
!36 = distinct !{!36, !"_ZN5Eigen9DenseBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEE8ConstantEllRKd"}
!37 = !{!38, !40}
!38 = distinct !{!38, !39, !"_ZN5Eigen9DenseBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEE11NullaryExprINS_8internal18scalar_constant_opIdEEEEKNS_14CwiseNullaryOpIT_S2_EEllRKS9_: %agg.result"}
!39 = distinct !{!39, !"_ZN5Eigen9DenseBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEE11NullaryExprINS_8internal18scalar_constant_opIdEEEEKNS_14CwiseNullaryOpIT_S2_EEllRKS9_"}
!40 = distinct !{!40, !41, !"_ZN5Eigen9DenseBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEE8ConstantEllRKd: %agg.result"}
!41 = distinct !{!41, !"_ZN5Eigen9DenseBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEE8ConstantEllRKd"}
!42 = !{!43}
!43 = distinct !{!43, !44, !"_ZNK5Eigen10MatrixBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEEmiIS2_EEKNS_13CwiseBinaryOpINS_8internal20scalar_difference_opIdNS6_6traitsIT_E6ScalarEEEKS2_KS9_EERKNS0_IS9_EE: %agg.result"}
!44 = distinct !{!44, !"_ZNK5Eigen10MatrixBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEEmiIS2_EEKNS_13CwiseBinaryOpINS_8internal20scalar_difference_opIdNS6_6traitsIT_E6ScalarEEEKS2_KS9_EERKNS0_IS9_EE"}
!45 = !{!46, !17, i64 0}
!46 = !{!"_ZTSN5Eigen13CwiseBinaryOpINS_8internal20scalar_difference_opIddEEKNS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEES6_EE", !17, i64 0, !17, i64 8, !47, i64 16}
!47 = !{!"_ZTSN5Eigen8internal20scalar_difference_opIddEE"}
!48 = !{!46, !17, i64 8}
!49 = !{!50, !17, i64 8}
!50 = !{!"_ZTSN5Eigen8internal31generic_dense_assignment_kernelINS0_9evaluatorINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEEENS2_INS_13CwiseBinaryOpINS0_20scalar_difference_opIddEEKS4_S9_EEEENS0_9assign_opIddEELi0EEE", !17, i64 0, !17, i64 8, !17, i64 16, !17, i64 24}
!51 = !{!52, !17, i64 0}
!52 = !{!"_ZTSN5Eigen7ProductINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEES2_Li0EEE", !17, i64 0, !17, i64 8}
!53 = !{!52, !17, i64 8}
!54 = !{!55, !17, i64 0}
!55 = !{!"_ZTSN5Eigen7ProductINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEES2_Li1EEE", !17, i64 0, !17, i64 8}
!56 = !{!55, !17, i64 8}
!57 = !{!58, !17, i64 8}
!58 = !{!"_ZTSN5Eigen8internal17product_evaluatorINS_7ProductINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEES4_Li1EEELi8ENS_10DenseShapeES6_ddEE", !17, i64 0, !17, i64 8, !59, i64 16, !59, i64 32, !4, i64 48}
!59 = !{!"_ZTSN5Eigen8internal9evaluatorINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEEE"}
!60 = !{!58, !4, i64 48}
!61 = !{!58, !17, i64 0}
!62 = !{!63, !17, i64 0}
!63 = !{!"_ZTSN5Eigen7MapBaseINS_5BlockIKNS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1ELin1ELb0EEELi0EEE", !17, i64 0, !64, i64 8, !3, i64 16}
!64 = !{!"_ZTSN5Eigen8internal19variable_if_dynamicIlLi1EEE"}
!65 = !{!66}
!66 = distinct !{!66, !67, !"_ZNK5Eigen9DenseBaseINS_5BlockIKNS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1ELin1ELb0EEEE9transposeEv: %agg.result"}
!67 = distinct !{!67, !"_ZNK5Eigen9DenseBaseINS_5BlockIKNS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1ELin1ELb0EEEE9transposeEv"}

; CHECK: define internal {} @diffe_ZL6matvecPKN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEES3_(double* noalias %ptrW, double* %"ptrW'", double* noalias %ptrM, double* %"ptrM'", double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %call.i.i.i.i = call noalias i8* @malloc(i64 128) #6
; CHECK-NEXT:   %"call.i.i.i.i'mi" = call noalias nonnull i8* @malloc(i64 128) #6
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull align 1 %"call.i.i.i.i'mi", i8 0, i64 128, i1 false)
; CHECK-NEXT:   %"doubles'ipc" = bitcast i8* %"call.i.i.i.i'mi" to double*
; CHECK-NEXT:   %doubles = bitcast i8* %call.i.i.i.i to double*
; CHECK-NEXT:   br label %dfor

; CHECK: dfor:                                             ; preds = %dfor, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %dfor ], [ 0, %entry ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %arrayidx.i.i.i.i.i.i.i.i.i = getelementptr inbounds double, double* %doubles, i64 %iv
; CHECK-NEXT:   %Wi = getelementptr inbounds double, double* %ptrW, i64 %iv
; CHECK-NEXT:   %Mi = getelementptr inbounds double, double* %ptrM, i64 %iv
; CHECK-NEXT:   %lwi = load double, double* %Wi, align 8, !tbaa !26
; CHECK-NEXT:   %lmi = load double, double* %Mi, align 8, !tbaa !26
; CHECK-NEXT:   %sub.i.i.i.i.i.i.i.i.i.i = fsub double %lwi, %lmi
; CHECK-NEXT:   store double %sub.i.i.i.i.i.i.i.i.i.i, double* %arrayidx.i.i.i.i.i.i.i.i.i, align 8, !tbaa !26
; CHECK-NEXT:   %exitcond.i.i.i.i.i.i.i = icmp eq i64 %iv.next, 16
; CHECK-NEXT:   br i1 %exitcond.i.i.i.i.i.i.i, label %mid, label %dfor

; CHECK: mid:                                              ; preds = %dfor
; CHECK-NEXT:   %malloccall = tail call noalias nonnull i8* @malloc(i64 512)
; CHECK-NEXT:   %val_malloccache = bitcast i8* %malloccall to double*
; CHECK-NEXT:   %malloccall7 = tail call noalias nonnull i8* @malloc(i64 128)
; CHECK-NEXT:   %"tdoub'ipl_malloccache" = bitcast i8* %malloccall7 to i64*
; CHECK-NEXT:   br label %for.cond1.preheader.i.i.i.i.i.i.i.i.i.i

; CHECK: for.cond1.preheader.i.i.i.i.i.i.i.i.i.i:          ; preds = %for.cond.cleanup4.i.i.i.i.i.i.i.i.i.i, %mid
; CHECK-NEXT:   %iv1 = phi i64 [ %iv.next2, %for.cond.cleanup4.i.i.i.i.i.i.i.i.i.i ], [ 0, %mid ]
; CHECK-NEXT:   %iv.next2 = add nuw nsw i64 %iv1, 1
; CHECK-NEXT:   br label %for.body5.i.i.i.i.i.i.i.i.i.i

; CHECK: for.body5.i.i.i.i.i.i.i.i.i.i:                    ; preds = %_ZNK5Eigen9DenseBaseINS_13CwiseBinaryOpINS_8internal17scalar_product_opIddEEKNS_9TransposeIKNS_5BlockIKNS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1ELin1ELb0EEEEEKNS6_IS9_Lin1ELi1ELb1EEEEEE5reduxINS2_13scalar_sum_opIddEEEEdRKT_.exit.i.i.i.i.i.i, %for.cond1.preheader.i.i.i.i.i.i.i.i.i.i
; CHECK-NEXT:   %iv3 = phi i64 [ %iv.next4, %_ZNK5Eigen9DenseBaseINS_13CwiseBinaryOpINS_8internal17scalar_product_opIddEEKNS_9TransposeIKNS_5BlockIKNS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1ELin1ELb0EEEEEKNS6_IS9_Lin1ELi1ELb1EEEEEE5reduxINS2_13scalar_sum_opIddEEEEdRKT_.exit.i.i.i.i.i.i ], [ 0, %for.cond1.preheader.i.i.i.i.i.i.i.i.i.i ]
; CHECK-NEXT:   %iv.next4 = add nuw nsw i64 %iv3, 1
; CHECK-NEXT:   %"add.ptr'ipg" = getelementptr inbounds double, double* %"doubles'ipc", i64 %iv3
; CHECK-NEXT:   %add.ptr = getelementptr inbounds double, double* %doubles, i64 %iv3
; CHECK-NEXT:   %".cast'ipc" = ptrtoint double* %"add.ptr'ipg" to i64
; CHECK-NEXT:   %0 = shl nuw nsw i64 %iv3, 2
; CHECK-NEXT:   %1 = add nuw nsw i64 %iv1, %0
; CHECK-NEXT:   %2 = getelementptr inbounds i64, i64* %"tdoub'ipl_malloccache", i64 %1
; CHECK-NEXT:   store i64 %".cast'ipc", i64* %2, align 8, !invariant.group !42
; CHECK-NEXT:   br label %for.body.i.i.i.i.i.i.i.i

; CHECK: for.body.i.i.i.i.i.i.i.i:                         ; preds = %for.body.i.i.i.i.i.i.i.i, %for.body5.i.i.i.i.i.i.i.i.i.i
; CHECK-NEXT:   %iv5 = phi i64 [ %iv.next6, %for.body.i.i.i.i.i.i.i.i ], [ 0, %for.body5.i.i.i.i.i.i.i.i.i.i ]
; CHECK-NEXT:   %iv.next6 = add nuw nsw i64 %iv5, 1
; CHECK-NEXT:   %mul.i.i.i.i20.i.i.i.i.i.i.i = shl nsw i64 %iv5, 2
; CHECK-NEXT:   %valptr = getelementptr inbounds double, double* %add.ptr, i64 %mul.i.i.i.i20.i.i.i.i.i.i.i
; CHECK-NEXT:   %val = load double, double* %valptr, align 8, !tbaa !26
; CHECK-NEXT:   %3 = shl nuw nsw i64 %iv5, 4
; CHECK-NEXT:   %4 = add nuw nsw i64 %1, %3
; CHECK-NEXT:   %5 = getelementptr inbounds double, double* %val_malloccache, i64 %4
; CHECK-NEXT:   store double %val, double* %5, align 8, !invariant.group !43
; CHECK-NEXT:   %cmp.i.i.i15.i.i.i.i.i = icmp eq i64 %iv.next6, 4
; CHECK-NEXT:   br i1 %cmp.i.i.i15.i.i.i.i.i, label %_ZNK5Eigen9DenseBaseINS_13CwiseBinaryOpINS_8internal17scalar_product_opIddEEKNS_9TransposeIKNS_5BlockIKNS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1ELin1ELb0EEEEEKNS6_IS9_Lin1ELi1ELb1EEEEEE5reduxINS2_13scalar_sum_opIddEEEEdRKT_.exit.i.i.i.i.i.i, label %for.body.i.i.i.i.i.i.i.i

; CHECK: _ZNK5Eigen9DenseBaseINS_13CwiseBinaryOpINS_8internal17scalar_product_opIddEEKNS_9TransposeIKNS_5BlockIKNS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1ELin1ELb0EEEEEKNS6_IS9_Lin1ELi1ELb1EEEEEE5reduxINS2_13scalar_sum_opIddEEEEdRKT_.exit.i.i.i.i.i.i: ; preds = %for.body.i.i.i.i.i.i.i.i
; CHECK-NEXT:   %cmp3.i.i.i.i.i.i.i.i.i.i = icmp eq i64 %iv.next4, 4
; CHECK-NEXT:   br i1 %cmp3.i.i.i.i.i.i.i.i.i.i, label %for.cond.cleanup4.i.i.i.i.i.i.i.i.i.i, label %for.body5.i.i.i.i.i.i.i.i.i.i

; CHECK: for.cond.cleanup4.i.i.i.i.i.i.i.i.i.i:            ; preds = %_ZNK5Eigen9DenseBaseINS_13CwiseBinaryOpINS_8internal17scalar_product_opIddEEKNS_9TransposeIKNS_5BlockIKNS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1ELin1ELb0EEEEEKNS6_IS9_Lin1ELi1ELb1EEEEEE5reduxINS2_13scalar_sum_opIddEEEEdRKT_.exit.i.i.i.i.i.i
; CHECK-NEXT:   %cmp.i1.i.i.i.i.i.i.i.i.i = icmp eq i64 %iv.next2, 4
; CHECK-NEXT:   br i1 %cmp.i1.i.i.i.i.i.i.i.i.i, label %invertfor.cond.cleanup4.i.i.i.i.i.i.i.i.i.i, label %for.cond1.preheader.i.i.i.i.i.i.i.i.i.i

; CHECK: invertentry:                                      ; preds = %invertdfor
; CHECK-NEXT:   tail call void @free(i8* nonnull %"call.i.i.i.i'mi")
; CHECK-NEXT:   tail call void @free(i8* %call.i.i.i.i)
; CHECK-NEXT:   ret {} undef

; CHECK: invertdfor:                                       ; preds = %invertmid, %incinvertdfor
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ 15, %invertmid ], [ %12, %incinvertdfor ]
; CHECK-NEXT:   %"arrayidx.i.i.i.i.i.i.i.i.i'ipg_unwrap" = getelementptr inbounds double, double* %"doubles'ipc", i64 %"iv'ac.0"
; CHECK-NEXT:   %6 = load double, double* %"arrayidx.i.i.i.i.i.i.i.i.i'ipg_unwrap", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"arrayidx.i.i.i.i.i.i.i.i.i'ipg_unwrap", align 8
; CHECK-NEXT:   %"Mi'ipg_unwrap" = getelementptr inbounds double, double* %"ptrM'", i64 %"iv'ac.0"
; CHECK-NEXT:   %7 = load double, double* %"Mi'ipg_unwrap", align 8
; CHECK-NEXT:   %8 = fsub fast double %7, %6
; CHECK-NEXT:   store double %8, double* %"Mi'ipg_unwrap", align 8
; CHECK-NEXT:   %"Wi'ipg_unwrap" = getelementptr inbounds double, double* %"ptrW'", i64 %"iv'ac.0"
; CHECK-NEXT:   %9 = load double, double* %"Wi'ipg_unwrap", align 8
; CHECK-NEXT:   %10 = fadd fast double %9, %6
; CHECK-NEXT:   store double %10, double* %"Wi'ipg_unwrap", align 8
; CHECK-NEXT:   %11 = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %11, label %invertentry, label %incinvertdfor

; CHECK: incinvertdfor:                                    ; preds = %invertdfor
; CHECK-NEXT:   %12 = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertdfor

; CHECK: invertmid:                                        ; preds = %invertfor.cond1.preheader.i.i.i.i.i.i.i.i.i.i
; CHECK-NEXT:   tail call void @free(i8* nonnull %malloccall)
; CHECK-NEXT:   tail call void @free(i8* nonnull %malloccall7)
; CHECK-NEXT:   br label %invertdfor

; CHECK: invertfor.cond1.preheader.i.i.i.i.i.i.i.i.i.i:    ; preds = %invertfor.body5.i.i.i.i.i.i.i.i.i.i
; CHECK-NEXT:   %13 = icmp eq i64 %"iv1'ac.0", 0
; CHECK-NEXT:   br i1 %13, label %invertmid, label %incinvertfor.cond1.preheader.i.i.i.i.i.i.i.i.i.i

; CHECK: incinvertfor.cond1.preheader.i.i.i.i.i.i.i.i.i.i: ; preds = %invertfor.cond1.preheader.i.i.i.i.i.i.i.i.i.i
; CHECK-NEXT:   %14 = add nsw i64 %"iv1'ac.0", -1
; CHECK-NEXT:   br label %invertfor.cond.cleanup4.i.i.i.i.i.i.i.i.i.i

; CHECK: invertfor.body5.i.i.i.i.i.i.i.i.i.i:              ; preds = %invertfor.body.i.i.i.i.i.i.i.i
; CHECK-NEXT:   %15 = icmp eq i64 %"iv3'ac.0", 0
; CHECK-NEXT:   br i1 %15, label %invertfor.cond1.preheader.i.i.i.i.i.i.i.i.i.i, label %incinvertfor.body5.i.i.i.i.i.i.i.i.i.i

; CHECK: incinvertfor.body5.i.i.i.i.i.i.i.i.i.i:           ; preds = %invertfor.body5.i.i.i.i.i.i.i.i.i.i
; CHECK-NEXT:   %16 = add nsw i64 %"iv3'ac.0", -1
; CHECK-NEXT:   br label %invert_ZNK5Eigen9DenseBaseINS_13CwiseBinaryOpINS_8internal17scalar_product_opIddEEKNS_9TransposeIKNS_5BlockIKNS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1ELin1ELb0EEEEEKNS6_IS9_Lin1ELi1ELb1EEEEEE5reduxINS2_13scalar_sum_opIddEEEEdRKT_.exit.i.i.i.i.i.i

; CHECK: invertfor.body.i.i.i.i.i.i.i.i:                   ; preds = %invert_ZNK5Eigen9DenseBaseINS_13CwiseBinaryOpINS_8internal17scalar_product_opIddEEKNS_9TransposeIKNS_5BlockIKNS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1ELin1ELb0EEEEEKNS6_IS9_Lin1ELi1ELb1EEEEEE5reduxINS2_13scalar_sum_opIddEEEEdRKT_.exit.i.i.i.i.i.i, %incinvertfor.body.i.i.i.i.i.i.i.i
; CHECK-NEXT:   %"iv5'ac.0" = phi i64 [ 3, %invert_ZNK5Eigen9DenseBaseINS_13CwiseBinaryOpINS_8internal17scalar_product_opIddEEKNS_9TransposeIKNS_5BlockIKNS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1ELin1ELb0EEEEEKNS6_IS9_Lin1ELi1ELb1EEEEEE5reduxINS2_13scalar_sum_opIddEEEEdRKT_.exit.i.i.i.i.i.i ], [ %31, %incinvertfor.body.i.i.i.i.i.i.i.i ]
; CHECK-NEXT:   %17 = shl nuw nsw i64 %"iv3'ac.0", 2
; CHECK-NEXT:   %18 = add nuw nsw i64 %"iv1'ac.0", %17
; CHECK-NEXT:   %19 = shl nuw nsw i64 %"iv5'ac.0", 4
; CHECK-NEXT:   %20 = add nuw nsw i64 %18, %19
; CHECK-NEXT:   %21 = getelementptr inbounds double, double* %val_malloccache, i64 %20
; CHECK-NEXT:   %22 = load double, double* %21, align 8, !invariant.group !43, !enzyme_fromcache !44
; CHECK-NEXT:   %23 = fadd fast double %22, %22
; CHECK-NEXT:   %24 = fmul fast double %23, %differeturn
; CHECK-NEXT:   %25 = getelementptr inbounds i64, i64* %"tdoub'ipl_malloccache", i64 %18
; CHECK-NEXT:   %26 = bitcast i64* %25 to double**
; CHECK-NEXT:   %27 = load double*, double** %26, align 8
; CHECK-NEXT:   %mul.i.i.i.i20.i.i.i.i.i.i.i_unwrap = shl nsw i64 %"iv5'ac.0", 2
; CHECK-NEXT:   %"valptr'ipg_unwrap" = getelementptr inbounds double, double* %27, i64 %mul.i.i.i.i20.i.i.i.i.i.i.i_unwrap
; CHECK-NEXT:   %28 = load double, double* %"valptr'ipg_unwrap", align 8
; CHECK-NEXT:   %29 = fadd fast double %28, %24
; CHECK-NEXT:   store double %29, double* %"valptr'ipg_unwrap", align 8
; CHECK-NEXT:   %30 = icmp eq i64 %"iv5'ac.0", 0
; CHECK-NEXT:   br i1 %30, label %invertfor.body5.i.i.i.i.i.i.i.i.i.i, label %incinvertfor.body.i.i.i.i.i.i.i.i

; CHECK: incinvertfor.body.i.i.i.i.i.i.i.i:                ; preds = %invertfor.body.i.i.i.i.i.i.i.i
; CHECK-NEXT:   %31 = add nsw i64 %"iv5'ac.0", -1
; CHECK-NEXT:   br label %invertfor.body.i.i.i.i.i.i.i.i

; CHECK: invert_ZNK5Eigen9DenseBaseINS_13CwiseBinaryOpINS_8internal17scalar_product_opIddEEKNS_9TransposeIKNS_5BlockIKNS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1ELin1ELb0EEEEEKNS6_IS9_Lin1ELi1ELb1EEEEEE5reduxINS2_13scalar_sum_opIddEEEEdRKT_.exit.i.i.i.i.i.i: ; preds = %invertfor.cond.cleanup4.i.i.i.i.i.i.i.i.i.i, %incinvertfor.body5.i.i.i.i.i.i.i.i.i.i
; CHECK-NEXT:   %"iv3'ac.0" = phi i64 [ 3, %invertfor.cond.cleanup4.i.i.i.i.i.i.i.i.i.i ], [ %16, %incinvertfor.body5.i.i.i.i.i.i.i.i.i.i ]
; CHECK-NEXT:   br label %invertfor.body.i.i.i.i.i.i.i.i

; CHECK: invertfor.cond.cleanup4.i.i.i.i.i.i.i.i.i.i:      ; preds = %for.cond.cleanup4.i.i.i.i.i.i.i.i.i.i, %incinvertfor.cond1.preheader.i.i.i.i.i.i.i.i.i.i
; CHECK-NEXT:   %"iv1'ac.0" = phi i64 [ %14, %incinvertfor.cond1.preheader.i.i.i.i.i.i.i.i.i.i ], [ 3, %for.cond.cleanup4.i.i.i.i.i.i.i.i.i.i ]
; CHECK-NEXT:   br label %invert_ZNK5Eigen9DenseBaseINS_13CwiseBinaryOpINS_8internal17scalar_product_opIddEEKNS_9TransposeIKNS_5BlockIKNS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1ELin1ELb0EEEEEKNS6_IS9_Lin1ELi1ELb1EEEEEE5reduxINS2_13scalar_sum_opIddEEEEdRKT_.exit.i.i.i.i.i.i
; CHECK-NEXT: }
