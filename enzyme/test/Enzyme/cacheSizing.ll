; RUN: %opt < %s %loadEnzyme -enzyme -enzyme_preopt=false -mem2reg -early-cse -simplifycfg -correlated-propagation -instsimplify -adce -S | FileCheck %s
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
%"struct.Eigen::internal::evaluator.26" = type { %"struct.Eigen::internal::product_evaluator.27" }
%"struct.Eigen::internal::product_evaluator.27" = type { %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"*, %"struct.Eigen::internal::evaluator.15", %"struct.Eigen::internal::evaluator.15", i64 }
%pair = type { double*, i64 }
@str1 = private unnamed_addr constant [13 x i8] c"DID original\00"
@str = private unnamed_addr constant [13 x i8] c"did original\00"

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
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %0) #6
  %1 = bitcast %"class.Eigen::CwiseNullaryOp"* %ref.tmp to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %1) #6
  %m_value.i.i.i.i1 = getelementptr inbounds %"class.Eigen::CwiseNullaryOp", %"class.Eigen::CwiseNullaryOp"* %ref.tmp, i64 0, i32 0, i32 0
  store i64 4, i64* %m_value.i.i.i.i1, align 8, !tbaa !2, !alias.scope !7
  %m_value.i1.i.i.i = getelementptr inbounds %"class.Eigen::CwiseNullaryOp", %"class.Eigen::CwiseNullaryOp"* %ref.tmp, i64 0, i32 1, i32 0
  store i64 4, i64* %m_value.i1.i.i.i, align 8, !tbaa !2, !alias.scope !7
  %m_functor.i.i.i = getelementptr inbounds %"class.Eigen::CwiseNullaryOp", %"class.Eigen::CwiseNullaryOp"* %ref.tmp, i64 0, i32 2
  %2 = bitcast %"struct.Eigen::internal::scalar_constant_op"* %m_functor.i.i.i to i64*
  store i64 4607182418800017408, i64* %2, align 8, !tbaa !12, !alias.scope !7
  call void @llvm.memset.p0i8.i64(i8* align 8 %0, i8 0, i64 24, i1 false) #6
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
  call void @free(i8* %7) #6
  %tobool.i = icmp eq i64 %3, 0
  br i1 %tobool.i, label %if.end8.sink.split.i, label %if.end.i.i

if.end.i.i:                                       ; preds = %if.then.i
  %mul.i.i8 = shl i64 %3, 5
  %call.i.i.i.i = call noalias i8* @malloc(i64 %mul.i.i8) #6
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
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %9) #6
  %10 = bitcast %"struct.Eigen::internal::evaluator.73"* %srcEvaluator.i.i.i.i.i.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %10) #6
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
  call void @free(i8* %18) #6
  %tobool.i16 = icmp eq i64 %mul.i.i.i.i.i.i.i.i, 0
  br i1 %tobool.i16, label %if.end8.sink.split.i34, label %if.end.i.i21

if.end.i.i21:                                     ; preds = %if.then.i17
  %mul.i.i23 = shl i64 %mul.i.i.i.i.i.i.i.i, 3
  %call.i.i.i.i24 = call noalias i8* @malloc(i64 %mul.i.i23) #6
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
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %20) #6
  %21 = bitcast %"class.Eigen::Matrix"* %W to i64*
  %22 = load i64, i64* %21, align 8, !tbaa !19
  %23 = bitcast %"struct.Eigen::internal::evaluator.15"* %dstEvaluator.i.i.i.i.i.i to i64*
  store i64 %22, i64* %23, align 8, !tbaa !20
  %24 = load i64, i64* %m_rows.i4, align 8, !tbaa !15
  %m_value.i.i.i65 = getelementptr inbounds %"struct.Eigen::internal::evaluator.15", %"struct.Eigen::internal::evaluator.15"* %dstEvaluator.i.i.i.i.i.i, i64 0, i32 0, i32 1, i32 0
  store i64 %24, i64* %m_value.i.i.i65, align 8, !tbaa !2
  %25 = bitcast %"class.Eigen::internal::generic_dense_assignment_kernel.76"* %kernel.i.i.i.i.i.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %25) #6
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
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %25) #6
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %20) #6
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %10) #6
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %9) #6
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %1) #6
  %35 = bitcast %"class.Eigen::Matrix"* %M to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %35) #6
  %36 = bitcast %"class.Eigen::CwiseNullaryOp"* %ref.tmp2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %36) #6
  %m_value.i.i.i.i38 = getelementptr inbounds %"class.Eigen::CwiseNullaryOp", %"class.Eigen::CwiseNullaryOp"* %ref.tmp2, i64 0, i32 0, i32 0
  store i64 4, i64* %m_value.i.i.i.i38, align 8, !tbaa !2, !alias.scope !27
  %m_value.i1.i.i.i40 = getelementptr inbounds %"class.Eigen::CwiseNullaryOp", %"class.Eigen::CwiseNullaryOp"* %ref.tmp2, i64 0, i32 1, i32 0
  store i64 4, i64* %m_value.i1.i.i.i40, align 8, !tbaa !2, !alias.scope !27
  %m_functor.i.i.i41 = getelementptr inbounds %"class.Eigen::CwiseNullaryOp", %"class.Eigen::CwiseNullaryOp"* %ref.tmp2, i64 0, i32 2
  %37 = bitcast %"struct.Eigen::internal::scalar_constant_op"* %m_functor.i.i.i41 to i64*
  store i64 4611686018427387904, i64* %37, align 8, !tbaa !12, !alias.scope !27
  call void @llvm.memset.p0i8.i64(i8* align 8 %35, i8 0, i64 24, i1 false) #6
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
  call void @free(i8* %42) #6
  %tobool.i47 = icmp eq i64 %38, 0
  br i1 %tobool.i47, label %if.end8.sink.split.i65, label %if.end.i.i52

if.end.i.i52:                                     ; preds = %if.then.i48
  %mul.i.i54 = shl i64 %38, 5
  %call.i.i.i.i55 = call noalias i8* @malloc(i64 %mul.i.i54) #6
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
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %44) #6
  %45 = bitcast %"struct.Eigen::internal::evaluator.73"* %srcEvaluator.i.i.i.i.i.i2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %45) #6
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
  call void @free(i8* %53) #6
  %tobool.i72 = icmp eq i64 %mul.i.i.i.i.i.i.i.i61, 0
  br i1 %tobool.i72, label %if.end8.sink.split.i90, label %if.end.i.i77

if.end.i.i77:                                     ; preds = %if.then.i73
  %mul.i.i79 = shl i64 %mul.i.i.i.i.i.i.i.i61, 3
  %call.i.i.i.i80 = call noalias i8* @malloc(i64 %mul.i.i79) #6
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
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %55) #6
  %56 = bitcast %"class.Eigen::Matrix"* %M to i64*
  %57 = load i64, i64* %56, align 8, !tbaa !19
  %58 = bitcast %"struct.Eigen::internal::evaluator.15"* %dstEvaluator.i.i.i.i.i.i3 to i64*
  store i64 %57, i64* %58, align 8, !tbaa !20
  %59 = load i64, i64* %m_rows.i42, align 8, !tbaa !15
  %m_value.i.i.i86 = getelementptr inbounds %"struct.Eigen::internal::evaluator.15", %"struct.Eigen::internal::evaluator.15"* %dstEvaluator.i.i.i.i.i.i3, i64 0, i32 0, i32 1, i32 0
  store i64 %59, i64* %m_value.i.i.i86, align 8, !tbaa !2
  %60 = bitcast %"class.Eigen::internal::generic_dense_assignment_kernel.76"* %kernel.i.i.i.i.i.i4 to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %60) #6
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
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %60) #6
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %55) #6
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %45) #6
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %44) #6
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %36) #6
  %70 = bitcast %"class.Eigen::Matrix"* %Wp to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %70) #6
  %71 = bitcast %"class.Eigen::CwiseNullaryOp"* %ref.tmp4 to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %71) #6
  %m_value.i.i.i.i94 = getelementptr inbounds %"class.Eigen::CwiseNullaryOp", %"class.Eigen::CwiseNullaryOp"* %ref.tmp4, i64 0, i32 0, i32 0
  store i64 4, i64* %m_value.i.i.i.i94, align 8, !tbaa !2, !alias.scope !32
  %m_value.i1.i.i.i96 = getelementptr inbounds %"class.Eigen::CwiseNullaryOp", %"class.Eigen::CwiseNullaryOp"* %ref.tmp4, i64 0, i32 1, i32 0
  store i64 4, i64* %m_value.i1.i.i.i96, align 8, !tbaa !2, !alias.scope !32
  %m_functor.i.i.i97 = getelementptr inbounds %"class.Eigen::CwiseNullaryOp", %"class.Eigen::CwiseNullaryOp"* %ref.tmp4, i64 0, i32 2
  %72 = bitcast %"struct.Eigen::internal::scalar_constant_op"* %m_functor.i.i.i97 to i64*
  store i64 0, i64* %72, align 8, !tbaa !12, !alias.scope !32
  call void @llvm.memset.p0i8.i64(i8* align 8 %70, i8 0, i64 24, i1 false) #6
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
  call void @free(i8* %77) #6
  %tobool.i103 = icmp eq i64 %73, 0
  br i1 %tobool.i103, label %if.end8.sink.split.i121, label %if.end.i.i108

if.end.i.i108:                                    ; preds = %if.then.i104
  %mul.i.i110 = shl i64 %73, 5
  %call.i.i.i.i111 = call noalias i8* @malloc(i64 %mul.i.i110) #6
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
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %79) #6
  %80 = bitcast %"struct.Eigen::internal::evaluator.73"* %srcEvaluator.i.i.i.i.i.i80 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %80) #6
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
  call void @free(i8* %88) #6
  %tobool.i128 = icmp eq i64 %mul.i.i.i.i.i.i.i.i139, 0
  br i1 %tobool.i128, label %if.end8.sink.split.i146, label %if.end.i.i133

if.end.i.i133:                                    ; preds = %if.then.i129
  %mul.i.i135 = shl i64 %mul.i.i.i.i.i.i.i.i139, 3
  %call.i.i.i.i136 = call noalias i8* @malloc(i64 %mul.i.i135) #6
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
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %90) #6
  %91 = bitcast %"class.Eigen::Matrix"* %Wp to i64*
  %92 = load i64, i64* %91, align 8, !tbaa !19
  %93 = bitcast %"struct.Eigen::internal::evaluator.15"* %dstEvaluator.i.i.i.i.i.i81 to i64*
  store i64 %92, i64* %93, align 8, !tbaa !20
  %94 = load i64, i64* %m_rows.i98, align 8, !tbaa !15
  %m_value.i.i.i53 = getelementptr inbounds %"struct.Eigen::internal::evaluator.15", %"struct.Eigen::internal::evaluator.15"* %dstEvaluator.i.i.i.i.i.i81, i64 0, i32 0, i32 1, i32 0
  store i64 %94, i64* %m_value.i.i.i53, align 8, !tbaa !2
  %95 = bitcast %"class.Eigen::internal::generic_dense_assignment_kernel.76"* %kernel.i.i.i.i.i.i82 to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %95) #6
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
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %95) #6
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %90) #6
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %80) #6
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %79) #6
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %71) #6
  %105 = bitcast %"class.Eigen::Matrix"* %Mp to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %105) #6
  %106 = bitcast %"class.Eigen::CwiseNullaryOp"* %ref.tmp6 to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %106) #6
  %m_value.i.i.i.i150 = getelementptr inbounds %"class.Eigen::CwiseNullaryOp", %"class.Eigen::CwiseNullaryOp"* %ref.tmp6, i64 0, i32 0, i32 0
  store i64 4, i64* %m_value.i.i.i.i150, align 8, !tbaa !2, !alias.scope !37
  %m_value.i1.i.i.i152 = getelementptr inbounds %"class.Eigen::CwiseNullaryOp", %"class.Eigen::CwiseNullaryOp"* %ref.tmp6, i64 0, i32 1, i32 0
  store i64 4, i64* %m_value.i1.i.i.i152, align 8, !tbaa !2, !alias.scope !37
  %m_functor.i.i.i153 = getelementptr inbounds %"class.Eigen::CwiseNullaryOp", %"class.Eigen::CwiseNullaryOp"* %ref.tmp6, i64 0, i32 2
  %107 = bitcast %"struct.Eigen::internal::scalar_constant_op"* %m_functor.i.i.i153 to i64*
  store i64 0, i64* %107, align 8, !tbaa !12, !alias.scope !37
  call void @llvm.memset.p0i8.i64(i8* align 8 %105, i8 0, i64 24, i1 false) #6
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
  call void @free(i8* %112) #6
  %tobool.i159 = icmp eq i64 %108, 0
  br i1 %tobool.i159, label %if.end8.sink.split.i177, label %if.end.i.i164

if.end.i.i164:                                    ; preds = %if.then.i160
  %mul.i.i166 = shl i64 %108, 5
  %call.i.i.i.i167 = call noalias i8* @malloc(i64 %mul.i.i166) #6
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
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %114) #6
  %115 = bitcast %"struct.Eigen::internal::evaluator.73"* %srcEvaluator.i.i.i.i.i.i158 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %115) #6
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
  call void @free(i8* %123) #6
  %tobool.i184 = icmp eq i64 %mul.i.i.i.i.i.i.i.i217, 0
  br i1 %tobool.i184, label %if.end8.sink.split.i202, label %if.end.i.i189

if.end.i.i189:                                    ; preds = %if.then.i185
  %mul.i.i191 = shl i64 %mul.i.i.i.i.i.i.i.i217, 3
  %call.i.i.i.i192 = call noalias i8* @malloc(i64 %mul.i.i191) #6
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
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %125) #6
  %126 = bitcast %"class.Eigen::Matrix"* %Mp to i64*
  %127 = load i64, i64* %126, align 8, !tbaa !19
  %128 = bitcast %"struct.Eigen::internal::evaluator.15"* %dstEvaluator.i.i.i.i.i.i159 to i64*
  store i64 %127, i64* %128, align 8, !tbaa !20
  %129 = load i64, i64* %m_rows.i154, align 8, !tbaa !15
  %m_value.i.i.i32 = getelementptr inbounds %"struct.Eigen::internal::evaluator.15", %"struct.Eigen::internal::evaluator.15"* %dstEvaluator.i.i.i.i.i.i159, i64 0, i32 0, i32 1, i32 0
  store i64 %129, i64* %m_value.i.i.i32, align 8, !tbaa !2
  %130 = bitcast %"class.Eigen::internal::generic_dense_assignment_kernel.76"* %kernel.i.i.i.i.i.i160 to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %130) #6
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
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %130) #6
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %125) #6
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %115) #6
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %114) #6
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %106) #6
  %w = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %W, i64 0, i32 0, i32 0, i32 0
  %wp = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %Wp, i64 0, i32 0, i32 0, i32 0
  %m = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %M, i64 0, i32 0, i32 0, i32 0
  %mp = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %Mp, i64 0, i32 0, i32 0, i32 0
  %_w = load double*, double** %w, align 8, !tbaa !19
  %_wp = load double*, double** %wp, align 8, !tbaa !19
  %_m = load double*, double** %m, align 8, !tbaa !19
  %_mp = load double*, double** %mp, align 8, !tbaa !19
  %call = call double (...) @__enzyme_autodiff(double (double*, double*)* nonnull @_ZL6matvecPKN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEES3_, double* nonnull %_w, double* nonnull %_wp, double* nonnull %_m, double* nonnull %_mp) #6
  br label %for.cond8.preheader

for.cond8.preheader:                              ; preds = %for.cond.cleanup11, %_ZN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEC2INS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES1_EEEERKNS_9EigenBaseIT_EE.exit234
  %indvars.iv103 = phi i64 [ 0, %_ZN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEC2INS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES1_EEEERKNS_9EigenBaseIT_EE.exit234 ], [ %indvars.iv.next104, %for.cond.cleanup11 ]
  %140 = trunc i64 %indvars.iv103 to i32
  br label %for.body12

for.cond.cleanup11:                               ; preds = %if.end
  %indvars.iv.next104 = add nuw nsw i64 %indvars.iv103, 1
  %cmp = icmp ult i64 %indvars.iv.next104, 4
  br i1 %cmp, label %for.cond8.preheader, label %for.cond35.preheader

for.body12:                                       ; preds = %if.end, %for.cond8.preheader
  %indvars.iv101 = phi i64 [ 0, %for.cond8.preheader ], [ %indvars.iv.next102, %if.end ]
  %141 = load double*, double** %wp, align 8, !tbaa !19
  %142 = load i64, i64* %m_rows.i98, align 8, !tbaa !15
  %mul.i.i.i = mul nsw i64 %142, %indvars.iv103
  %add.i.i.i = add nsw i64 %mul.i.i.i, %indvars.iv101
  %arrayidx.i.i.i = getelementptr inbounds double, double* %141, i64 %add.i.i.i
  %143 = load double, double* %arrayidx.i.i.i, align 8, !tbaa !26
  %sub = fadd double %143, 8.000000e+00
  %144 = call double @llvm.fabs.f64(double %sub)
  %cmp16 = fcmp ogt double %144, 1.000000e-10
  %145 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !22
  br i1 %cmp16, label %if.then, label %if.end

if.then:                                          ; preds = %for.body12
  %call20 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %145, i8* getelementptr inbounds ([68 x i8], [68 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str.1, i64 0, i64 0), double %143, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.2, i64 0, i64 0), double -8.000000e+00, double 1.000000e-10, i8* getelementptr inbounds ([63 x i8], [63 x i8]* @.str.3, i64 0, i64 0), i32 61, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @__PRETTY_FUNCTION__.main, i64 0, i64 0)) #8
  call void @abort() #9
  unreachable

if.end:                                           ; preds = %for.body12
  %146 = trunc i64 %indvars.iv101 to i32
  %call24 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %145, i8* getelementptr inbounds ([19 x i8], [19 x i8]* @.str.4, i64 0, i64 0), i32 %146, i32 %140, double %143) #8
  %indvars.iv.next102 = add nuw nsw i64 %indvars.iv101, 1
  %cmp10 = icmp ult i64 %indvars.iv.next102, 4
  br i1 %cmp10, label %for.body12, label %for.cond.cleanup11

for.cond35.preheader:                             ; preds = %for.cond.cleanup11, %for.cond.cleanup38
  %indvars.iv99 = phi i64 [ %indvars.iv.next100, %for.cond.cleanup38 ], [ 0, %for.cond.cleanup11 ]
  %147 = trunc i64 %indvars.iv99 to i32
  br label %for.body39

for.cond.cleanup32:                               ; preds = %for.cond.cleanup38
  %148 = bitcast %"class.Eigen::Matrix"* %Mp to i8**
  %149 = load i8*, i8** %148, align 8, !tbaa !19
  call void @free(i8* %149) #6
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %105) #6
  %150 = bitcast %"class.Eigen::Matrix"* %Wp to i8**
  %151 = load i8*, i8** %150, align 8, !tbaa !19
  call void @free(i8* %151) #6
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %70) #6
  %152 = bitcast %"class.Eigen::Matrix"* %M to i8**
  %153 = load i8*, i8** %152, align 8, !tbaa !19
  call void @free(i8* %153) #6
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %35) #6
  %154 = bitcast %"class.Eigen::Matrix"* %W to i8**
  %155 = load i8*, i8** %154, align 8, !tbaa !19
  call void @free(i8* %155) #6
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %0) #6
  ret i32 0

for.cond.cleanup38:                               ; preds = %if.end50
  %indvars.iv.next100 = add nuw nsw i64 %indvars.iv99, 1
  %cmp31 = icmp ult i64 %indvars.iv.next100, 4
  br i1 %cmp31, label %for.cond35.preheader, label %for.cond.cleanup32

for.body39:                                       ; preds = %if.end50, %for.cond35.preheader
  %indvars.iv = phi i64 [ 0, %for.cond35.preheader ], [ %indvars.iv.next, %if.end50 ]
  %156 = load double*, double** %mp, align 8, !tbaa !19
  %157 = load i64, i64* %m_rows.i154, align 8, !tbaa !15
  %mul.i.i.i251 = mul nsw i64 %157, %indvars.iv99
  %add.i.i.i252 = add nsw i64 %mul.i.i.i251, %indvars.iv
  %arrayidx.i.i.i253 = getelementptr inbounds double, double* %156, i64 %add.i.i.i252
  %158 = load double, double* %arrayidx.i.i.i253, align 8, !tbaa !26
  %sub43 = fadd double %158, -8.000000e+00
  %159 = call double @llvm.fabs.f64(double %sub43)
  %cmp44 = fcmp ogt double %159, 1.000000e-10
  %160 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !22
  br i1 %cmp44, label %if.then45, label %if.end50

if.then45:                                        ; preds = %for.body39
  %call49 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %160, i8* getelementptr inbounds ([68 x i8], [68 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str.5, i64 0, i64 0), double %158, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.6, i64 0, i64 0), double 8.000000e+00, double 1.000000e-10, i8* getelementptr inbounds ([63 x i8], [63 x i8]* @.str.3, i64 0, i64 0), i32 67, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @__PRETTY_FUNCTION__.main, i64 0, i64 0)) #8
  call void @abort() #9
  unreachable

if.end50:                                         ; preds = %for.body39
  %161 = trunc i64 %indvars.iv to i32
  %call54 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %160, i8* getelementptr inbounds ([19 x i8], [19 x i8]* @.str.7, i64 0, i64 0), i32 %161, i32 %147, double %158) #8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %cmp37 = icmp ult i64 %indvars.iv.next, 4
  br i1 %cmp37, label %for.body39, label %for.cond.cleanup38
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

declare dso_local double @__enzyme_autodiff(...)

; Function Attrs: alwaysinline uwtable
define internal double @_ZL6matvecPKN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEES3_(double* noalias %W, double* noalias %M) {
entry:
  %srcEvaluator2 = alloca %"struct.Eigen::internal::evaluator.26", align 8
  %diff = alloca { double*, i64 }, align 8
  %diffptr = bitcast { double*, i64 }* %diff to double**
  %m_rows.i1 = getelementptr inbounds { double*, i64 }, { double*, i64 }* %diff, i64 0, i32 1
  %.cast = alloca double, i64 16
  store double* %.cast, double** %diffptr, align 8, !tbaa !19
  store i64 4, i64* %m_rows.i1, align 8, !tbaa !15
  br label %subfor

subfor:                                           ; preds = %subfor, %entry
  %i = phi i64 [ %inc, %subfor ], [ 0, %entry ]
  %resi = getelementptr inbounds double, double* %.cast, i64 %i
  %Wi = getelementptr inbounds double, double* %W, i64 %i
  %Mi = getelementptr inbounds double, double* %M, i64 %i
  %wwi = load double, double* %Wi, align 8
  %mmi = load double, double* %Mi, align 8
  %sub = fsub double %wwi, %mmi
  store double %sub, double* %resi, align 8
  %inc = add nuw nsw i64 %i, 1
  %exitcond.i.i.i.i.i.i.i = icmp eq i64 %inc, 16
  br i1 %exitcond.i.i.i.i.i.i.i, label %internal, label %subfor

internal:                                         ; preds = %subfor
  %savedstack = call i8* @llvm.stacksave()
  %a = ptrtoint { double*, i64 }* %diff to i64
  br label %matfor1

matfor1:                                          ; preds = %for.cond.cleanup4, %internal
  %m1 = phi i64 [ 1, %for.cond.cleanup4 ], [ 0, %internal ]
  %rows = load i64, i64* %m_rows.i1, align 8
  br label %matfor2

matfor2:                                          ; preds = %scalar, %matfor1
  %m2 = phi i64 [ %m2.next, %scalar ], [ 0, %matfor1 ]
  %sum2 = phi double [ %zadd, %scalar ], [ 0.000000e+00, %matfor1 ]
  %add.ptr.Z = getelementptr inbounds double, double* %.cast, i64 %m2
  br label %matfor3

matfor3:                                          ; preds = %matfor3, %matfor2
  %res = phi double [ 0.000000e+00, %matfor2 ], [ %madd, %matfor3 ]
  %m3 = phi i64 [ 0, %matfor2 ], [ %m3.next, %matfor3 ]
  %mul.i.i.i.i20.i.i.i.i.i.i.i = shl nsw i64 %m3, 2
  %arrayidx = getelementptr inbounds double, double* %add.ptr.Z, i64 %mul.i.i.i.i20.i.i.i.i.i.i.i
  %val = load double, double* %arrayidx, align 8
  %mul26 = fmul double %val, %val
  %mul4 = fmul double %mul26, 4.000000e+00
  %madd = fadd double %res, %mul4
  %m3.next = add nuw nsw i64 %m3, 1
  %cmp3 = icmp slt i64 %m3.next, %rows
  br i1 %cmp3, label %matfor3, label %scalar

scalar:                                           ; preds = %matfor3
  %zadd = fadd double %madd, %sum2
  %m2.next = add nuw nsw i64 %m2, 1
  %cmp2 = icmp ult i64 %m2.next, 4
  br i1 %cmp2, label %matfor2, label %for.cond.cleanup4

for.cond.cleanup4:                                ; preds = %scalar
  br i1 false, label %matfor1, label %dense

dense:                                            ; preds = %for.cond.cleanup4
  call void @llvm.stackrestore(i8* %savedstack)
  ret double %zadd
}

declare i32 @puts(i8* nocapture readonly) local_unnamed_addr

; Function Attrs: nounwind readnone speculatable
declare double @llvm.fabs.f64(double) #3

; Function Attrs: nounwind
declare dso_local i32 @fprintf(%struct._IO_FILE* nocapture, i8* nocapture readonly, ...) local_unnamed_addr #4

; Function Attrs: noreturn nounwind
declare dso_local void @abort() local_unnamed_addr #5

; Function Attrs: nounwind
declare dso_local void @free(i8* nocapture) local_unnamed_addr #4

; Function Attrs: nounwind
declare dso_local noalias i8* @malloc(i64) local_unnamed_addr #4

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1) #1

; Function Attrs: nounwind
declare dso_local i32 @__cxa_guard_acquire(i64*) local_unnamed_addr #6

; Function Attrs: nounwind
declare dso_local void @__cxa_guard_release(i64*) local_unnamed_addr #6

; Function Attrs: inaccessiblemem_or_argmemonly nounwind
declare void @llvm.prefetch(i8* nocapture readonly, i32, i32, i32) #7

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1) #1

; Function Attrs: nounwind
declare i8* @llvm.stacksave() #6

; Function Attrs: nounwind
declare void @llvm.stackrestore(i8*) #6

attributes #0 = { alwaysinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { alwaysinline uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind readnone speculatable }
attributes #4 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { noreturn nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { nounwind }
attributes #7 = { inaccessiblemem_or_argmemonly nounwind }
attributes #8 = { cold }
attributes #9 = { noreturn nounwind }

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
!42 = !{!43, !17, i64 0}
!43 = !{!"_ZTSN5Eigen8internal17product_evaluatorINS_7ProductINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEES4_Li1EEELi8ENS_10DenseShapeES6_ddEE", !17, i64 0, !17, i64 8, !44, i64 16, !44, i64 32, !4, i64 48}
!44 = !{!"_ZTSN5Eigen8internal9evaluatorINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEEE"}

; CHECK: define internal void @diffe_ZL6matvecPKN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEES3_(double* noalias %W, double* %"W'", double* noalias %M, double* %"M'", double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"diff'ipa" = alloca { double*, i64 }, align 8
; CHECK-NEXT:   store { double*, i64 } zeroinitializer, { double*, i64 }* %"diff'ipa", align 8
; CHECK-NEXT:   %diff = alloca { double*, i64 }, align 8
; CHECK-NEXT:   %"diffptr'ipc" = bitcast { double*, i64 }* %"diff'ipa" to double**
; CHECK-NEXT:   %diffptr = bitcast { double*, i64 }* %diff to double**
; CHECK-NEXT:   %m_rows.i1 = getelementptr inbounds { double*, i64 }, { double*, i64 }* %diff, i64 0, i32 1
; CHECK-NEXT:   %".cast'ipa" = alloca double, i64 16
; CHECK-NEXT:   %0 = bitcast double* %".cast'ipa" to i8*
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull %0, i8 0, i64 128, i1 false)
; CHECK-NEXT:   %.cast = alloca double, i64 16
; CHECK-NEXT:   store double* %".cast'ipa", double** %"diffptr'ipc", align 8
; CHECK-NEXT:   store double* %.cast, double** %diffptr, align 8, !tbaa !19
; CHECK-NEXT:   store i64 4, i64* %m_rows.i1, align 8, !tbaa !15
; CHECK-NEXT:   br label %subfor

; CHECK: subfor:                                           ; preds = %subfor, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %subfor ], [ 0, %entry ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %resi = getelementptr inbounds double, double* %.cast, i64 %iv
; CHECK-NEXT:   %Wi = getelementptr inbounds double, double* %W, i64 %iv
; CHECK-NEXT:   %Mi = getelementptr inbounds double, double* %M, i64 %iv
; CHECK-NEXT:   %wwi = load double, double* %Wi, align 8
; CHECK-NEXT:   %mmi = load double, double* %Mi, align 8
; CHECK-NEXT:   %sub = fsub double %wwi, %mmi
; CHECK-NEXT:   store double %sub, double* %resi, align 8
; CHECK-NEXT:   %exitcond.i.i.i.i.i.i.i = icmp eq i64 %iv.next, 16
; CHECK-NEXT:   br i1 %exitcond.i.i.i.i.i.i.i, label %internal, label %subfor

; CHECK: internal:                                         ; preds = %subfor
; CHECK-NEXT:   %malloccall = tail call noalias nonnull i8* @malloc(i64 32)
; CHECK-NEXT:   %val_malloccache = bitcast i8* %malloccall to double**
; TODO don't need malloccall12
; CHECK-NEXT:   %malloccall12 = tail call noalias nonnull i8* @malloc(i64 1)
; CHECK-NEXT:   %_malloccache = bitcast i8* %malloccall12 to i1*
; CHECK-NEXT:   %malloccall13 = tail call noalias nonnull i8* @malloc(i64 8)
; CHECK-NEXT:   %smax_malloccache = bitcast i8* %malloccall13 to i64*
; CHECK-NEXT:   %rows = load i64, i64* %m_rows.i1, align 8
; CHECK-NEXT:   %1 = icmp sgt i64 %rows, 1
; CHECK-NEXT:   store i1 %1, i1* %_malloccache, align 1
; CHECK-NEXT:   %smax = select i1 %1, i64 %rows, i64 1
; CHECK-NEXT:   store i64 %smax, i64* %smax_malloccache, align 8
; CHECK-NEXT:   br label %matfor2

; CHECK: matfor2:                                          ; preds = %scalar, %internal
; CHECK-NEXT:   %iv3 = phi i64 [ %iv.next4, %scalar ], [ 0, %internal ]
; CHECK-NEXT:   %iv.next4 = add nuw nsw i64 %iv3, 1
; CHECK-NEXT:   %add.ptr.Z = getelementptr inbounds double, double* %.cast, i64 %iv3
; CHECK-NEXT:   %2 = getelementptr inbounds double*, double** %val_malloccache, i64 %iv3
; CHECK-NEXT:   %mallocsize = mul i64 %smax, 8
; CHECK-NEXT:   %malloccall7 = tail call noalias nonnull i8* @malloc(i64 %mallocsize)
; CHECK-NEXT:   %val_malloccache8 = bitcast i8* %malloccall7 to double*
; CHECK-NEXT:   store double* %val_malloccache8, double** %2, align 8
; CHECK-NEXT:   br label %matfor3

; CHECK: matfor3:                                          ; preds = %matfor3, %matfor2
; CHECK-NEXT:   %iv5 = phi i64 [ %iv.next6, %matfor3 ], [ 0, %matfor2 ]
; CHECK-NEXT:   %iv.next6 = add nuw nsw i64 %iv5, 1
; CHECK-NEXT:   %mul.i.i.i.i20.i.i.i.i.i.i.i = shl nsw i64 %iv5, 2
; CHECK-NEXT:   %arrayidx = getelementptr inbounds double, double* %add.ptr.Z, i64 %mul.i.i.i.i20.i.i.i.i.i.i.i
; CHECK-NEXT:   %val = load double, double* %arrayidx, align 8
; CHECK-NEXT:   %[[db:.+]] = load double*, double** %2, align 8, !dereferenceable
; CHECK-NEXT:   %[[gepz:.+]] = getelementptr inbounds double, double* %[[db]], i64 %iv5
; CHECK-NEXT:   store double %val, double* %[[gepz]], align 8
; CHECK-NEXT:   %cmp3 = icmp slt i64 %iv.next6, %rows
; CHECK-NEXT:   br i1 %cmp3, label %matfor3, label %scalar

; CHECK: scalar:                                           ; preds = %matfor3
; CHECK-NEXT:   %cmp2 = icmp ne i64 %iv.next4, 4
; CHECK-NEXT:   br i1 %cmp2, label %matfor2, label %invertfor.cond.cleanup4

; CHECK: invertentry:                                      ; preds = %invertsubfor
; CHECK-NEXT:   ret void

; CHECK: invertsubfor:                                     ; preds = %invertinternal, %incinvertsubfor
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ 15, %invertinternal ], [ %[[ivsub:.+]], %incinvertsubfor ]
; CHECK-NEXT:   %"resi'ipg_unwrap" = getelementptr inbounds double, double* %".cast'ipa", i64 %"iv'ac.0"
; CHECK-NEXT:   %[[preres:.+]] = load double, double* %"resi'ipg_unwrap", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"resi'ipg_unwrap", align 8
; CHECK-NEXT:   %[[fneg:.+]] = fsub fast double 0.000000e+00, %[[preres]]
; CHECK-NEXT:   %"Mi'ipg_unwrap" = getelementptr inbounds double, double* %"M'", i64 %"iv'ac.0"
; CHECK-NEXT:   %[[prem:.+]] = load double, double* %"Mi'ipg_unwrap", align 8
; CHECK-NEXT:   %[[postm:.+]] = fadd fast double %[[prem]], %[[fneg]]
; CHECK-NEXT:   store double %[[postm]], double* %"Mi'ipg_unwrap", align 8
; CHECK-NEXT:   %"Wi'ipg_unwrap" = getelementptr inbounds double, double* %"W'", i64 %"iv'ac.0"
; CHECK-NEXT:   %[[prew:.+]] = load double, double* %"Wi'ipg_unwrap", align 8
; CHECK-NEXT:   %[[postw:.+]] = fadd fast double %[[prew]], %[[preres]]
; CHECK-NEXT:   store double %[[postw]], double* %"Wi'ipg_unwrap", align 8
; CHECK-NEXT:   %[[iveq:.+]] = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %[[iveq]], label %invertentry, label %incinvertsubfor

; CHECK: incinvertsubfor:                                  ; preds = %invertsubfor
; CHECK-NEXT:   %[[ivsub]] = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertsubfor

; CHECK: invertinternal:                                   ; preds = %invertmatfor1
; CHECK-NEXT:   tail call void @free(i8* nonnull %malloccall)
; CHECK-NEXT:   tail call void @free(i8* nonnull %malloccall12)
; CHECK-NEXT:   tail call void @free(i8* nonnull %malloccall13)
; CHECK-NEXT:   br label %invertsubfor

; CHECK: invertmatfor1:                                    ; preds = %invertmatfor2
; CHECK-NEXT:   %[[iv1eq:.+]] = icmp eq i64 %"iv1'ac.0", 0
; CHECK-NEXT:   br i1 %[[iv1eq]], label %invertinternal, label %incinvertmatfor1

; CHECK: incinvertmatfor1:                                 ; preds = %invertmatfor1
; CHECK-NEXT:   %[[iv1sub:.+]] = add nsw i64 %"iv1'ac.0", -1
; CHECK-NEXT:   br label %invertfor.cond.cleanup4

; CHECK: invertmatfor2:                                    ; preds = %invertmatfor3
; CHECK-NEXT:   %[[endcmp:.+]] = icmp eq i64 %"iv3'ac.0", 0
; CHECK-NEXT:   %[[dtofree:.+]] = load double*, double** %[[valPtr:.+]], align 8, !dereferenceable
; CHECK-NEXT:   %[[tofree:.+]] = bitcast double* %[[dtofree]] to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[tofree]])
; CHECK-NEXT:   br i1 %[[endcmp]], label %invertmatfor1, label %incinvertmatfor2

; CHECK: incinvertmatfor2:                                 ; preds = %invertmatfor2
; CHECK-NEXT:   %[[iv3sub:.+]] = add nsw i64 %"iv3'ac.0", -1
; CHECK-NEXT:   br label %invertscalar

; CHECK: invertmatfor3:                                    ; preds = %invertscalar, %incinvertmatfor3
; CHECK-NEXT:   %"iv5'ac.0" = phi i64 [ %[[iv5start:.+]], %invertscalar ], [ %[[iv5sub:.+]], %incinvertmatfor3 ]
; CHECK-NEXT:   %m0diffemul26 = fmul fast double %"zadd'de.1", 4.000000e+00
; CHECK-NEXT:   %[[iv1p3:.+]] = add nuw nsw i64 %"iv1'ac.0", %"iv3'ac.0"
; CHECK-NEXT:   %[[valPtr]] = getelementptr inbounds double*, double** %val_malloccache, i64 %[[iv1p3]]
; CHECK-NEXT:   %[[vallc:.+]] = load double*, double** %[[valPtr]], align 8, !dereferenceable
; CHECK-NEXT:   %[[fvalptr:.+]] = getelementptr inbounds double, double* %[[vallc]], i64 %"iv5'ac.0"
; CHECK-NEXT:   %[[ival:.+]] = load double, double* %[[fvalptr]], align 8
; CHECK-NEXT:   %[[mv2:.+]] = fmul fast double %m0diffemul26, %[[ival]]
; CHECK-NEXT:   %[[val2:.+]] = fadd fast double %[[mv2]], %[[mv2]]
; CHECK-NEXT:   %"add.ptr.Z'ipg_unwrap" = getelementptr inbounds double, double* %".cast'ipa", i64 %"iv3'ac.0"
; CHECK-NEXT:   %mul.i.i.i.i20.i.i.i.i.i.i.i_unwrap = shl nsw i64 %"iv5'ac.0", 2
; CHECK-NEXT:   %"arrayidx'ipg_unwrap" = getelementptr inbounds double, double* %"add.ptr.Z'ipg_unwrap", i64 %mul.i.i.i.i20.i.i.i.i.i.i.i_unwrap
; CHECK-NEXT:   %[[pidx:.+]] = load double, double* %"arrayidx'ipg_unwrap", align 8
; CHECK-NEXT:   %[[padd:.+]] = fadd fast double %[[pidx]], %[[val2]]
; CHECK-NEXT:   store double %[[padd]], double* %"arrayidx'ipg_unwrap", align 8
; CHECK-NEXT:   %[[ecmp:.+]] = icmp eq i64 %"iv5'ac.0", 0
; CHECK-NEXT:   br i1 %[[ecmp]], label %invertmatfor2, label %incinvertmatfor3

; CHECK: incinvertmatfor3:                                 ; preds = %invertmatfor3
; CHECK-NEXT:   %[[iv5sub]] = add nsw i64 %"iv5'ac.0", -1
; CHECK-NEXT:   br label %invertmatfor3

; CHECK: invertscalar:                                     ; preds = %invertfor.cond.cleanup4, %incinvertmatfor2
; CHECK-NEXT:   %"iv3'ac.0" = phi i64 [ 3, %invertfor.cond.cleanup4 ], [ %[[iv3sub]], %incinvertmatfor2 ]
; CHECK-NEXT:   %[[smax_gep:.+]] = getelementptr inbounds i64, i64* %smax_malloccache, i64 %"iv1'ac.0"
; CHECK-NEXT:   %[[smax:.+]] = load i64, i64* %[[smax_gep]], align 8
; CHECK-NEXT:   %[[iv5start]] = add{{( nsw)?}} i64 %[[smax]], -1
; CHECK-NEXT:   br label %invertmatfor3

; CHECK: invertfor.cond.cleanup4:                          ; preds = %scalar, %incinvertmatfor1
; CHECK-NEXT:   %"zadd'de.1" = phi double [ 0.000000e+00, %incinvertmatfor1 ], [ %differeturn, %scalar ]
; CHECK-NEXT:   %"iv1'ac.0" = phi i64 [ %[[iv1sub]], %incinvertmatfor1 ], [ 0, %scalar ]
; CHECK-NEXT:   br label %invertscalar
; CHECK-NEXT: }
