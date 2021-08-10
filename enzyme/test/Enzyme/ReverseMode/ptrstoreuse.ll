; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -adce -loop-deletion -correlated-propagation -simplifycfg -S | FileCheck %s

source_filename = "/mnt/pci4/wmdata/Enzyme2/enzyme/test/Integration/ReverseMode/eigensumsqdyn.cpp"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%"class.Eigen::Matrix" = type { %"class.Eigen::PlainObjectBase" }
%"class.Eigen::PlainObjectBase" = type { %"class.Eigen::DenseStorage" }
%"class.Eigen::DenseStorage" = type { double*, i64, i64 }
%"class.Eigen::internal::redux_evaluator" = type { %"struct.Eigen::internal::evaluator.18", %"class.Eigen::Product"* }
%"struct.Eigen::internal::evaluator.18" = type { %"struct.Eigen::internal::product_evaluator" }
%"struct.Eigen::internal::product_evaluator" = type { %"struct.Eigen::internal::evaluator.15", %"class.Eigen::Matrix" }
%"struct.Eigen::internal::evaluator.15" = type { %"struct.Eigen::internal::evaluator.16" }
%"struct.Eigen::internal::evaluator.16" = type { double*, %"class.Eigen::internal::variable_if_dynamic" }
%"class.Eigen::internal::variable_if_dynamic" = type { i64 }
%"class.Eigen::Product" = type { %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }
%"struct.Eigen::internal::scalar_sum_op" = type { i8 }
%"class.Eigen::internal::gemm_blocking_space" = type { %"class.Eigen::internal::level3_blocking", i64, i64 }
%"class.Eigen::internal::level3_blocking" = type { double*, double*, i64, i64, i64 }
%"struct.Eigen::internal::gemm_functor" = type { %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"*, double, %"class.Eigen::internal::gemm_blocking_space"* }
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
%"struct.Eigen::EigenBase" = type { i8 }
%"class.Eigen::DenseBase.33" = type { i8 }
%"class.Eigen::DenseBase" = type { i8 }
%"struct.Eigen::internal::GemmParallelInfo" = type opaque
%"struct.Eigen::internal::gemm_pack_lhs" = type { i8 }
%"class.Eigen::internal::const_blas_data_mapper" = type { %"class.Eigen::internal::blas_data_mapper" }
%"class.Eigen::internal::blas_data_mapper" = type { double*, i64 }

$_ZN5Eigen8internal19throw_std_bad_allocEv = comdat any

$_ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EE6resizeElll = comdat any

$_ZN5Eigen8internal20generic_product_implINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEES3_NS_10DenseShapeES4_Li8EE6evalToIS3_EEvRT_RKS3_SA_ = comdat any

$_ZN5Eigen8internal15BlockImpl_denseIKNS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1ELin1ELb0ELb1EEC2ERS4_l = comdat any

$_ZN5Eigen8internal19variable_if_dynamicIlLi1EEC2ERKS2_ = comdat any

$_ZNK5Eigen9DenseBaseINS_13CwiseBinaryOpINS_8internal17scalar_product_opIddEEKNS_9TransposeIKNS_5BlockIKNS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1ELin1ELb0EEEEEKNS6_IS9_Lin1ELi1ELb1EEEEEE5reduxINS2_13scalar_sum_opIddEEEEdRKT_ = comdat any

$_ZN5Eigen9DenseBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEE11setConstantERKd = comdat any

$_ZSt3minIlERKT_S2_S2_ = comdat any

$_ZNK5Eigen8internal12gemm_functorIdlNS0_29general_matrix_matrix_productIldLi0ELb0EdLi0ELb0ELi0EEENS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEES5_S5_NS0_19gemm_blocking_spaceILi0EddLin1ELin1ELin1ELi1ELb0EEEEclEllllPNS0_16GemmParallelInfoIlEE = comdat any

$_ZN5Eigen8internal29general_matrix_matrix_productIldLi0ELb0EdLi0ELb0ELi0EE3runElllPKdlS4_lPdldRNS0_15level3_blockingIddEEPNS0_16GemmParallelInfoIlEE = comdat any

$_ZN5Eigen8internal13gemm_pack_lhsIdlNS0_22const_blas_data_mapperIdlLi0EEELi1ELi1ELi0ELb0ELb0EEclEPdRKS3_llll = comdat any

define void @caller(%"class.Eigen::Matrix"* nonnull %W, %"class.Eigen::Matrix"* nonnull %Wp, %"class.Eigen::Matrix"* nonnull %M, %"class.Eigen::Matrix"* nonnull %Mp) local_unnamed_addr {
entry:
  %call = call double @__enzyme_autodiff(i8* bitcast (double (%"class.Eigen::Matrix"*, %"class.Eigen::Matrix"*)* @_ZL6matvecPKN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEES3_ to i8*), %"class.Eigen::Matrix"* nonnull %W, %"class.Eigen::Matrix"* nonnull %Wp, %"class.Eigen::Matrix"* nonnull %M, %"class.Eigen::Matrix"* nonnull %Mp) #6
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #0

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #0

declare dso_local double @__enzyme_autodiff(i8*, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"*) local_unnamed_addr #1

; Function Attrs: noinline nounwind uwtable
define internal double @_ZL6matvecPKN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEES3_(%"class.Eigen::Matrix"* noalias %W, %"class.Eigen::Matrix"* noalias %M) #2 {
entry:
  %thisEval.i.i = alloca %"class.Eigen::internal::redux_evaluator", align 8
  %ref.tmp1 = alloca %"class.Eigen::Product", align 8
  %tmp1 = bitcast %"class.Eigen::Product"* %ref.tmp1 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %tmp1) #6
  %tmp.i.i = bitcast %"class.Eigen::internal::redux_evaluator"* %thisEval.i.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 48, i8* nonnull %tmp.i.i) #6
  %m_result.i.i = getelementptr inbounds %"class.Eigen::internal::redux_evaluator", %"class.Eigen::internal::redux_evaluator"* %thisEval.i.i, i64 0, i32 0, i32 0, i32 1
  %tmp2 = bitcast %"class.Eigen::Matrix"* %m_result.i.i to i64*
  %tmp.i9 = load i64, i64* %tmp2, align 8, !tbaa !2
  %tmp7 = bitcast %"class.Eigen::internal::redux_evaluator"* %thisEval.i.i to i64*
  store i64 %tmp.i9, i64* %tmp7, align 8, !tbaa !8
  %m_rows.i.i.i.i.i = getelementptr inbounds %"class.Eigen::internal::redux_evaluator", %"class.Eigen::internal::redux_evaluator"* %thisEval.i.i, i64 0, i32 0, i32 0, i32 1, i32 0, i32 0, i32 1
  %tmp.i.i.i.i.i = load i64, i64* %m_rows.i.i.i.i.i, align 8, !tbaa !11
  %m_value.i.i.i = getelementptr inbounds %"class.Eigen::internal::redux_evaluator", %"class.Eigen::internal::redux_evaluator"* %thisEval.i.i, i64 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 0
  store i64 %tmp.i.i.i.i.i, i64* %m_value.i.i.i, align 8, !tbaa !12
  %m_lhs.i = getelementptr inbounds %"class.Eigen::Product", %"class.Eigen::Product"* %ref.tmp1, i64 0, i32 0
  %tmp.i7 = load %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"** %m_lhs.i, align 8, !tbaa !13
  %m_rhs.i = getelementptr inbounds %"class.Eigen::Product", %"class.Eigen::Product"* %ref.tmp1, i64 0, i32 1
  %tmp.i8 = load %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"** %m_rhs.i, align 8, !tbaa !15
  call void @_ZN5Eigen8internal20generic_product_implINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEES3_NS_10DenseShapeES4_Li8EE6evalToIS3_EEvRT_RKS3_SA_(%"class.Eigen::Matrix"* nonnull dereferenceable(24) %m_result.i.i, %"class.Eigen::Matrix"* nonnull dereferenceable(24) %tmp.i7, %"class.Eigen::Matrix"* nonnull dereferenceable(24) %tmp.i8) #6
  %m_xpr.i.i = getelementptr inbounds %"class.Eigen::internal::redux_evaluator", %"class.Eigen::internal::redux_evaluator"* %thisEval.i.i, i64 0, i32 1
  store %"class.Eigen::Product"* %ref.tmp1, %"class.Eigen::Product"** %m_xpr.i.i, align 8, !tbaa !16
  %m_data.i.i.i = getelementptr inbounds %"class.Eigen::internal::redux_evaluator", %"class.Eigen::internal::redux_evaluator"* %thisEval.i.i, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %tmp.i.i2.i = load double*, double** %m_data.i.i.i, align 8, !tbaa !8
  %tmp1.i3.i = load double, double* %tmp.i.i2.i, align 8, !tbaa !17
  ret double %tmp1.i3.i
}

; Function Attrs: inlinehint norecurse nounwind uwtable
define linkonce_odr dso_local void @_ZN5Eigen8internal19throw_std_bad_allocEv() local_unnamed_addr #3 comdat {
entry:
  ret void
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EE6resizeElll(%"class.Eigen::DenseStorage"* %this, i64 %size, i64 %rows, i64 %cols) local_unnamed_addr #4 comdat align 2 {
entry:
  %m_rows = getelementptr inbounds %"class.Eigen::DenseStorage", %"class.Eigen::DenseStorage"* %this, i64 0, i32 1
  %tmp = load i64, i64* %m_rows, align 8, !tbaa !11
  %m_cols = getelementptr inbounds %"class.Eigen::DenseStorage", %"class.Eigen::DenseStorage"* %this, i64 0, i32 2
  %tmp2 = load i64, i64* %m_cols, align 8, !tbaa !19
  %mul = mul nsw i64 %tmp2, %tmp
  %cmp = icmp eq i64 %mul, %size
  br i1 %cmp, label %if.end8, label %if.then

if.then:                                          ; preds = %entry
  %m_data = getelementptr inbounds %"class.Eigen::DenseStorage", %"class.Eigen::DenseStorage"* %this, i64 0, i32 0
  %tmp3 = bitcast %"class.Eigen::DenseStorage"* %this to i8**
  %tmp4 = load i8*, i8** %tmp3, align 8, !tbaa !2
  tail call void @free(i8* %tmp4) #6
  %tobool = icmp eq i64 %size, 0
  br i1 %tobool, label %if.end8.sink.split, label %if.end.i

if.end.i:                                         ; preds = %if.then
  %cmp.i.i = icmp ugt i64 %size, 2305843009213693951
  br i1 %cmp.i.i, label %if.then.i.i, label %_ZN5Eigen8internal23check_size_for_overflowIdEEvm.exit.i

if.then.i.i:                                      ; preds = %if.end.i
  tail call void @_ZN5Eigen8internal19throw_std_bad_allocEv() #6
  br label %_ZN5Eigen8internal23check_size_for_overflowIdEEvm.exit.i

_ZN5Eigen8internal23check_size_for_overflowIdEEvm.exit.i: ; preds = %if.then.i.i, %if.end.i
  %mul.i = shl i64 %size, 3
  %call.i.i.i = tail call noalias i8* @malloc(i64 %mul.i) #6
  %tobool.i.i.i = icmp eq i8* %call.i.i.i, null
  %tobool1.i.i.i = icmp ne i64 %mul.i, 0
  %or.cond.i.i.i = and i1 %tobool1.i.i.i, %tobool.i.i.i
  br i1 %or.cond.i.i.i, label %if.then.i.i.i, label %_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit

if.then.i.i.i:                                    ; preds = %_ZN5Eigen8internal23check_size_for_overflowIdEEvm.exit.i
  tail call void @_ZN5Eigen8internal19throw_std_bad_allocEv() #6
  br label %_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit

_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit: ; preds = %if.then.i.i.i, %_ZN5Eigen8internal23check_size_for_overflowIdEEvm.exit.i
  %tmp.i1 = bitcast i8* %call.i.i.i to double*
  br label %if.end8.sink.split

if.end8.sink.split:                               ; preds = %_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit, %if.then
  %call.sink = phi double* [ %tmp.i1, %_ZN5Eigen8internal28conditional_aligned_new_autoIdLb1EEEPT_m.exit ], [ null, %if.then ]
  store double* %call.sink, double** %m_data, align 8, !tbaa !2
  br label %if.end8

if.end8:                                          ; preds = %if.end8.sink.split, %entry
  store i64 %rows, i64* %m_rows, align 8, !tbaa !11
  store i64 %cols, i64* %m_cols, align 8, !tbaa !19
  ret void
}

; Function Attrs: nounwind
declare dso_local void @free(i8* nocapture) local_unnamed_addr #5

; Function Attrs: nounwind
declare dso_local noalias i8* @malloc(i64) local_unnamed_addr #5

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZN5Eigen8internal20generic_product_implINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEES3_NS_10DenseShapeES4_Li8EE6evalToIS3_EEvRT_RKS3_SA_(%"class.Eigen::Matrix"* dereferenceable(24) %dst, %"class.Eigen::Matrix"* dereferenceable(24) %lhs, %"class.Eigen::Matrix"* dereferenceable(24) %rhs) local_unnamed_addr #4 comdat align 2 {
entry:
  %ref.tmp.i = alloca double, align 8
  %ref.tmp.i30 = alloca %"struct.Eigen::internal::scalar_sum_op", align 1
  %blocking.i = alloca %"class.Eigen::internal::gemm_blocking_space", align 8
  %ref.tmp.i8 = alloca %"struct.Eigen::internal::gemm_functor", align 8
  %ref.tmp.i.i.i.i.i.i.i.i = alloca %"class.Eigen::CwiseBinaryOp.30", align 8
  %ref.tmp2.i.i.i.i.i.i.i.i = alloca %"class.Eigen::Transpose", align 8
  %ref.tmp3.i.i.i.i.i.i.i.i = alloca %"class.Eigen::Block", align 8
  %ref.tmp4.i.i.i.i.i.i.i.i = alloca %"class.Eigen::Block.47", align 8
  %m_rows.i.i = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %rhs, i64 0, i32 0, i32 0, i32 1
  %tmp.i.i = load i64, i64* %m_rows.i.i, align 8, !tbaa !11
  %m_rows.i.i2 = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %dst, i64 0, i32 0, i32 0, i32 1
  %tmp.i.i3 = load i64, i64* %m_rows.i.i2, align 8, !tbaa !11
  %add = add nsw i64 %tmp.i.i3, %tmp.i.i
  %m_cols.i.i = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %dst, i64 0, i32 0, i32 0, i32 2
  %tmp.i.i5 = load i64, i64* %m_cols.i.i, align 8, !tbaa !19
  %add3 = add nsw i64 %add, %tmp.i.i5
  %cmp = icmp slt i64 %add3, 20
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %m_rows.i.i.i = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %lhs, i64 0, i32 0, i32 0, i32 1
  %tmp.i.i.i3 = load i64, i64* %m_rows.i.i.i, align 8, !tbaa !11
  %m_cols.i.i.i = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %rhs, i64 0, i32 0, i32 0, i32 2
  %tmp.i.i.i6 = load i64, i64* %m_cols.i.i.i, align 8, !tbaa !19
  %m_rows.i.i10 = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %dst, i64 0, i32 0, i32 0, i32 1
  %tmp.i.i11 = load i64, i64* %m_rows.i.i10, align 8, !tbaa !11
  %cmp.i.i.i.i.i = icmp eq i64 %tmp.i.i11, %tmp.i.i.i3
  br i1 %cmp.i.i.i.i.i, label %lor.lhs.false.i.i.i.i.i, label %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEE6resizeEll.exit.i.i.i.i.i

lor.lhs.false.i.i.i.i.i:                          ; preds = %if.then
  %m_cols.i.i13 = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %dst, i64 0, i32 0, i32 0, i32 2
  %tmp.i.i14 = load i64, i64* %m_cols.i.i13, align 8, !tbaa !19
  %cmp4.i.i.i.i.i = icmp eq i64 %tmp.i.i14, %tmp.i.i.i6
  br i1 %cmp4.i.i.i.i.i, label %_ZN5Eigen8internal17resize_if_allowedINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_7ProductIS3_S3_Li1EEEddEEvRT_RKT0_RKNS0_9assign_opIT1_T2_EE.exit.i.i.i.i, label %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEE6resizeEll.exit.i.i.i.i.i

_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEE6resizeEll.exit.i.i.i.i.i: ; preds = %lor.lhs.false.i.i.i.i.i, %if.then
  %m_storage.i.i.i.i.i.i = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %dst, i64 0, i32 0, i32 0
  %mul.i.i.i.i.i.i = mul nsw i64 %tmp.i.i.i6, %tmp.i.i.i3
  call void @_ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EE6resizeElll(%"class.Eigen::DenseStorage"* nonnull %m_storage.i.i.i.i.i.i, i64 %mul.i.i.i.i.i.i, i64 %tmp.i.i.i3, i64 %tmp.i.i.i6) #6
  br label %_ZN5Eigen8internal17resize_if_allowedINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_7ProductIS3_S3_Li1EEEddEEvRT_RKT0_RKNS0_9assign_opIT1_T2_EE.exit.i.i.i.i

_ZN5Eigen8internal17resize_if_allowedINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_7ProductIS3_S3_Li1EEEddEEvRT_RKT0_RKNS0_9assign_opIT1_T2_EE.exit.i.i.i.i: ; preds = %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEE6resizeEll.exit.i.i.i.i.i, %lor.lhs.false.i.i.i.i.i
  %tmp = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %dst, i64 0, i32 0, i32 0, i32 0
  %tmp.i157 = load double*, double** %tmp, align 8, !tbaa !2
  %m_rows.i.i.i.i.i = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %dst, i64 0, i32 0, i32 0, i32 1
  %tmp.i.i.i.i.i = load i64, i64* %m_rows.i.i.i.i.i, align 8, !tbaa !11
  br label %for.cond1.preheader.i.i.i.i.i

for.cond1.preheader.i.i.i.i.i:                    ; preds = %for.cond.cleanup4.i.i.i.i.i, %_ZN5Eigen8internal17resize_if_allowedINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_7ProductIS3_S3_Li1EEEddEEvRT_RKT0_RKNS0_9assign_opIT1_T2_EE.exit.i.i.i.i
  %outer.022.i.i.i.i.i = phi i64 [ %inc7.i.i.i.i.i, %for.cond.cleanup4.i.i.i.i.i ], [ 0, %_ZN5Eigen8internal17resize_if_allowedINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_7ProductIS3_S3_Li1EEEddEEvRT_RKT0_RKNS0_9assign_opIT1_T2_EE.exit.i.i.i.i ]
  br label %for.body5.i.i.i.i.i

for.cond.cleanup4.i.i.i.i.i:                      ; preds = %_ZNK5Eigen9DenseBaseINS_13CwiseBinaryOpINS_8internal17scalar_product_opIddEEKNS_9TransposeIKNS_5BlockIKNS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1ELin1ELb0EEEEEKNS6_IS9_Lin1ELi1ELb1EEEEEE3sumEv.exit
  %inc7.i.i.i.i.i = add nuw nsw i64 %outer.022.i.i.i.i.i, 1
  %m_cols.i.i.i14 = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %dst, i64 0, i32 0, i32 0, i32 2
  %tmp.i.i.i17 = load i64, i64* %m_cols.i.i.i14, align 8, !tbaa !19
  %cmp.i2.i.i.i.i = icmp slt i64 %inc7.i.i.i.i.i, %tmp.i.i.i17
  br i1 %cmp.i2.i.i.i.i, label %for.cond1.preheader.i.i.i.i.i, label %if.end

for.body5.i.i.i.i.i:                              ; preds = %_ZNK5Eigen9DenseBaseINS_13CwiseBinaryOpINS_8internal17scalar_product_opIddEEKNS_9TransposeIKNS_5BlockIKNS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1ELin1ELb0EEEEEKNS6_IS9_Lin1ELi1ELb1EEEEEE3sumEv.exit, %for.cond1.preheader.i.i.i.i.i
  %inner.019.i.i.i.i.i = phi i64 [ %inc.i.i.i.i.i, %_ZNK5Eigen9DenseBaseINS_13CwiseBinaryOpINS_8internal17scalar_product_opIddEEKNS_9TransposeIKNS_5BlockIKNS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1ELin1ELb0EEEEEKNS6_IS9_Lin1ELi1ELb1EEEEEE3sumEv.exit ], [ 0, %for.cond1.preheader.i.i.i.i.i ]
  %mul.i19 = mul nsw i64 %tmp.i.i.i.i.i, %outer.022.i.i.i.i.i
  %add.i = add nsw i64 %mul.i19, %inner.019.i.i.i.i.i
  %arrayidx.i = getelementptr inbounds double, double* %tmp.i157, i64 %add.i
  %tmp.i.i.i.i.i.i.i.i = bitcast %"class.Eigen::CwiseBinaryOp.30"* %ref.tmp.i.i.i.i.i.i.i.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 120, i8* nonnull %tmp.i.i.i.i.i.i.i.i) #6
  %tmp1.i.i.i.i.i.i.i.i = bitcast %"class.Eigen::Transpose"* %ref.tmp2.i.i.i.i.i.i.i.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 56, i8* nonnull %tmp1.i.i.i.i.i.i.i.i) #6
  %tmp2.i.i.i.i.i.i.i.i = bitcast %"class.Eigen::Block"* %ref.tmp3.i.i.i.i.i.i.i.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 56, i8* nonnull %tmp2.i.i.i.i.i.i.i.i) #6
  %tmp.i.i.i21 = getelementptr inbounds %"class.Eigen::Block", %"class.Eigen::Block"* %ref.tmp3.i.i.i.i.i.i.i.i, i64 0, i32 0, i32 0
  call void @_ZN5Eigen8internal15BlockImpl_denseIKNS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1ELin1ELb0ELb1EEC2ERS4_l(%"class.Eigen::internal::BlockImpl_dense"* nonnull %tmp.i.i.i21, %"class.Eigen::Matrix"* nonnull dereferenceable(24) %lhs, i64 %inner.019.i.i.i.i.i) #6
  %tmp.i2.i = bitcast %"class.Eigen::Block"* %ref.tmp3.i.i.i.i.i.i.i.i to i64*
  %tmp1.i.i22 = load i64, i64* %tmp.i2.i, align 8, !tbaa !20, !noalias !23
  %tmp2.i.i23 = bitcast %"class.Eigen::Transpose"* %ref.tmp2.i.i.i.i.i.i.i.i to i64*
  store i64 %tmp1.i.i22, i64* %tmp2.i.i23, align 8, !tbaa !20, !alias.scope !23
  %m_rows.i.i24 = getelementptr inbounds %"class.Eigen::Transpose", %"class.Eigen::Transpose"* %ref.tmp2.i.i.i.i.i.i.i.i, i64 0, i32 0, i32 0, i32 0, i32 0, i32 1
  %m_rows3.i.i = getelementptr inbounds %"class.Eigen::Block", %"class.Eigen::Block"* %ref.tmp3.i.i.i.i.i.i.i.i, i64 0, i32 0, i32 0, i32 0, i32 1
  call void @_ZN5Eigen8internal19variable_if_dynamicIlLi1EEC2ERKS2_(%"class.Eigen::internal::variable_if_dynamic.46"* nonnull %m_rows.i.i24, %"class.Eigen::internal::variable_if_dynamic.46"* nonnull dereferenceable(1) %m_rows3.i.i) #6
  %tmp1.i27 = getelementptr inbounds %"class.Eigen::Block", %"class.Eigen::Block"* %ref.tmp3.i.i.i.i.i.i.i.i, i64 0, i32 0, i32 0, i32 0, i32 2, i32 0
  %tmp4.i.i28 = getelementptr inbounds %"class.Eigen::Transpose", %"class.Eigen::Transpose"* %ref.tmp2.i.i.i.i.i.i.i.i, i64 0, i32 0, i32 0, i32 0, i32 0, i32 2, i32 0
  %tmp5.i.i29 = load i64, i64* %tmp1.i27, align 8, !tbaa !26, !noalias !23
  store i64 %tmp5.i.i29, i64* %tmp4.i.i28, align 8, !tbaa !26, !alias.scope !23
  %m_xpr.i.i.i.i.i = getelementptr inbounds %"class.Eigen::Transpose", %"class.Eigen::Transpose"* %ref.tmp2.i.i.i.i.i.i.i.i, i64 0, i32 0, i32 0, i32 0, i32 1
  %tmp2.i.i.i.i.i = bitcast %"class.Eigen::Matrix"** %m_xpr.i.i.i.i.i to i8*
  %tmp3.i.i.i.i.i = getelementptr inbounds %"class.Eigen::Block", %"class.Eigen::Block"* %ref.tmp3.i.i.i.i.i.i.i.i, i64 0, i32 0, i32 0, i32 1
  %tmp10 = bitcast %"class.Eigen::Matrix"** %tmp3.i.i.i.i.i to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 %tmp2.i.i.i.i.i, i8* nonnull align 8 %tmp10, i64 32, i1 false) #6
  %tmp6.i.i.i.i.i.i.i.i = bitcast %"class.Eigen::Block.47"* %ref.tmp4.i.i.i.i.i.i.i.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 56, i8* nonnull %tmp6.i.i.i.i.i.i.i.i) #6
  %m_data.i.i.i.i.i = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %rhs, i64 0, i32 0, i32 0, i32 0
  %tmp.i.i.i.i.i3 = load double*, double** %m_data.i.i.i.i.i, align 8, !tbaa !2, !noalias !27
  %m_rows.i.i.i.i.i.i.i.i.i = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %rhs, i64 0, i32 0, i32 0, i32 1
  %tmp.i.i.i.i.i.i.i.i.i4 = load i64, i64* %m_rows.i.i.i.i.i.i.i.i.i, align 8, !tbaa !11, !noalias !27
  %mul.i.i.i.i = mul nsw i64 %tmp.i.i.i.i.i.i.i.i.i4, %outer.022.i.i.i.i.i
  %add.ptr.i.i.i.i = getelementptr inbounds double, double* %tmp.i.i.i.i.i3, i64 %mul.i.i.i.i
  %m_data.i2.i.i.i.i = getelementptr inbounds %"class.Eigen::Block.47", %"class.Eigen::Block.47"* %ref.tmp4.i.i.i.i.i.i.i.i, i64 0, i32 0, i32 0, i32 0, i32 0
  store double* %add.ptr.i.i.i.i, double** %m_data.i2.i.i.i.i, align 8, !tbaa !30, !alias.scope !27
  %tmp26 = getelementptr inbounds %"class.Eigen::Block.47", %"class.Eigen::Block.47"* %ref.tmp4.i.i.i.i.i.i.i.i, i64 0, i32 0, i32 0, i32 0, i32 1, i32 0
  store i64 %tmp.i.i.i.i.i.i.i.i.i4, i64* %tmp26, align 8, !tbaa !12, !alias.scope !27
  %m_xpr.i.i.i.i = getelementptr inbounds %"class.Eigen::Block.47", %"class.Eigen::Block.47"* %ref.tmp4.i.i.i.i.i.i.i.i, i64 0, i32 0, i32 0, i32 1
  store %"class.Eigen::Matrix"* %rhs, %"class.Eigen::Matrix"** %m_xpr.i.i.i.i, align 8, !tbaa !16, !alias.scope !27
  %m_value.i.i.i.i.i = getelementptr inbounds %"class.Eigen::Block.47", %"class.Eigen::Block.47"* %ref.tmp4.i.i.i.i.i.i.i.i, i64 0, i32 0, i32 0, i32 2, i32 0
  store i64 0, i64* %m_value.i.i.i.i.i, align 8, !tbaa !12, !alias.scope !27
  %m_value.i2.i.i.i.i = getelementptr inbounds %"class.Eigen::Block.47", %"class.Eigen::Block.47"* %ref.tmp4.i.i.i.i.i.i.i.i, i64 0, i32 0, i32 0, i32 3, i32 0
  store i64 %outer.022.i.i.i.i.i, i64* %m_value.i2.i.i.i.i, align 8, !tbaa !12, !alias.scope !27
  %m_xpr.i = getelementptr inbounds %"class.Eigen::Block.47", %"class.Eigen::Block.47"* %ref.tmp4.i.i.i.i.i.i.i.i, i64 0, i32 0, i32 0, i32 1
  %tmp.i26 = bitcast %"class.Eigen::Matrix"** %m_xpr.i to %"struct.Eigen::EigenBase"**
  %tmp12.i = load %"struct.Eigen::EigenBase"*, %"struct.Eigen::EigenBase"** %tmp.i26, align 8, !tbaa !32
  %m_rows.i.i.i.i.i.i = getelementptr inbounds %"struct.Eigen::EigenBase", %"struct.Eigen::EigenBase"* %tmp12.i, i64 8
  %tmp.i.i.i.i27 = bitcast %"struct.Eigen::EigenBase"* %m_rows.i.i.i.i.i.i to i64*
  %tmp.i.i.i.i.i.i28 = load i64, i64* %tmp.i.i.i.i27, align 8, !tbaa !11
  %m_outerStride.i = getelementptr inbounds %"class.Eigen::Block.47", %"class.Eigen::Block.47"* %ref.tmp4.i.i.i.i.i.i.i.i, i64 0, i32 0, i32 0, i32 4
  store i64 %tmp.i.i.i.i.i.i28, i64* %m_outerStride.i, align 8, !tbaa !34
  %tmp.i5 = bitcast %"class.Eigen::Transpose"* %ref.tmp2.i.i.i.i.i.i.i.i to i64*
  %tmp1.i6 = load i64, i64* %tmp.i5, align 8, !tbaa !20
  %tmp2.i7 = bitcast %"class.Eigen::CwiseBinaryOp.30"* %ref.tmp.i.i.i.i.i.i.i.i to i64*
  store i64 %tmp1.i6, i64* %tmp2.i7, align 8, !tbaa !20
  %m_rows.i = getelementptr inbounds %"class.Eigen::CwiseBinaryOp.30", %"class.Eigen::CwiseBinaryOp.30"* %ref.tmp.i.i.i.i.i.i.i.i, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1
  %m_rows3.i = getelementptr inbounds %"class.Eigen::Transpose", %"class.Eigen::Transpose"* %ref.tmp2.i.i.i.i.i.i.i.i, i64 0, i32 0, i32 0, i32 0, i32 0, i32 1
  call void @_ZN5Eigen8internal19variable_if_dynamicIlLi1EEC2ERKS2_(%"class.Eigen::internal::variable_if_dynamic.46"* nonnull %m_rows.i, %"class.Eigen::internal::variable_if_dynamic.46"* nonnull dereferenceable(1) %m_rows3.i) #6
  %tmp3.i8 = getelementptr inbounds %"class.Eigen::Transpose", %"class.Eigen::Transpose"* %ref.tmp2.i.i.i.i.i.i.i.i, i64 0, i32 0, i32 0, i32 0, i32 0, i32 2, i32 0
  %tmp4.i9 = getelementptr inbounds %"class.Eigen::CwiseBinaryOp.30", %"class.Eigen::CwiseBinaryOp.30"* %ref.tmp.i.i.i.i.i.i.i.i, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 2, i32 0
  %tmp5.i10 = load i64, i64* %tmp3.i8, align 8, !tbaa !26
  store i64 %tmp5.i10, i64* %tmp4.i9, align 8, !tbaa !26
  %m_xpr.i.i.i.i.i.i.i.i.i.i.i.i.i.i = getelementptr inbounds %"class.Eigen::CwiseBinaryOp.30", %"class.Eigen::CwiseBinaryOp.30"* %ref.tmp.i.i.i.i.i.i.i.i, i64 0, i32 0, i32 0, i32 0, i32 0, i32 1
  %m_xpr2.i.i.i.i.i.i.i.i.i.i.i.i.i.i = getelementptr inbounds %"class.Eigen::Transpose", %"class.Eigen::Transpose"* %ref.tmp2.i.i.i.i.i.i.i.i, i64 0, i32 0, i32 0, i32 0, i32 1
  %tmp2.i.i.i.i.i.i.i.i.i.i.i.i.i.i = bitcast %"class.Eigen::Matrix"** %m_xpr.i.i.i.i.i.i.i.i.i.i.i.i.i.i to i8*
  %tmp3.i.i.i.i.i.i.i.i.i.i.i.i.i.i = bitcast %"class.Eigen::Matrix"** %m_xpr2.i.i.i.i.i.i.i.i.i.i.i.i.i.i to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 %tmp2.i.i.i.i.i.i.i.i.i.i.i.i.i.i, i8* nonnull align 8 %tmp3.i.i.i.i.i.i.i.i.i.i.i.i.i.i, i64 32, i1 false) #6
  %m_rhs.i.i.i.i.i.i.i.i.i.i = getelementptr inbounds %"class.Eigen::CwiseBinaryOp.30", %"class.Eigen::CwiseBinaryOp.30"* %ref.tmp.i.i.i.i.i.i.i.i, i64 0, i32 1
  %tmp.i.i.i.i1.i.i.i.i.i.i.i.i.i.i = bitcast %"class.Eigen::Block.47"* %m_rhs.i.i.i.i.i.i.i.i.i.i to i8*
  %tmp1.i.i.i.i2.i.i.i.i.i.i.i.i.i.i = bitcast %"class.Eigen::Block.47"* %ref.tmp4.i.i.i.i.i.i.i.i to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %tmp.i.i.i.i1.i.i.i.i.i.i.i.i.i.i, i8* nonnull align 8 %tmp1.i.i.i.i2.i.i.i.i.i.i.i.i.i.i, i64 16, i1 false) #6
  %tmp.i.i.i.i.i.i.i.i.i.i.i.i.i = getelementptr inbounds %"class.Eigen::CwiseBinaryOp.30", %"class.Eigen::CwiseBinaryOp.30"* %ref.tmp.i.i.i.i.i.i.i.i, i64 0, i32 1, i32 0, i32 0, i32 0, i32 2
  %tmp1.i.i.i.i.i.i.i.i.i.i.i.i.i = getelementptr inbounds %"class.Eigen::Block.47", %"class.Eigen::Block.47"* %ref.tmp4.i.i.i.i.i.i.i.i, i64 0, i32 0, i32 0, i32 0, i32 2
  call void @_ZN5Eigen8internal19variable_if_dynamicIlLi1EEC2ERKS2_(%"class.Eigen::internal::variable_if_dynamic.46"* nonnull %tmp.i.i.i.i.i.i.i.i.i.i.i.i.i, %"class.Eigen::internal::variable_if_dynamic.46"* nonnull dereferenceable(1) %tmp1.i.i.i.i.i.i.i.i.i.i.i.i.i) #6
  %m_xpr.i.i.i.i.i.i.i.i.i.i.i.i.i = getelementptr inbounds %"class.Eigen::CwiseBinaryOp.30", %"class.Eigen::CwiseBinaryOp.30"* %ref.tmp.i.i.i.i.i.i.i.i, i64 0, i32 1, i32 0, i32 0, i32 1
  %m_xpr2.i.i.i.i.i.i.i.i.i.i.i.i.i = getelementptr inbounds %"class.Eigen::Block.47", %"class.Eigen::Block.47"* %ref.tmp4.i.i.i.i.i.i.i.i, i64 0, i32 0, i32 0, i32 1
  %tmp2.i.i.i.i.i.i.i.i.i.i.i.i.i = bitcast %"class.Eigen::Matrix"** %m_xpr.i.i.i.i.i.i.i.i.i.i.i.i.i to i8*
  %tmp3.i.i.i.i.i.i.i.i.i.i.i.i.i = bitcast %"class.Eigen::Matrix"** %m_xpr2.i.i.i.i.i.i.i.i.i.i.i.i.i to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 %tmp2.i.i.i.i.i.i.i.i.i.i.i.i.i, i8* nonnull align 8 %tmp3.i.i.i.i.i.i.i.i.i.i.i.i.i, i64 32, i1 false) #6
  %tmp6 = getelementptr inbounds %"class.Eigen::CwiseBinaryOp.30", %"class.Eigen::CwiseBinaryOp.30"* %ref.tmp.i.i.i.i.i.i.i.i, i64 0, i32 1, i32 0, i32 0, i32 0, i32 1, i32 0
  %tmp.i.i.i.i.i24 = load i64, i64* %tmp6, align 8, !tbaa !12
  %cmp.i32 = icmp eq i64 %tmp.i.i.i.i.i24, 0
  br i1 %cmp.i32, label %_ZNK5Eigen9DenseBaseINS_13CwiseBinaryOpINS_8internal17scalar_product_opIddEEKNS_9TransposeIKNS_5BlockIKNS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1ELin1ELb0EEEEEKNS6_IS9_Lin1ELi1ELb1EEEEEE3sumEv.exit, label %if.end.i35

if.end.i35:                                       ; preds = %for.body5.i.i.i.i.i
  %tmp1.i33 = bitcast %"class.Eigen::CwiseBinaryOp.30"* %ref.tmp.i.i.i.i.i.i.i.i to %"class.Eigen::DenseBase.33"*
  %tmp2.i34 = getelementptr inbounds %"struct.Eigen::internal::scalar_sum_op", %"struct.Eigen::internal::scalar_sum_op"* %ref.tmp.i30, i64 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %tmp2.i34) #6
  %call3.i = call double @_ZNK5Eigen9DenseBaseINS_13CwiseBinaryOpINS_8internal17scalar_product_opIddEEKNS_9TransposeIKNS_5BlockIKNS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1ELin1ELb0EEEEEKNS6_IS9_Lin1ELi1ELb1EEEEEE5reduxINS2_13scalar_sum_opIddEEEEdRKT_(%"class.Eigen::DenseBase.33"* nonnull %tmp1.i33, %"struct.Eigen::internal::scalar_sum_op"* nonnull dereferenceable(1) %ref.tmp.i30) #6
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %tmp2.i34) #6
  br label %_ZNK5Eigen9DenseBaseINS_13CwiseBinaryOpINS_8internal17scalar_product_opIddEEKNS_9TransposeIKNS_5BlockIKNS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1ELin1ELb0EEEEEKNS6_IS9_Lin1ELi1ELb1EEEEEE3sumEv.exit

_ZNK5Eigen9DenseBaseINS_13CwiseBinaryOpINS_8internal17scalar_product_opIddEEKNS_9TransposeIKNS_5BlockIKNS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1ELin1ELb0EEEEEKNS6_IS9_Lin1ELi1ELb1EEEEEE3sumEv.exit: ; preds = %if.end.i35, %for.body5.i.i.i.i.i
  %retval.0.i = phi double [ %call3.i, %if.end.i35 ], [ 0.000000e+00, %for.body5.i.i.i.i.i ]
  call void @llvm.lifetime.end.p0i8(i64 56, i8* nonnull %tmp6.i.i.i.i.i.i.i.i) #6
  call void @llvm.lifetime.end.p0i8(i64 56, i8* nonnull %tmp2.i.i.i.i.i.i.i.i) #6
  call void @llvm.lifetime.end.p0i8(i64 56, i8* nonnull %tmp1.i.i.i.i.i.i.i.i) #6
  call void @llvm.lifetime.end.p0i8(i64 120, i8* nonnull %tmp.i.i.i.i.i.i.i.i) #6
  store double %retval.0.i, double* %arrayidx.i, align 8, !tbaa !17
  %inc.i.i.i.i.i = add nuw nsw i64 %inner.019.i.i.i.i.i, 1
  %m_rows.i.i.i.i.i23 = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %dst, i64 0, i32 0, i32 0, i32 1
  %tmp.i.i.i.i.i25 = load i64, i64* %m_rows.i.i.i.i.i23, align 8, !tbaa !11
  %cmp3.i.i.i.i.i = icmp slt i64 %inc.i.i.i.i.i, %tmp.i.i.i.i.i25
  br i1 %cmp3.i.i.i.i.i, label %for.body5.i.i.i.i.i, label %for.cond.cleanup4.i.i.i.i.i

if.else:                                          ; preds = %entry
  %tmp2 = bitcast %"class.Eigen::Matrix"* %dst to %"class.Eigen::DenseBase"*
  %tmp.i18 = bitcast double* %ref.tmp.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %tmp.i18) #6
  store double 0.000000e+00, double* %ref.tmp.i, align 8, !tbaa !17
  %call.i19 = call dereferenceable(24) %"class.Eigen::Matrix"* @_ZN5Eigen9DenseBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEE11setConstantERKd(%"class.Eigen::DenseBase"* %tmp2, double* nonnull dereferenceable(8) %ref.tmp.i)
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %tmp.i18) #6
  %m_cols.i.i.i9 = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %lhs, i64 0, i32 0, i32 0, i32 2
  %tmp.i.i.i10 = load i64, i64* %m_cols.i.i.i9, align 8, !tbaa !19
  %cmp.i = icmp eq i64 %tmp.i.i.i10, 0
  br i1 %cmp.i, label %if.end, label %if.end.i

if.end.i:                                         ; preds = %if.else
  %tmp4.i14 = bitcast %"class.Eigen::internal::gemm_blocking_space"* %blocking.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 56, i8* nonnull %tmp4.i14) #6
  %tmp7.i15 = bitcast %"struct.Eigen::internal::gemm_functor"* %ref.tmp.i8 to i8*
  call void @llvm.lifetime.start.p0i8(i64 40, i8* nonnull %tmp7.i15) #6
  %m_lhs.i8 = getelementptr inbounds %"struct.Eigen::internal::gemm_functor", %"struct.Eigen::internal::gemm_functor"* %ref.tmp.i8, i64 0, i32 0
  store %"class.Eigen::Matrix"* %lhs, %"class.Eigen::Matrix"** %m_lhs.i8, align 8, !tbaa !16
  %m_rhs.i9 = getelementptr inbounds %"struct.Eigen::internal::gemm_functor", %"struct.Eigen::internal::gemm_functor"* %ref.tmp.i8, i64 0, i32 1
  store %"class.Eigen::Matrix"* %rhs, %"class.Eigen::Matrix"** %m_rhs.i9, align 8, !tbaa !16
  %m_dest.i = getelementptr inbounds %"struct.Eigen::internal::gemm_functor", %"struct.Eigen::internal::gemm_functor"* %ref.tmp.i8, i64 0, i32 2
  store %"class.Eigen::Matrix"* %dst, %"class.Eigen::Matrix"** %m_dest.i, align 8, !tbaa !16
  %m_actualAlpha.i = getelementptr inbounds %"struct.Eigen::internal::gemm_functor", %"struct.Eigen::internal::gemm_functor"* %ref.tmp.i8, i64 0, i32 3
  %tmp2.i13 = bitcast double* %m_actualAlpha.i to i64*
  store i64 4607182418800017408, i64* %tmp2.i13, align 8, !tbaa !35
  %m_blocking.i = getelementptr inbounds %"struct.Eigen::internal::gemm_functor", %"struct.Eigen::internal::gemm_functor"* %ref.tmp.i8, i64 0, i32 4
  store %"class.Eigen::internal::gemm_blocking_space"* %blocking.i, %"class.Eigen::internal::gemm_blocking_space"** %m_blocking.i, align 8, !tbaa !16
  %m_rows.i.i13.i = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %lhs, i64 0, i32 0, i32 0, i32 1
  %tmp.i.i14.i = load i64, i64* %m_rows.i.i13.i, align 8, !tbaa !11
  %m_cols.i.i10.i = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %rhs, i64 0, i32 0, i32 0, i32 2
  %tmp.i.i11.i = load i64, i64* %m_cols.i.i10.i, align 8, !tbaa !19
  call void @_ZNK5Eigen8internal12gemm_functorIdlNS0_29general_matrix_matrix_productIldLi0ELb0EdLi0ELb0ELi0EEENS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEES5_S5_NS0_19gemm_blocking_spaceILi0EddLin1ELin1ELin1ELi1ELb0EEEEclEllllPNS0_16GemmParallelInfoIlEE(%"struct.Eigen::internal::gemm_functor"* nonnull %ref.tmp.i8, i64 0, i64 %tmp.i.i14.i, i64 0, i64 %tmp.i.i11.i, %"struct.Eigen::internal::GemmParallelInfo"* null)
  call void @llvm.lifetime.end.p0i8(i64 40, i8* nonnull %tmp7.i15) #6
  %tmp.i4 = bitcast %"class.Eigen::internal::gemm_blocking_space"* %blocking.i to i8**
  %tmp1.i = load i8*, i8** %tmp.i4, align 8, !tbaa !37
  call void @free(i8* %tmp1.i) #6
  %m_blockB.i = getelementptr inbounds %"class.Eigen::internal::gemm_blocking_space", %"class.Eigen::internal::gemm_blocking_space"* %blocking.i, i64 0, i32 0, i32 1
  %tmp2.i = bitcast double** %m_blockB.i to i8**
  %tmp32.i = load i8*, i8** %tmp2.i, align 8, !tbaa !39
  call void @free(i8* %tmp32.i) #6
  call void @llvm.lifetime.end.p0i8(i64 56, i8* nonnull %tmp4.i14) #6
  br label %if.end

if.end:                                           ; preds = %if.end.i, %if.else, %for.cond.cleanup4.i.i.i.i.i
  ret void
}

; Function Attrs: inlinehint norecurse nounwind uwtable
define linkonce_odr dso_local void @_ZN5Eigen8internal15BlockImpl_denseIKNS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1ELin1ELb0ELb1EEC2ERS4_l(%"class.Eigen::internal::BlockImpl_dense"* %this, %"class.Eigen::Matrix"* dereferenceable(24) %xpr, i64 %i) unnamed_addr #3 comdat align 2 {
entry:
  %m_data.i = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %xpr, i64 0, i32 0, i32 0, i32 0
  %tmp.i = load double*, double** %m_data.i, align 8, !tbaa !2
  %add.ptr = getelementptr inbounds double, double* %tmp.i, i64 %i
  %m_cols.i.i = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %xpr, i64 0, i32 0, i32 0, i32 2
  %tmp.i.i = load i64, i64* %m_cols.i.i, align 8, !tbaa !19
  %m_data.i3 = getelementptr inbounds %"class.Eigen::internal::BlockImpl_dense", %"class.Eigen::internal::BlockImpl_dense"* %this, i64 0, i32 0, i32 0
  store double* %add.ptr, double** %m_data.i3, align 8, !tbaa !20
  %m_value.i.i = getelementptr inbounds %"class.Eigen::internal::BlockImpl_dense", %"class.Eigen::internal::BlockImpl_dense"* %this, i64 0, i32 0, i32 2, i32 0
  store i64 %tmp.i.i, i64* %m_value.i.i, align 8, !tbaa !12
  %m_xpr = getelementptr inbounds %"class.Eigen::internal::BlockImpl_dense", %"class.Eigen::internal::BlockImpl_dense"* %this, i64 0, i32 1
  store %"class.Eigen::Matrix"* %xpr, %"class.Eigen::Matrix"** %m_xpr, align 8, !tbaa !16
  %m_value.i = getelementptr inbounds %"class.Eigen::internal::BlockImpl_dense", %"class.Eigen::internal::BlockImpl_dense"* %this, i64 0, i32 2, i32 0
  store i64 %i, i64* %m_value.i, align 8, !tbaa !12
  %m_value.i2 = getelementptr inbounds %"class.Eigen::internal::BlockImpl_dense", %"class.Eigen::internal::BlockImpl_dense"* %this, i64 0, i32 3, i32 0
  store i64 0, i64* %m_value.i2, align 8, !tbaa !12
  %m_outerStride.i = getelementptr inbounds %"class.Eigen::internal::BlockImpl_dense", %"class.Eigen::internal::BlockImpl_dense"* %this, i64 0, i32 4
  store i64 1, i64* %m_outerStride.i, align 8, !tbaa !40
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1) #0

; Function Attrs: inlinehint norecurse nounwind uwtable
define linkonce_odr dso_local void @_ZN5Eigen8internal19variable_if_dynamicIlLi1EEC2ERKS2_(%"class.Eigen::internal::variable_if_dynamic.46"* %this, %"class.Eigen::internal::variable_if_dynamic.46"* dereferenceable(1) %arg) unnamed_addr #3 comdat align 2 {
entry:
  ret void
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local double @_ZNK5Eigen9DenseBaseINS_13CwiseBinaryOpINS_8internal17scalar_product_opIddEEKNS_9TransposeIKNS_5BlockIKNS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1ELin1ELb0EEEEEKNS6_IS9_Lin1ELi1ELb1EEEEEE5reduxINS2_13scalar_sum_opIddEEEEdRKT_(%"class.Eigen::DenseBase.33"* %this, %"struct.Eigen::internal::scalar_sum_op"* dereferenceable(1) %func) local_unnamed_addr #4 comdat align 2 {
entry:
  %tmp = bitcast %"class.Eigen::DenseBase.33"* %this to i64*
  %tmp.i2.i.i.i.i.i.i.i.i.i.i113 = load i64, i64* %tmp, align 8, !tbaa !20
  %m_xpr.i.i.i.i.i.i.i.i.i.i.i = getelementptr inbounds %"class.Eigen::DenseBase.33", %"class.Eigen::DenseBase.33"* %this, i64 24
  %tmp.i1.i.i.i.i.i.i.i.i.i.i = bitcast %"class.Eigen::DenseBase.33"* %m_xpr.i.i.i.i.i.i.i.i.i.i.i to %"struct.Eigen::EigenBase"**
  %tmp12.i.i.i.i.i.i.i.i.i.i.i = load %"struct.Eigen::EigenBase"*, %"struct.Eigen::EigenBase"** %tmp.i1.i.i.i.i.i.i.i.i.i.i, align 8, !tbaa !42
  %m_rows.i.i.i.i.i.i.i.i.i.i.i.i.i.i.i.i = getelementptr inbounds %"struct.Eigen::EigenBase", %"struct.Eigen::EigenBase"* %tmp12.i.i.i.i.i.i.i.i.i.i.i, i64 8
  %tmp.i.i.i.i.i.i.i.i.i.i.i.i.i.i = bitcast %"struct.Eigen::EigenBase"* %m_rows.i.i.i.i.i.i.i.i.i.i.i.i.i.i.i.i to i64*
  %tmp.i.i.i.i.i.i.i.i.i.i.i.i.i.i.i.i = load i64, i64* %tmp.i.i.i.i.i.i.i.i.i.i.i.i.i.i, align 8, !tbaa !11
  %m_rhs.i.i.i.i = getelementptr inbounds %"class.Eigen::DenseBase.33", %"class.Eigen::DenseBase.33"* %this, i64 56
  %tmp1 = bitcast %"class.Eigen::DenseBase.33"* %m_rhs.i.i.i.i to i64*
  %tmp.i1.i.i.i.i.i.i.i114 = load i64, i64* %tmp1, align 8, !tbaa !30
  %tmp4 = inttoptr i64 %tmp.i2.i.i.i.i.i.i.i.i.i.i113 to double*
  %tmp1.i.i.i95115126 = load double, double* %tmp4, align 8, !tbaa !17
  %tmp5 = inttoptr i64 %tmp.i1.i.i.i.i.i.i.i114 to double*
  %tmp1.i6.i102117119 = load double, double* %tmp5, align 8, !tbaa !17
  %mul.i.i105 = fmul double %tmp1.i.i.i95115126, %tmp1.i6.i102117119
  %m_value.i.i.i.i.i.i.i7912 = getelementptr inbounds %"class.Eigen::DenseBase.33", %"class.Eigen::DenseBase.33"* %this, i64 64
  %tmp7 = bitcast %"class.Eigen::DenseBase.33"* %m_value.i.i.i.i.i.i.i7912 to i64*
  %tmp.i.i.i.i.i.i.i80 = load i64, i64* %tmp7, align 8, !tbaa !12
  %cmp46.i = icmp sgt i64 %tmp.i.i.i.i.i.i.i80, 1
  br i1 %cmp46.i, label %for.body.i, label %_ZN5Eigen8internal10redux_implINS0_13scalar_sum_opIddEENS0_15redux_evaluatorINS_13CwiseBinaryOpINS0_17scalar_product_opIddEEKNS_9TransposeIKNS_5BlockIKNS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1ELin1ELb0EEEEEKNS9_ISC_Lin1ELi1ELb1EEEEEEELi0ELi0EE3runERKSK_RKS3_.exit

for.body.i:                                       ; preds = %for.body.i, %entry
  %res.i.0 = phi double [ %mul.i.i105, %entry ], [ %add.i43, %for.body.i ]
  %i.047.i = phi i64 [ 1, %entry ], [ %inc.i, %for.body.i ]
  %tmp8 = inttoptr i64 %tmp.i2.i.i.i.i.i.i.i.i.i.i113 to double*
  %mul.i.i.i55 = mul nsw i64 %tmp.i.i.i.i.i.i.i.i.i.i.i.i.i.i.i.i, %i.047.i
  %arrayidx.i.i.i56 = getelementptr inbounds double, double* %tmp8, i64 %mul.i.i.i55
  %tmp1.i.i.i57122128 = load double, double* %arrayidx.i.i.i56, align 8, !tbaa !17
  %tmp9 = inttoptr i64 %tmp.i1.i.i.i.i.i.i.i114 to double*
  %arrayidx.i.i64 = getelementptr inbounds double, double* %tmp9, i64 %i.047.i
  %tmp1.i6.i65123124 = load double, double* %arrayidx.i.i64, align 8, !tbaa !17
  %mul.i.i68 = fmul double %tmp1.i.i.i57122128, %tmp1.i6.i65123124
  %add.i43 = fadd double %res.i.0, %mul.i.i68
  %inc.i = add nuw nsw i64 %i.047.i, 1
  %m_value.i.i.i.i.i.i.i3613 = getelementptr inbounds %"class.Eigen::DenseBase.33", %"class.Eigen::DenseBase.33"* %this, i64 64
  %tmp11 = bitcast %"class.Eigen::DenseBase.33"* %m_value.i.i.i.i.i.i.i3613 to i64*
  %tmp.i.i.i.i.i.i.i37 = load i64, i64* %tmp11, align 8, !tbaa !12
  %cmp.i = icmp slt i64 %inc.i, %tmp.i.i.i.i.i.i.i37
  br i1 %cmp.i, label %for.body.i, label %_ZN5Eigen8internal10redux_implINS0_13scalar_sum_opIddEENS0_15redux_evaluatorINS_13CwiseBinaryOpINS0_17scalar_product_opIddEEKNS_9TransposeIKNS_5BlockIKNS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1ELin1ELb0EEEEEKNS9_ISC_Lin1ELi1ELb1EEEEEEELi0ELi0EE3runERKSK_RKS3_.exit

_ZN5Eigen8internal10redux_implINS0_13scalar_sum_opIddEENS0_15redux_evaluatorINS_13CwiseBinaryOpINS0_17scalar_product_opIddEEKNS_9TransposeIKNS_5BlockIKNS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1ELin1ELb0EEEEEKNS9_ISC_Lin1ELi1ELb1EEEEEEELi0ELi0EE3runERKSK_RKS3_.exit: ; preds = %for.body.i, %entry
  %res.i.1 = phi double [ %add.i43, %for.body.i ], [ %mul.i.i105, %entry ]
  ret double %res.i.1
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local dereferenceable(24) %"class.Eigen::Matrix"* @_ZN5Eigen9DenseBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEE11setConstantERKd(%"class.Eigen::DenseBase"* %this, double* dereferenceable(8) %val) local_unnamed_addr #4 comdat align 2 {
entry:
  %m_rows.i.i.i6 = getelementptr inbounds %"class.Eigen::DenseBase", %"class.Eigen::DenseBase"* %this, i64 8
  %tmp.i = bitcast %"class.Eigen::DenseBase"* %m_rows.i.i.i6 to i64*
  %tmp.i.i.i1 = load i64, i64* %tmp.i, align 8, !tbaa !11
  %m_cols.i.i.i25 = getelementptr inbounds %"class.Eigen::DenseBase", %"class.Eigen::DenseBase"* %this, i64 16
  %tmp.i1 = bitcast %"class.Eigen::DenseBase"* %m_cols.i.i.i25 to i64*
  %tmp.i.i.i2 = load i64, i64* %tmp.i1, align 8, !tbaa !19
  %tmp.i.i14 = bitcast double* %val to i64*
  %tmp1.i.i15 = load i64, i64* %tmp.i.i14, align 8, !tbaa !17, !noalias !43
  %tmp.i16 = bitcast %"class.Eigen::DenseBase"* %this to %"class.Eigen::Matrix"*
  %m_rows.i.i = getelementptr inbounds %"class.Eigen::DenseBase", %"class.Eigen::DenseBase"* %this, i64 8
  %tmp = bitcast %"class.Eigen::DenseBase"* %m_rows.i.i to i64*
  %tmp.i.i1 = load i64, i64* %tmp, align 8, !tbaa !11
  %cmp.i.i.i.i.i.i.i.i = icmp eq i64 %tmp.i.i1, %tmp.i.i.i1
  br i1 %cmp.i.i.i.i.i.i.i.i, label %lor.lhs.false.i.i.i.i.i.i.i.i, label %if.then.i.i.i.i.i.i.i.i

lor.lhs.false.i.i.i.i.i.i.i.i:                    ; preds = %entry
  %m_cols.i.i = getelementptr inbounds %"class.Eigen::DenseBase", %"class.Eigen::DenseBase"* %this, i64 16
  %tmp1 = bitcast %"class.Eigen::DenseBase"* %m_cols.i.i to i64*
  %tmp.i.i3 = load i64, i64* %tmp1, align 8, !tbaa !19
  %cmp4.i.i.i.i.i.i.i.i = icmp eq i64 %tmp.i.i3, %tmp.i.i.i2
  br i1 %cmp4.i.i.i.i.i.i.i.i, label %_ZN5Eigen8internal17resize_if_allowedINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EEddEEvRT_RKT0_RKNS0_9assign_opIT1_T2_EE.exit.i.i.i.i.i.i.i, label %if.then.i.i.i.i.i.i.i.i

if.then.i.i.i.i.i.i.i.i:                          ; preds = %lor.lhs.false.i.i.i.i.i.i.i.i, %entry
  %cmp.i.i.i.i.i.i.i.i.i.i = icmp eq i64 %tmp.i.i.i1, 0
  %cmp1.i.i.i.i.i.i.i.i.i.i = icmp eq i64 %tmp.i.i.i2, 0
  %or.cond.i.i.i.i.i.i.i.i.i.i = or i1 %cmp.i.i.i.i.i.i.i.i.i.i, %cmp1.i.i.i.i.i.i.i.i.i.i
  br i1 %or.cond.i.i.i.i.i.i.i.i.i.i, label %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEE6resizeEll.exit.i.i.i.i.i.i.i.i, label %cond.false.i.i.i.i.i.i.i.i.i.i

cond.false.i.i.i.i.i.i.i.i.i.i:                   ; preds = %if.then.i.i.i.i.i.i.i.i
  %div.i.i.i.i.i.i.i.i.i.i = sdiv i64 9223372036854775807, %tmp.i.i.i2
  %cmp2.i.i.i.i.i.i.i.i.i.i = icmp slt i64 %div.i.i.i.i.i.i.i.i.i.i, %tmp.i.i.i1
  br i1 %cmp2.i.i.i.i.i.i.i.i.i.i, label %if.then.i.i.i.i.i.i.i.i.i.i, label %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEE6resizeEll.exit.i.i.i.i.i.i.i.i

if.then.i.i.i.i.i.i.i.i.i.i:                      ; preds = %cond.false.i.i.i.i.i.i.i.i.i.i
  call void @_ZN5Eigen8internal19throw_std_bad_allocEv() #6
  br label %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEE6resizeEll.exit.i.i.i.i.i.i.i.i

_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEE6resizeEll.exit.i.i.i.i.i.i.i.i: ; preds = %if.then.i.i.i.i.i.i.i.i.i.i, %cond.false.i.i.i.i.i.i.i.i.i.i, %if.then.i.i.i.i.i.i.i.i
  %m_storage.i.i.i.i.i.i.i.i.i = bitcast %"class.Eigen::DenseBase"* %this to %"class.Eigen::DenseStorage"*
  %mul.i.i.i.i.i.i.i.i.i = mul nsw i64 %tmp.i.i.i2, %tmp.i.i.i1
  call void @_ZN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EE6resizeElll(%"class.Eigen::DenseStorage"* nonnull %m_storage.i.i.i.i.i.i.i.i.i, i64 %mul.i.i.i.i.i.i.i.i.i, i64 %tmp.i.i.i1, i64 %tmp.i.i.i2) #6
  br label %_ZN5Eigen8internal17resize_if_allowedINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EEddEEvRT_RKT0_RKNS0_9assign_opIT1_T2_EE.exit.i.i.i.i.i.i.i

_ZN5Eigen8internal17resize_if_allowedINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EEddEEvRT_RKT0_RKNS0_9assign_opIT1_T2_EE.exit.i.i.i.i.i.i.i: ; preds = %_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEE6resizeEll.exit.i.i.i.i.i.i.i.i, %lor.lhs.false.i.i.i.i.i.i.i.i
  %tmp2 = bitcast %"class.Eigen::DenseBase"* %this to double**
  %tmp.i275 = load double*, double** %tmp2, align 8, !tbaa !2
  %m_rows.i.i.i.i.i826 = getelementptr inbounds %"class.Eigen::DenseBase", %"class.Eigen::DenseBase"* %this, i64 8
  %tmp.i.i.i9 = bitcast %"class.Eigen::DenseBase"* %m_rows.i.i.i.i.i826 to i64*
  %tmp.i.i.i.i.i10 = load i64, i64* %tmp.i.i.i9, align 8, !tbaa !11
  %m_cols.i.i.i.i.i27 = getelementptr inbounds %"class.Eigen::DenseBase", %"class.Eigen::DenseBase"* %this, i64 16
  %tmp.i1.i.i = bitcast %"class.Eigen::DenseBase"* %m_cols.i.i.i.i.i27 to i64*
  %tmp.i.i.i2.i.i = load i64, i64* %tmp.i1.i.i, align 8, !tbaa !19
  %mul.i.i = mul nsw i64 %tmp.i.i.i2.i.i, %tmp.i.i.i.i.i10
  %cmp6.i.i.i.i.i.i.i.i = icmp sgt i64 %mul.i.i, 0
  br i1 %cmp6.i.i.i.i.i.i.i.i, label %for.body.i.i.i.i.i.i.i.i, label %_ZN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEaSINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES1_EEEERS1_RKNS_9DenseBaseIT_EE.exit

for.body.i.i.i.i.i.i.i.i:                         ; preds = %for.body.i.i.i.i.i.i.i.i, %_ZN5Eigen8internal17resize_if_allowedINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EEddEEvRT_RKT0_RKNS0_9assign_opIT1_T2_EE.exit.i.i.i.i.i.i.i
  %i.07.i.i.i.i.i.i.i.i = phi i64 [ %inc.i.i.i.i.i.i.i.i, %for.body.i.i.i.i.i.i.i.i ], [ 0, %_ZN5Eigen8internal17resize_if_allowedINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EEddEEvRT_RKT0_RKNS0_9assign_opIT1_T2_EE.exit.i.i.i.i.i.i.i ]
  %arrayidx.i = getelementptr inbounds double, double* %tmp.i275, i64 %i.07.i.i.i.i.i.i.i.i
  %tmp2.i = bitcast double* %arrayidx.i to i64*
  store i64 %tmp1.i.i15, i64* %tmp2.i, align 8, !tbaa !17
  %inc.i.i.i.i.i.i.i.i = add nuw nsw i64 %i.07.i.i.i.i.i.i.i.i, 1
  %exitcond.i.i.i.i.i.i.i.i = icmp eq i64 %inc.i.i.i.i.i.i.i.i, %mul.i.i
  br i1 %exitcond.i.i.i.i.i.i.i.i, label %_ZN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEaSINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES1_EEEERS1_RKNS_9DenseBaseIT_EE.exit, label %for.body.i.i.i.i.i.i.i.i

_ZN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEaSINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES1_EEEERS1_RKNS_9DenseBaseIT_EE.exit: ; preds = %for.body.i.i.i.i.i.i.i.i, %_ZN5Eigen8internal17resize_if_allowedINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EEddEEvRT_RKT0_RKNS0_9assign_opIT1_T2_EE.exit.i.i.i.i.i.i.i
  ret %"class.Eigen::Matrix"* %tmp.i16
}

; Function Attrs: inlinehint norecurse nounwind uwtable
define linkonce_odr dso_local dereferenceable(8) i64* @_ZSt3minIlERKT_S2_S2_(i64* dereferenceable(8) %__a, i64* dereferenceable(8) %__b) local_unnamed_addr #3 comdat {
entry:
  %tmp = load i64, i64* %__b, align 8, !tbaa !26
  %tmp1 = load i64, i64* %__a, align 8, !tbaa !26
  %cmp = icmp slt i64 %tmp, %tmp1
  %__b.__a = select i1 %cmp, i64* %__b, i64* %__a
  ret i64* %__b.__a
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZNK5Eigen8internal12gemm_functorIdlNS0_29general_matrix_matrix_productIldLi0ELb0EdLi0ELb0ELi0EEENS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEES5_S5_NS0_19gemm_blocking_spaceILi0EddLin1ELin1ELin1ELi1ELb0EEEEclEllllPNS0_16GemmParallelInfoIlEE(%"struct.Eigen::internal::gemm_functor"* %this, i64 %row, i64 %rows, i64 %col, i64 %cols, %"struct.Eigen::internal::GemmParallelInfo"* %info) local_unnamed_addr #4 comdat align 2 {
entry:
  %tmp2 = bitcast %"struct.Eigen::internal::gemm_functor"* %this to %"class.Eigen::PlainObjectBase"**
  %tmp3 = load %"class.Eigen::PlainObjectBase"*, %"class.Eigen::PlainObjectBase"** %tmp2, align 8, !tbaa !46
  %m_cols.i.i2 = getelementptr inbounds %"class.Eigen::PlainObjectBase", %"class.Eigen::PlainObjectBase"* %tmp3, i64 0, i32 0, i32 2
  %tmp.i.i3 = load i64, i64* %m_cols.i.i2, align 8, !tbaa !19
  %m_data.i.i = getelementptr inbounds %"class.Eigen::PlainObjectBase", %"class.Eigen::PlainObjectBase"* %tmp3, i64 0, i32 0, i32 0
  %tmp.i1.i = load double*, double** %m_data.i.i, align 8, !tbaa !2
  %arrayidx.i = getelementptr inbounds double, double* %tmp.i1.i, i64 %row
  %tmp4 = bitcast %"struct.Eigen::internal::gemm_functor"* %this to %"struct.Eigen::EigenBase"**
  %tmp51314 = load %"struct.Eigen::EigenBase"*, %"struct.Eigen::EigenBase"** %tmp4, align 8, !tbaa !46
  %m_rows.i.i.i.i.i = getelementptr inbounds %"struct.Eigen::EigenBase", %"struct.Eigen::EigenBase"* %tmp51314, i64 8
  %tmp.i.i.i = bitcast %"struct.Eigen::EigenBase"* %m_rows.i.i.i.i.i to i64*
  %tmp.i.i.i.i.i = load i64, i64* %tmp.i.i.i, align 8, !tbaa !11
  %m_rhs7 = getelementptr inbounds %"struct.Eigen::internal::gemm_functor", %"struct.Eigen::internal::gemm_functor"* %this, i64 0, i32 1
  %tmp6 = bitcast %"class.Eigen::Matrix"** %m_rhs7 to %"class.Eigen::PlainObjectBase"**
  %tmp7 = load %"class.Eigen::PlainObjectBase"*, %"class.Eigen::PlainObjectBase"** %tmp6, align 8, !tbaa !47
  %m_data.i.i2 = getelementptr inbounds %"class.Eigen::PlainObjectBase", %"class.Eigen::PlainObjectBase"* %tmp7, i64 0, i32 0, i32 0
  %tmp.i1.i3 = load double*, double** %m_data.i.i2, align 8, !tbaa !2
  %m_rows.i.i4 = getelementptr inbounds %"class.Eigen::PlainObjectBase", %"class.Eigen::PlainObjectBase"* %tmp7, i64 0, i32 0, i32 1
  %tmp.i.i5 = load i64, i64* %m_rows.i.i4, align 8, !tbaa !11
  %mul.i = mul nsw i64 %tmp.i.i5, %col
  %arrayidx.i6 = getelementptr inbounds double, double* %tmp.i1.i3, i64 %mul.i
  %tmp5 = bitcast %"class.Eigen::Matrix"** %m_rhs7 to %"struct.Eigen::EigenBase"**
  %tmp81516 = load %"struct.Eigen::EigenBase"*, %"struct.Eigen::EigenBase"** %tmp5, align 8, !tbaa !47
  %m_rows.i.i.i.i.i5 = getelementptr inbounds %"struct.Eigen::EigenBase", %"struct.Eigen::EigenBase"* %tmp81516, i64 8
  %tmp.i.i.i6 = bitcast %"struct.Eigen::EigenBase"* %m_rows.i.i.i.i.i5 to i64*
  %tmp.i.i.i.i.i7 = load i64, i64* %tmp.i.i.i6, align 8, !tbaa !11
  %m_dest = getelementptr inbounds %"struct.Eigen::internal::gemm_functor", %"struct.Eigen::internal::gemm_functor"* %this, i64 0, i32 2
  %tmp9 = bitcast %"class.Eigen::Matrix"** %m_dest to %"class.Eigen::PlainObjectBase"**
  %tmp10 = load %"class.Eigen::PlainObjectBase"*, %"class.Eigen::PlainObjectBase"** %tmp9, align 8, !tbaa !48
  %m_data.i.i7 = getelementptr inbounds %"class.Eigen::PlainObjectBase", %"class.Eigen::PlainObjectBase"* %tmp10, i64 0, i32 0, i32 0
  %tmp.i1.i8 = load double*, double** %m_data.i.i7, align 8, !tbaa !2
  %m_rows.i.i9 = getelementptr inbounds %"class.Eigen::PlainObjectBase", %"class.Eigen::PlainObjectBase"* %tmp10, i64 0, i32 0, i32 1
  %tmp.i.i10 = load i64, i64* %m_rows.i.i9, align 8, !tbaa !11
  %mul.i11 = mul nsw i64 %tmp.i.i10, %col
  %add.i = add nsw i64 %mul.i11, %row
  %arrayidx.i12 = getelementptr inbounds double, double* %tmp.i1.i8, i64 %add.i
  %tmp8 = bitcast %"class.Eigen::Matrix"** %m_dest to %"struct.Eigen::EigenBase"**
  %tmp111718 = load %"struct.Eigen::EigenBase"*, %"struct.Eigen::EigenBase"** %tmp8, align 8, !tbaa !48
  %m_rows.i.i.i.i.i10 = getelementptr inbounds %"struct.Eigen::EigenBase", %"struct.Eigen::EigenBase"* %tmp111718, i64 8
  %tmp.i.i.i11 = bitcast %"struct.Eigen::EigenBase"* %m_rows.i.i.i.i.i10 to i64*
  %tmp.i.i.i.i.i12 = load i64, i64* %tmp.i.i.i11, align 8, !tbaa !11
  %m_actualAlpha = getelementptr inbounds %"struct.Eigen::internal::gemm_functor", %"struct.Eigen::internal::gemm_functor"* %this, i64 0, i32 3
  %tmp12 = load double, double* %m_actualAlpha, align 8, !tbaa !35
  %m_blocking = getelementptr inbounds %"struct.Eigen::internal::gemm_functor", %"struct.Eigen::internal::gemm_functor"* %this, i64 0, i32 4
  %tmp13 = bitcast %"class.Eigen::internal::gemm_blocking_space"** %m_blocking to %"class.Eigen::internal::level3_blocking"**
  %tmp14 = load %"class.Eigen::internal::level3_blocking"*, %"class.Eigen::internal::level3_blocking"** %tmp13, align 8, !tbaa !49
  tail call void @_ZN5Eigen8internal29general_matrix_matrix_productIldLi0ELb0EdLi0ELb0ELi0EE3runElllPKdlS4_lPdldRNS0_15level3_blockingIddEEPNS0_16GemmParallelInfoIlEE(i64 %rows, i64 %cols, i64 %tmp.i.i3, double* nonnull %arrayidx.i, i64 %tmp.i.i.i.i.i, double* nonnull %arrayidx.i6, i64 %tmp.i.i.i.i.i7, double* nonnull %arrayidx.i12, i64 %tmp.i.i.i.i.i12, double %tmp12, %"class.Eigen::internal::level3_blocking"* dereferenceable(40) %tmp14, %"struct.Eigen::internal::GemmParallelInfo"* %info)
  ret void
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZN5Eigen8internal29general_matrix_matrix_productIldLi0ELb0EdLi0ELb0ELi0EE3runElllPKdlS4_lPdldRNS0_15level3_blockingIddEEPNS0_16GemmParallelInfoIlEE(i64 %rows, i64 %cols, i64 %depth, double* %_lhs, i64 %lhsStride, double* %_rhs, i64 %rhsStride, double* %_res, i64 %resStride, double %alpha, %"class.Eigen::internal::level3_blocking"* dereferenceable(40) %blocking, %"struct.Eigen::internal::GemmParallelInfo"* %info) local_unnamed_addr #4 comdat align 2 {
entry:
  %rows.addr = alloca i64, align 8
  %cols.addr = alloca i64, align 8
  %depth.addr = alloca i64, align 8
  %ref.tmp = alloca i64, align 8
  %pack_lhs = alloca %"struct.Eigen::internal::gemm_pack_lhs", align 1
  %ref.tmp60 = alloca i64, align 8
  %ref.tmp64 = alloca %"class.Eigen::internal::const_blas_data_mapper", align 8
  %ref.tmp70 = alloca i64, align 8
  store i64 %rows, i64* %rows.addr, align 8, !tbaa !26
  store i64 %cols, i64* %cols.addr, align 8, !tbaa !26
  store i64 %depth, i64* %depth.addr, align 8, !tbaa !26
  %m_kc.i = getelementptr inbounds %"class.Eigen::internal::level3_blocking", %"class.Eigen::internal::level3_blocking"* %blocking, i64 0, i32 4
  %tmp.i1 = load i64, i64* %m_kc.i, align 8, !tbaa !50
  %m_mc.i = getelementptr inbounds %"class.Eigen::internal::level3_blocking", %"class.Eigen::internal::level3_blocking"* %blocking, i64 0, i32 2
  %tmp.i2 = load i64, i64* %m_mc.i, align 8, !tbaa !51
  store i64 %tmp.i2, i64* %ref.tmp, align 8, !tbaa !26
  %call2 = call dereferenceable(8) i64* @_ZSt3minIlERKT_S2_S2_(i64* nonnull dereferenceable(8) %rows.addr, i64* nonnull dereferenceable(8) %ref.tmp)
  %tmp8 = load i64, i64* %rows.addr, align 8, !tbaa !26
  %tmp10 = load i64, i64* %cols.addr, align 8, !tbaa !26
  %tmp11 = getelementptr inbounds %"struct.Eigen::internal::gemm_pack_lhs", %"struct.Eigen::internal::gemm_pack_lhs"* %pack_lhs, i64 0, i32 0
  %mul = mul nsw i64 %tmp8, %tmp.i1
  %m_blockA.i6 = getelementptr inbounds %"class.Eigen::internal::level3_blocking", %"class.Eigen::internal::level3_blocking"* %blocking, i64 0, i32 0
  %tmp.i7 = load double*, double** %m_blockA.i6, align 8, !tbaa !37
  %mul24 = shl i64 %mul, 3
  %cmp25 = icmp ult i64 %mul24, 131073
  %tmp28 = getelementptr inbounds %"class.Eigen::internal::const_blas_data_mapper", %"class.Eigen::internal::const_blas_data_mapper"* %ref.tmp64, i64 0, i32 0, i32 0
  %tmp29 = getelementptr inbounds %"class.Eigen::internal::const_blas_data_mapper", %"class.Eigen::internal::const_blas_data_mapper"* %ref.tmp64, i64 0, i32 0, i32 1
  br label %for.body

for.cond.loopexit:                                ; preds = %for.body59
  %tmp37 = load i64, i64* %rows.addr, align 8, !tbaa !26
  %cmp53 = icmp slt i64 %add, %tmp37
  br i1 %cmp53, label %for.body, label %_ZN5Eigen8internal28aligned_stack_memory_handlerIdED2Ev.exit

_ZN5Eigen8internal28aligned_stack_memory_handlerIdED2Ev.exit: ; preds = %for.cond.loopexit
  br i1 %cmp25, label %_ZN5Eigen8internal28aligned_stack_memory_handlerIdED2Ev.exit10, label %if.then.i8

if.then.i8:                                       ; preds = %_ZN5Eigen8internal28aligned_stack_memory_handlerIdED2Ev.exit
  %tmp2 = bitcast double* %tmp.i7 to i8*
  call void @free(i8* %tmp2) #6
  br label %_ZN5Eigen8internal28aligned_stack_memory_handlerIdED2Ev.exit10

_ZN5Eigen8internal28aligned_stack_memory_handlerIdED2Ev.exit10: ; preds = %if.then.i8, %_ZN5Eigen8internal28aligned_stack_memory_handlerIdED2Ev.exit
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %tmp11) #6
  ret void

for.body:                                         ; preds = %for.cond.loopexit, %entry
  %i2.0183 = phi i64 [ 0, %entry ], [ %add, %for.cond.loopexit ]
  %add = add nsw i64 %i2.0183, %tmp8
  %tmp38 = load i64, i64* %rows.addr, align 8, !tbaa !26
  %sub = sub nsw i64 %tmp38, %i2.0183
  br label %for.body59

for.body59:                                       ; preds = %for.body59, %for.body
  %k2.0181 = phi i64 [ %add61, %for.body59 ], [ 0, %for.body ]
  %add61 = add nsw i64 %k2.0181, %tmp.i1
  store i64 %add61, i64* %ref.tmp60, align 8, !tbaa !26
  %call62 = call dereferenceable(8) i64* @_ZSt3minIlERKT_S2_S2_(i64* nonnull dereferenceable(8) %ref.tmp60, i64* nonnull dereferenceable(8) %depth.addr)
  %tmp41 = load i64, i64* %depth.addr, align 8, !tbaa !26
  %sub63 = sub nsw i64 %tmp41, %k2.0181
  %mul.i.i167 = mul nsw i64 %k2.0181, %lhsStride
  %add.i.i168 = add nsw i64 %mul.i.i167, %i2.0183
  %arrayidx.i.i169 = getelementptr inbounds double, double* %_lhs, i64 %add.i.i168
  store double* %arrayidx.i.i169, double** %tmp28, align 8
  store i64 %lhsStride, i64* %tmp29, align 8
  call void @_ZN5Eigen8internal13gemm_pack_lhsIdlNS0_22const_blas_data_mapperIdlLi0EEELi1ELi1ELi0ELb0ELb0EEclEPdRKS3_llll(%"struct.Eigen::internal::gemm_pack_lhs"* nonnull %pack_lhs, double* %tmp.i7, %"class.Eigen::internal::const_blas_data_mapper"* nonnull dereferenceable(16) %ref.tmp64, i64 %sub63, i64 %sub, i64 0, i64 0)
  store i64 %tmp10, i64* %ref.tmp70, align 8, !tbaa !26
  %call72 = call dereferenceable(8) i64* @_ZSt3minIlERKT_S2_S2_(i64* nonnull dereferenceable(8) %ref.tmp70, i64* nonnull dereferenceable(8) %cols.addr)
  %tmp40 = load i64, i64* %depth.addr, align 8, !tbaa !26
  %cmp57 = icmp slt i64 %add61, %tmp40
  br i1 %cmp57, label %for.body59, label %for.cond.loopexit
}

; Function Attrs: noinline nounwind uwtable
define linkonce_odr dso_local void @_ZN5Eigen8internal13gemm_pack_lhsIdlNS0_22const_blas_data_mapperIdlLi0EEELi1ELi1ELi0ELb0ELb0EEclEPdRKS3_llll(%"struct.Eigen::internal::gemm_pack_lhs"* %this, double* %blockA, %"class.Eigen::internal::const_blas_data_mapper"* dereferenceable(16) %lhs, i64 %depth, i64 %rows, i64 %stride, i64 %offset) local_unnamed_addr #2 comdat align 2 {
entry:
  ret void
}

attributes #0 = { argmemonly nounwind }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { inlinehint norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !4, i64 0}
!3 = !{!"_ZTSN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EEE", !4, i64 0, !7, i64 8, !7, i64 16}
!4 = !{!"any pointer", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}
!7 = !{!"long", !5, i64 0}
!8 = !{!9, !4, i64 0}
!9 = !{!"_ZTSN5Eigen8internal9evaluatorINS_15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEEEEE", !4, i64 0, !10, i64 8}
!10 = !{!"_ZTSN5Eigen8internal19variable_if_dynamicIlLin1EEE", !7, i64 0}
!11 = !{!3, !7, i64 8}
!12 = !{!10, !7, i64 0}
!13 = !{!14, !4, i64 0}
!14 = !{!"_ZTSN5Eigen7ProductINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEES2_Li0EEE", !4, i64 0, !4, i64 8}
!15 = !{!14, !4, i64 8}
!16 = !{!4, !4, i64 0}
!17 = !{!18, !18, i64 0}
!18 = !{!"double", !5, i64 0}
!19 = !{!3, !7, i64 16}
!20 = !{!21, !4, i64 0}
!21 = !{!"_ZTSN5Eigen7MapBaseINS_5BlockIKNS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1ELin1ELb0EEELi0EEE", !4, i64 0, !22, i64 8, !10, i64 16}
!22 = !{!"_ZTSN5Eigen8internal19variable_if_dynamicIlLi1EEE"}
!23 = !{!24}
!24 = distinct !{!24, !25, !"_ZNK5Eigen9DenseBaseINS_5BlockIKNS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1ELin1ELb0EEEE9transposeEv: %agg.result"}
!25 = distinct !{!25, !"_ZNK5Eigen9DenseBaseINS_5BlockIKNS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1ELin1ELb0EEEE9transposeEv"}
!26 = !{!7, !7, i64 0}
!27 = !{!28}
!28 = distinct !{!28, !29, !"_ZNK5Eigen9DenseBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEE3colEl: %agg.result"}
!29 = distinct !{!29, !"_ZNK5Eigen9DenseBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEE3colEl"}
!30 = !{!31, !4, i64 0}
!31 = !{!"_ZTSN5Eigen7MapBaseINS_5BlockIKNS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELin1ELi1ELb1EEELi0EEE", !4, i64 0, !10, i64 8, !22, i64 16}
!32 = !{!33, !4, i64 24}
!33 = !{!"_ZTSN5Eigen8internal15BlockImpl_denseIKNS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELin1ELi1ELb1ELb1EEE", !4, i64 24, !10, i64 32, !10, i64 40, !7, i64 48}
!34 = !{!33, !7, i64 48}
!35 = !{!36, !18, i64 24}
!36 = !{!"_ZTSN5Eigen8internal12gemm_functorIdlNS0_29general_matrix_matrix_productIldLi0ELb0EdLi0ELb0ELi0EEENS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEES5_S5_NS0_19gemm_blocking_spaceILi0EddLin1ELin1ELin1ELi1ELb0EEEEE", !4, i64 0, !4, i64 8, !4, i64 16, !18, i64 24, !4, i64 32}
!37 = !{!38, !4, i64 0}
!38 = !{!"_ZTSN5Eigen8internal15level3_blockingIddEE", !4, i64 0, !4, i64 8, !7, i64 16, !7, i64 24, !7, i64 32}
!39 = !{!38, !4, i64 8}
!40 = !{!41, !7, i64 48}
!41 = !{!"_ZTSN5Eigen8internal15BlockImpl_denseIKNS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1ELin1ELb0ELb1EEE", !4, i64 24, !10, i64 32, !10, i64 40, !7, i64 48}
!42 = !{!41, !4, i64 24}
!43 = !{!44}
!44 = distinct !{!44, !45, !"_ZN5Eigen9DenseBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEE8ConstantEllRKd: %agg.result"}
!45 = distinct !{!45, !"_ZN5Eigen9DenseBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEE8ConstantEllRKd"}
!46 = !{!36, !4, i64 0}
!47 = !{!36, !4, i64 8}
!48 = !{!36, !4, i64 16}
!49 = !{!36, !4, i64 32}
!50 = !{!38, !7, i64 32}
!51 = !{!38, !7, i64 16}

; CHECK: define internal { double } @diffe_ZN5Eigen8internal29general_matrix_matrix_productIldLi0ELb0EdLi0ELb0ELi0EE3runElllPKdlS4_lPdldRNS0_15level3_blockingIddEEPNS0_16GemmParallelInfoIlEE(i64 %rows, i64 %cols, i64 %depth, double* %_lhs, double* %"_lhs'", i64 %lhsStride, double* %_rhs, double* %"_rhs'", i64 %rhsStride, double* %_res, double* %"_res'", i64 %resStride, double %alpha, %"class.Eigen::internal::level3_blocking"* dereferenceable(40) %blocking, %"class.Eigen::internal::level3_blocking"* %"blocking'", %"struct.Eigen::internal::GemmParallelInfo"* %info
