; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -simplifycfg -adce -early-cse -S | FileCheck %s
source_filename = "/mnt/pci4/wmdata/Enzyme2/enzyme/test/Integration/ReverseMode/eigensumsqdyn.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }
%"struct.Eigen::internal::CacheSizes" = type { i64, i64, i64 }
%"class.Eigen::Matrix" = type { %"class.Eigen::PlainObjectBase" }
%"class.Eigen::PlainObjectBase" = type { %"class.Eigen::DenseStorage" }
%"class.Eigen::DenseStorage" = type { double*, i64, i64 }
%"class.Eigen::Product.19" = type { %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }
%"struct.Eigen::internal::assign_op" = type { i8 }
%"class.Eigen::internal::redux_evaluator" = type { %"struct.Eigen::internal::evaluator.18", %"class.Eigen::Product"* }
%"struct.Eigen::internal::evaluator.18" = type { %"struct.Eigen::internal::product_evaluator" }
%"struct.Eigen::internal::product_evaluator" = type { %"struct.Eigen::internal::evaluator.15", %"class.Eigen::Matrix" }
%"struct.Eigen::internal::evaluator.15" = type { %"struct.Eigen::internal::evaluator.16" }
%"struct.Eigen::internal::evaluator.16" = type { double*, %"class.Eigen::internal::variable_if_dynamic" }
%"class.Eigen::internal::variable_if_dynamic" = type { i64 }
%"class.Eigen::Product" = type { %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }
%"class.Eigen::internal::gemm_blocking_space" = type { %"class.Eigen::internal::level3_blocking", i64, i64 }
%"class.Eigen::internal::level3_blocking" = type { double*, double*, i64, i64, i64 }
%"struct.Eigen::internal::GemmParallelInfo" = type opaque
%"class.Eigen::DenseBase" = type { i8 }
%"struct.Eigen::internal::gemm_pack_lhs" = type { i8 }
%"struct.Eigen::internal::gemm_pack_rhs" = type { i8 }
%"struct.Eigen::internal::gebp_kernel" = type { i8 }
%"class.Eigen::internal::const_blas_data_mapper" = type { %"class.Eigen::internal::blas_data_mapper" }
%"class.Eigen::internal::blas_data_mapper" = type { double*, i64 }
%"class.Eigen::internal::blas_data_mapper.77" = type { double*, i64 }

define void @caller() {
_ZN5Eigen8internal17resize_if_allowedINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EEddEEvRT_RKT0_RKNS0_9assign_opIT1_T2_EE.exit.i.i.i.i.i.i:
  %W = alloca %"class.Eigen::Matrix", align 8
  %M = alloca %"class.Eigen::Matrix", align 8
  %Wp = alloca %"class.Eigen::Matrix", align 8
  %Mp = alloca %"class.Eigen::Matrix", align 8
  %call = call fast double @__enzyme_autodiff(i8* bitcast (double (%"class.Eigen::Matrix"*, %"class.Eigen::Matrix"*)* @_ZL6matvecPKN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEES3_ to i8*), %"class.Eigen::Matrix"* nonnull %W, %"class.Eigen::Matrix"* nonnull %Wp, %"class.Eigen::Matrix"* nonnull %M, %"class.Eigen::Matrix"* nonnull %Mp) #14
  ret void
}

declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i1)

declare dso_local double @__enzyme_autodiff(i8*, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"*) local_unnamed_addr #2

; Function Attrs: noinline nounwind uwtable
define internal double @_ZL6matvecPKN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEES3_(%"class.Eigen::Matrix"* noalias %W, %"class.Eigen::Matrix"* noalias %M) #3 {
entry:
  %ref.tmp.i.i.i.i = alloca %"class.Eigen::Product.19", align 8
  %thisEval.i.i = alloca %"class.Eigen::internal::redux_evaluator", align 8
  %diff = alloca %"class.Eigen::Matrix", align 8
  %ref.tmp1 = alloca %"class.Eigen::Product", align 8
  %i = bitcast %"class.Eigen::Matrix"* %diff to i8*
  call void @llvm.memset.p0i8.i64(i8* nonnull align 8 dereferenceable(24) %i, i8 0, i64 24, i1 false) #14
  %m_cols.i.i.i.i.i = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %diff, i64 0, i32 0, i32 0, i32 2
  %m_rows.i.i.i.i.i = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %diff, i64 0, i32 0, i32 0, i32 1
  %i1 = getelementptr %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %M, i64 0, i32 0, i32 0, i32 0
  %i2 = bitcast %"class.Eigen::Matrix"* %diff to i8**
  %call.i.i.i.i.i.i.i.i = tail call noalias dereferenceable_or_null(128) i8* @malloc(i64 128) #14
  store i8* %call.i.i.i.i.i.i.i.i, i8** %i2, align 8, !tbaa !2
  %i3 = bitcast i8* %call.i.i.i.i.i.i.i.i to double*
  store i64 4, i64* %m_rows.i.i.i.i.i, align 8, !tbaa !14
  store i64 4, i64* %m_cols.i.i.i.i.i, align 8, !tbaa !19
  %i4 = getelementptr %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %W, i64 0, i32 0, i32 0, i32 0
  %i5 = load double*, double** %i4, align 8, !tbaa !2
  %i6 = load double*, double** %i1, align 8, !tbaa !2
  %arrayidx.i.i.i.i.i.i.i.i.i = getelementptr inbounds double, double* %i3, i64 0; %i.07.i.i.i.i.i.i.i
  %arrayidx.i.i.i.i.i.i.i.i.i.i = getelementptr inbounds double, double* %i5, i64 0; %i.07.i.i.i.i.i.i.i
  %arrayidx.i6.i.i.i.i.i.i.i.i.i = getelementptr inbounds double, double* %i6, i64 0; %i.07.i.i.i.i.i.i.i
  %i7 = load double, double* %arrayidx.i.i.i.i.i.i.i.i.i.i, align 8, !tbaa !9
  %i8 = load double, double* %arrayidx.i6.i.i.i.i.i.i.i.i.i, align 8, !tbaa !9
  %sub.i.i.i.i.i.i.i.i.i.i = fsub fast double %i7, %i8
  store double %sub.i.i.i.i.i.i.i.i.i.i, double* %arrayidx.i.i.i.i.i.i.i.i.i, align 8, !tbaa !9
  %i9 = bitcast %"class.Eigen::Product"* %ref.tmp1 to i8*
  %i12 = bitcast %"class.Eigen::internal::redux_evaluator"* %thisEval.i.i to i8*
  %m_data.i.i.i.i = getelementptr inbounds %"class.Eigen::internal::redux_evaluator", %"class.Eigen::internal::redux_evaluator"* %thisEval.i.i, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %m_value.i.i.i.i.i = getelementptr inbounds %"class.Eigen::internal::redux_evaluator", %"class.Eigen::internal::redux_evaluator"* %thisEval.i.i, i64 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 0
  %m_result.i.i = getelementptr inbounds %"class.Eigen::internal::redux_evaluator", %"class.Eigen::internal::redux_evaluator"* %thisEval.i.i, i64 0, i32 0, i32 0, i32 1
  %i13 = bitcast %"class.Eigen::Matrix"* %m_result.i.i to i8**
  %call.i.i.i.i.i.i.i.i.i = call noalias dereferenceable_or_null(128) i8* @malloc(i64 128) #14
  store i8* %call.i.i.i.i.i.i.i.i.i, i8** %i13, align 8, !tbaa !2
  %i14 = ptrtoint i8* %call.i.i.i.i.i.i.i.i.i to i64
  %m_cols.i.i.i.i.i.i5 = getelementptr inbounds %"class.Eigen::internal::redux_evaluator", %"class.Eigen::internal::redux_evaluator"* %thisEval.i.i, i64 0, i32 0, i32 0, i32 1, i32 0, i32 0, i32 2
  %m_rows.i.i.i.i.i.i6 = getelementptr inbounds %"class.Eigen::internal::redux_evaluator", %"class.Eigen::internal::redux_evaluator"* %thisEval.i.i, i64 0, i32 0, i32 0, i32 1, i32 0, i32 0, i32 1
  store i64 4, i64* %m_rows.i.i.i.i.i.i6, align 8, !tbaa !14
  store i64 4, i64* %m_cols.i.i.i.i.i.i5, align 8, !tbaa !19
  %i15 = bitcast %"class.Eigen::internal::redux_evaluator"* %thisEval.i.i to i64*
  store i64 %i14, i64* %i15, align 8, !tbaa !21
  store i64 4, i64* %m_value.i.i.i.i.i, align 8, !tbaa !24
  %i16 = bitcast %"class.Eigen::Product.19"* %ref.tmp.i.i.i.i to i8*
  %i17 = getelementptr inbounds %"class.Eigen::Product.19", %"class.Eigen::Product.19"* %ref.tmp.i.i.i.i, i64 0, i32 0
  store %"class.Eigen::Matrix"* %diff, %"class.Eigen::Matrix"** %i17, align 8
  %i18 = getelementptr inbounds %"class.Eigen::Product.19", %"class.Eigen::Product.19"* %ref.tmp.i.i.i.i, i64 0, i32 1
  store %"class.Eigen::Matrix"* %diff, %"class.Eigen::Matrix"** %i18, align 8
  
  
  call void @_ZN5Eigen8internal26call_dense_assignment_loopINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_7ProductIS3_S3_Li1EEENS0_9assign_opIddEEEEvRT_RKT0_RKT1_(i64* %m_rows.i.i.i.i.i, double* %i5, %"class.Eigen::Matrix"* nonnull align 8 dereferenceable(24) %m_result.i.i, %"class.Eigen::Product.19"* nonnull align 8 dereferenceable(16) %ref.tmp.i.i.i.i) #14

  %m_xpr.i.i.i = getelementptr inbounds %"class.Eigen::internal::redux_evaluator", %"class.Eigen::internal::redux_evaluator"* %thisEval.i.i, i64 0, i32 1
  store %"class.Eigen::Product"* %ref.tmp1, %"class.Eigen::Product"** %m_xpr.i.i.i, align 8, !tbaa !15
  %i20 = load double*, double** %m_data.i.i.i.i, align 8, !tbaa !21
  %i21 = load double, double* %i20, align 8, !tbaa !9
  %arrayidx.i.i45.i.i.i = getelementptr inbounds double, double* %i20, i64 1
  %i24 = load double, double* %arrayidx.i.i45.i.i.i, align 8, !tbaa !9
  %add.i42.i.i.i = fadd fast double %i24, %i21
  %i22 = load i64, i64* %m_value.i.i.i.i.i, align 8
  %arrayidx.i.i.us.i.i.i = getelementptr inbounds double, double* %i20, i64 %i22
  %i23 = load double, double* %arrayidx.i.i.us.i.i.i, align 8, !tbaa !9
  %add.i.us.i.i.i = fadd fast double %i23, %add.i42.i.i.i
  ret double %add.i.us.i.i.i
}

; Function Attrs: nounwind readnone speculatable willreturn
declare double @llvm.fabs.f64(double) #4

; Function Attrs: nofree nounwind
declare dso_local i32 @fprintf(%struct._IO_FILE* nocapture, i8* nocapture readonly, ...) local_unnamed_addr #5

; Function Attrs: noreturn nounwind
declare dso_local void @abort() local_unnamed_addr #6

; Function Attrs: nounwind
declare dso_local void @free(i8* nocapture) local_unnamed_addr #7

; Function Attrs: nofree nounwind
declare dso_local noalias i8* @malloc(i64) local_unnamed_addr #5


define internal void @_ZN5Eigen8internal26call_dense_assignment_loopINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_7ProductIS3_S3_Li1EEENS0_9assign_opIddEEEEvRT_RKT0_RKT1_(i64* %i14, double* %i13, %"class.Eigen::Matrix"* nonnull align 8 dereferenceable(24) %dst, %"class.Eigen::Product.19"* nonnull align 8 dereferenceable(16) %src) local_unnamed_addr #7 {
entry:
  %i = bitcast %"class.Eigen::Product.19"* %src to %"class.Eigen::DenseBase"**
  %i1 = load %"class.Eigen::DenseBase"*, %"class.Eigen::DenseBase"** %i, align 8
  %m_rhs.i.i.i = getelementptr inbounds %"class.Eigen::Product.19", %"class.Eigen::Product.19"* %src, i64 0, i32 1
  %i2 = bitcast %"class.Eigen::Matrix"** %m_rhs.i.i.i to %"class.Eigen::DenseBase"**
  %i3 = load %"class.Eigen::DenseBase"*, %"class.Eigen::DenseBase"** %i2, align 8
  %m_rows.i.i.i.i = getelementptr inbounds %"class.Eigen::DenseBase", %"class.Eigen::DenseBase"* %i1, i64 8
  %i4 = bitcast %"class.Eigen::DenseBase"* %m_rows.i.i.i.i to i64*
  %m_cols.i.i.i.i7 = getelementptr inbounds %"class.Eigen::DenseBase", %"class.Eigen::DenseBase"* %i3, i64 16
  %i6 = bitcast %"class.Eigen::DenseBase"* %m_cols.i.i.i.i7 to i64*
  %i7 = load i64, i64* %i6, align 8
  %m_rows.i.i12.i = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %dst, i64 0, i32 0, i32 0, i32 1
  %m_cols.i.i13.i = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %dst, i64 0, i32 0, i32 0, i32 2
  br label %for.cond1.preheader.i

for.cond1.preheader.i:                            ; preds = %for.cond.cleanup4.i, %entry
  %tiv = phi i64 [ %tiv.next, %for.cond.cleanup4.i ], [ 0, %entry ]
  %tiv.next = add nuw nsw i64 %tiv, 1
  %i18 = load i64, i64* %i14, align 8
  br label %for.body.i.i.i.i.i.i.us29.i

for.body.i.i.i.i.i.i.us29.i:                      ; preds = %for.body.i.i.i.i.i.i.us29.i, %for.body5.us24.i
  %i.059 = phi i64 [ %inc.i.i.i.i.i.i.us37.i, %for.body.i.i.i.i.i.i.us29.i ], [ 1, %for.cond1.preheader.i ]
  %i24 = load double, double* %i13, align 8
  %mul.i.i.i50.i.i.i.i.i.i.us35.i = fmul fast double %i24, %i24
  store double %mul.i.i.i50.i.i.i.i.i.i.us35.i, double* %i13, align 8
  %inc.i.i.i.i.i.i.us37.i = add nuw nsw i64 %i.059, 1
  %exitcond.not.i.i.i.i.i.i.us38.i = icmp eq i64 %inc.i.i.i.i.i.i.us37.i, %i18
  br i1 %exitcond.not.i.i.i.i.i.i.us38.i, label %for.cond.cleanup4.i, label %for.body.i.i.i.i.i.i.us29.i, !llvm.loop !26

for.cond.cleanup4.i:                              ; preds = %_ZN5Eigen8internal31generic_dense_assignment_kernelINS0_9evaluatorINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEEENS2_INS_7ProductIS4_S4_Li1EEEEENS0_9assign_opIddEELi0EE23assignCoeffByOuterInnerEll.exit.loopexit.us46.i, %for.body5.us.preheader.i
  %inc7.i = add nuw nsw i64 %tiv, 1
  %exitcond56.not.i = icmp eq i64 %inc7.i, %i7
  br i1 %exitcond56.not.i, label %_ZN5Eigen8internal21dense_assignment_loopINS0_31generic_dense_assignment_kernelINS0_9evaluatorINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEEENS3_INS_7ProductIS5_S5_Li1EEEEENS0_9assign_opIddEELi0EEELi0ELi0EE3runERSC_.exit, label %for.cond1.preheader.i, !llvm.loop !28

_ZN5Eigen8internal21dense_assignment_loopINS0_31generic_dense_assignment_kernelINS0_9evaluatorINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEEENS3_INS_7ProductIS5_S5_Li1EEEEENS0_9assign_opIddEELi0EEELi0ELi0EE3runERSC_.exit: ; preds = %for.cond.cleanup4.i
  ret void
}

; Function Attrs: nofree nounwind
declare dso_local i32 @__cxa_guard_acquire(i64*) local_unnamed_addr #10

; Function Attrs: nofree nounwind
declare dso_local void @__cxa_guard_release(i64*) local_unnamed_addr #10

attributes #0 = { norecurse nounwind uwtable }
attributes #1 = { argmemonly nounwind }
attributes #3 = { noinline nounwind uwtable }
attributes #4 = { nounwind readnone }
attributes #10 = { nounwind }
attributes #11 = { inaccessiblemem_or_argmemonly nounwind }
attributes #12 = { argmemonly nounwind }
attributes #13 = { nounwind readnone }
attributes #14 = { nounwind }
attributes #15 = { cold }
attributes #16 = { noreturn nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 11.0.0 (git@github.com:llvm/llvm-project b78e5de029c26c309f541ab883fa5d6d953b073d)"}
!2 = !{!3, !4, i64 0}
!3 = !{!"_ZTSN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EEE", !4, i64 0, !7, i64 8, !7, i64 16}
!4 = !{!"any pointer", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}
!7 = !{!"long", !5, i64 0}
!8 = !{!7, !7, i64 0}
!9 = !{!10, !10, i64 0}
!10 = !{!"double", !5, i64 0}
!11 = distinct !{!11, !12}
!12 = !{!"llvm.loop.unroll.disable"}
!13 = distinct !{!13, !12}
!14 = !{!3, !7, i64 8}
!15 = !{!4, !4, i64 0}
!16 = distinct !{!16, !12}
!17 = distinct !{!17, !12}
!18 = distinct !{!18, !12}
!19 = !{!3, !7, i64 16}
!20 = distinct !{!20, !12}
!21 = !{!22, !4, i64 0}
!22 = !{!"_ZTSN5Eigen8internal9evaluatorINS_15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEEEEE", !4, i64 0, !23, i64 8}
!23 = !{!"_ZTSN5Eigen8internal19variable_if_dynamicIlLin1EEE", !7, i64 0}
!24 = !{!23, !7, i64 0}
!25 = !{!26, !7, i64 32}
!26 = !{!"_ZTSN5Eigen8internal15level3_blockingIddEE", !4, i64 0, !4, i64 8, !7, i64 16, !7, i64 24, !7, i64 32}
!27 = !{!26, !7, i64 16}
!28 = !{!29, !7, i64 40}
!29 = !{!"_ZTSN5Eigen8internal19gemm_blocking_spaceILi0EddLin1ELin1ELin1ELi1ELb0EEE", !7, i64 40, !7, i64 48}
!30 = !{!26, !7, i64 24}
!31 = !{!29, !7, i64 48}
!32 = !{!26, !4, i64 0}
!33 = !{!26, !4, i64 8}
!34 = !{!35, !4, i64 0}
!35 = !{!"_ZTSN5Eigen7ProductINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEES2_Li1EEE", !4, i64 0, !4, i64 8}
!36 = !{!35, !4, i64 8}
!37 = !{!38}
!38 = distinct !{!38, !39, !"_ZNK5Eigen9DenseBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEE3rowEl: %agg.result"}
!39 = distinct !{!39, !"_ZNK5Eigen9DenseBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEE3rowEl"}
!40 = !{!41}
!41 = distinct !{!41, !42, !"_ZNK5Eigen9DenseBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEE3colEl: %agg.result"}
!42 = distinct !{!42, !"_ZNK5Eigen9DenseBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEE3colEl"}
!43 = distinct !{!43, !12}
!44 = distinct !{!44, !12}
!45 = distinct !{!45, !12}
!46 = !{!"branch_weights", i32 1, i32 1048575}
!47 = !{!48, !48, i64 0}
!48 = !{!"int", !5, i64 0}
!49 = !{!50, !7, i64 0}
!50 = !{!"_ZTSN5Eigen8internal10CacheSizesE", !7, i64 0, !7, i64 8, !7, i64 16}
!51 = !{!50, !7, i64 8}
!52 = !{!50, !7, i64 16}
!53 = !{i32 -2143794158}
!54 = !{i32 -2143794770}
!55 = distinct !{!55, !12}
!56 = !{i32 -2143794464}
!57 = !{i32 -2143794311}
!58 = !{i32 -2143794617}
!59 = distinct !{!59, !12}
!60 = !{!5, !5, i64 0}
!61 = distinct !{!61, !12}
!62 = distinct !{!62, !12}
!63 = !{i32 -2142431813}
!64 = distinct !{!64, !12}
!65 = distinct !{!65, !12}
!66 = !{i32 -2142431108}
!67 = !{!68, !4, i64 0}
!68 = !{!"_ZTSN5Eigen8internal16blas_data_mapperIKdlLi0ELi0EEE", !4, i64 0, !7, i64 8}
!69 = !{!68, !7, i64 8}
!70 = distinct !{!70, !12}
!71 = distinct !{!71, !12}
!72 = distinct !{!72, !12}
!73 = distinct !{!73, !12}
!74 = !{!75, !4, i64 0}
!75 = !{!"_ZTSN5Eigen8internal16blas_data_mapperIdlLi0ELi0EEE", !4, i64 0, !7, i64 8}
!76 = !{!75, !7, i64 8}
!77 = distinct !{!77, !12}
!78 = !{i32 -2142432234}
!79 = !{i32 -2142432180}
!80 = !{i32 -2142432117}
!81 = distinct !{!81, !12}
!82 = !{i32 -2142439040}
!83 = !{i32 -2142438398}
!84 = !{i32 -2142438344}
!85 = !{i32 -2142438281}
!86 = !{i32 -2142437633}
!87 = !{i32 -2142437579}
!88 = !{i32 -2142437516}
!89 = !{i32 -2142436868}
!90 = !{i32 -2142436814}
!91 = !{i32 -2142436751}
!92 = !{i32 -2142436103}
!93 = !{i32 -2142436049}
!94 = !{i32 -2142435986}
!95 = !{i32 -2142435338}
!96 = !{i32 -2142435284}
!97 = !{i32 -2142435221}
!98 = !{i32 -2142434573}
!99 = !{i32 -2142434519}
!100 = !{i32 -2142434456}
!101 = !{i32 -2142433808}
!102 = !{i32 -2142433754}
!103 = !{i32 -2142433691}
!104 = !{i32 -2142433043}
!105 = !{i32 -2142432989}
!106 = !{i32 -2142432926}
!107 = !{i32 -2142432874}
!108 = distinct !{!108, !12}
!109 = !{i32 -2142448220}
!110 = !{i32 -2142447333}
!111 = !{i32 -2142447279}
!112 = !{i32 -2142447216}
!113 = !{i32 -2142446323}
!114 = !{i32 -2142446269}
!115 = !{i32 -2142446206}
!116 = !{i32 -2142445313}
!117 = !{i32 -2142445259}
!118 = !{i32 -2142445196}
!119 = !{i32 -2142444303}
!120 = !{i32 -2142444249}
!121 = !{i32 -2142444186}
!122 = !{i32 -2142443293}
!123 = !{i32 -2142443239}
!124 = !{i32 -2142443176}
!125 = !{i32 -2142442283}
!126 = !{i32 -2142442229}
!127 = !{i32 -2142442166}
!128 = !{i32 -2142441273}
!129 = !{i32 -2142441219}
!130 = !{i32 -2142441156}
!131 = !{i32 -2142440263}
!132 = !{i32 -2142440209}
!133 = !{i32 -2142440146}
!134 = !{i32 -2142440094}
!135 = distinct !{!135, !12}
!136 = distinct !{!136, !12}
!137 = !{i32 -2142439209}
!138 = !{i32 -2142439155}
!139 = !{i32 -2142439092}
!140 = distinct !{!140, !12}
!141 = distinct !{!141, !12}

; CHECK: define internal void @diffe_ZN5Eigen8internal26call_dense_assignment_loopINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_7ProductIS3_S3_Li1EEENS0_9assign_opIddEEEEvRT_RKT0_RKT1_
