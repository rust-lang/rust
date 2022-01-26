; RUN: if [ %llvmver -ge 9 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -adce -loop-deletion -correlated-propagation -simplifycfg -adce -simplifycfg -S | FileCheck %s; fi
source_filename = "lulesh.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%0 = type { i8 }
%1 = type { i32, i32, i32, i32, i8* }

@0 = internal global %0 zeroinitializer, align 1
@__dso_handle = external hidden global i8
@1 = private unnamed_addr constant [43 x i8] c";lulesh.cc;CalcSoundSpeedForElems;2175;1;;\00", align 1
@2 = private unnamed_addr constant %1 { i32 0, i32 514, i32 0, i32 0, i8* getelementptr inbounds ([43 x i8], [43 x i8]* @1, i32 0, i32 0) }, align 8
@3 = private unnamed_addr constant [44 x i8] c";lulesh.cc;CalcSoundSpeedForElems;2175;51;;\00", align 1
@4 = private unnamed_addr constant %1 { i32 0, i32 514, i32 0, i32 0, i8* getelementptr inbounds ([44 x i8], [44 x i8]* @3, i32 0, i32 0) }, align 8
@5 = private unnamed_addr constant %1 { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([43 x i8], [43 x i8]* @1, i32 0, i32 0) }, align 8

; Function Attrs: nounwind
declare dso_local void @__kmpc_for_static_init_8(%1*, i32, i32, i64*, i64*, i64*, i64*, i32, i32)

; Function Attrs: nounwind
declare void @__kmpc_for_static_fini(%1*, i32)

; Function Attrs: nounwind
declare !callback !5 void @__kmpc_fork_call(%1*, i32, void (i32*, i32*, ...)*, ...)

; Function Attrs: mustprogress nofree nounwind willreturn
declare dso_local double @sqrt(double)

define void @caller(i8* %arg, i8* %arg1) {
bb:
  call void @_Z17__enzyme_autodiffPvS_S_(i8* bitcast (void (double*, double*)* @func to i8*), i8* nonnull %arg, i8* nonnull %arg1, i8* nonnull %arg, i8* nonnull %arg1)
  ret void
}

declare dso_local void @_Z17__enzyme_autodiffPvS_S_(i8*, i8*, i8*, i8*, i8*)

define internal void @func(double* %arg, double* %arg1) {
entry:
  call void (%1*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%1* nonnull @5, i32 10, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, double*, double*)* @outlined to void (i32*, i32*, ...)*), double* nonnull %arg, double* nonnull %arg1)
  ret void
}

; Function Attrs: alwaysinline norecurse nounwind uwtable
define internal void @outlined(i32* noalias nocapture readonly %arg, i32* noalias nocapture readnone %arg1, double* nocapture nonnull readonly %arg2, double* nocapture nonnull %arg3) {
entry:
  %omp.lb = alloca i64, align 4
  %omp.ub = alloca i64, align 4
  %omp.step = alloca i64, align 4
  %omp.last = alloca i64, align 4
  store i64 0, i64* %omp.lb, align 4, !tbaa !7
  store i64 9, i64* %omp.ub, align 4, !tbaa !7
  store i64 1, i64* %omp.step, align 4, !tbaa !7
  store i64 0, i64* %omp.last, align 4, !tbaa !7
  %i7 = load i32, i32* %arg, align 4, !tbaa !7
  call void @__kmpc_for_static_init_8(%1* nonnull @2, i32 %i7, i32 34, i64* nonnull %omp.last, i64* nonnull %omp.lb, i64* nonnull %omp.ub, i64* nonnull %omp.step, i32 1, i32 1)
  %i8 = load i64, i64* %omp.ub, align 4, !tbaa !7
  %i9 = icmp slt i64 %i8, 10
  %i10 = select i1 %i9, i64 %i8, i64 9
  store i64 %i10, i64* %omp.ub, align 4, !tbaa !7
  %i11 = load i64, i64* %omp.lb, align 4, !tbaa !7
  %i12 = icmp sgt i64 %i11, %i10
  br i1 %i12, label %exit, label %bb13

bb13:                                             ; preds = %bb
  br label %loop.body

loop.body:                                             ; preds = %bb23, %bb13
  %i15 = phi i64 [ %i10, %bb13 ], [ %i24, %bb23 ]
  %i16 = phi i64 [ %i11, %bb13 ], [ %i27, %bb23 ]
  %i17 = getelementptr inbounds double, double* %arg2, i64 %i16
  %i18 = load double, double* %i17, align 8, !tbaa !11
  %i19 = fcmp ugt double %i18, 0x3842E7922A37D1A0
  br i1 %i19, label %bb20, label %bb23

bb20:                                             ; preds = %bb14
  %i21 = call double @sqrt(double %i18) #4
  %i22 = load i64, i64* %omp.ub, align 4, !tbaa !7
  br label %bb23

bb23:                                             ; preds = %bb20, %bb14
  %i24 = phi i64 [ %i22, %bb20 ], [ %i15, %loop.body ]
  %i25 = phi double [ %i21, %bb20 ], [ 0x3C18987CEE7F439D, %loop.body ]
  %i26 = getelementptr inbounds double, double* %arg3, i64 %i16
  store double %i25, double* %i26, align 8, !tbaa !11
  %i27 = add nsw i64 %i16, 1
  %i28 = icmp slt i64 %i16, %i24
  br i1 %i28, label %loop.body, label %exit

exit:                                             ; preds = %bb23, %bb
  call void @__kmpc_for_static_fini(%1* nonnull @4, i32 %i7)
  ret void
}

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}
!nvvm.annotations = !{}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"openmp", i32 50}
!3 = !{i32 7, !"uwtable", i32 1}
!4 = !{!"clang version 13.0.1 (git@github.com:llvm/llvm-project cf15ccdeb6d5254ee7d46c7535c29200003a3880)"}
!5 = !{!6}
!6 = !{i64 2, i64 -1, i64 -1, i1 true}
!7 = !{!8, !8, i64 0}
!8 = !{!"int", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C++ TBAA"}
!11 = !{!12, !12, i64 0}
!12 = !{!"double", !9, i64 0}

; CHECK: define internal void @diffefunc(double* %arg, double* %"arg'", double* %arg1, double* %"arg1'") {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[i0:.+]] = alloca double*
; CHECK-NEXT:   %[[i1:.+]] = alloca double*
; CHECK-NEXT:   %[[i2:.+]] = tail call noalias nonnull dereferenceable(80) dereferenceable_or_null(80) i8* @malloc(i64 80)
; CHECK-NEXT:   %i18_malloccache_unwrap = bitcast i8* %2 to double*
; CHECK-NEXT:   store double* %i18_malloccache_unwrap, double** %[[i0]]
; CHECK-NEXT:   call void (%1*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%1* @5, i32 5, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, double*, double*, double*, double*, double**)* @augmented_outlined.1 to void (i32*, i32*, ...)*), double* %arg, double* %"arg'", double* %arg1, double* %"arg1'", double** nonnull %[[i0]])
; CHECK-NEXT:   store double* %i18_malloccache_unwrap, double** %[[i1]]
; CHECK-NEXT:   call void (%1*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%1* @5, i32 5, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, double*, double*, double*, double*, double**)* @diffeoutlined to void (i32*, i32*, ...)*), double* %arg, double* %"arg'", double* %arg1, double* %"arg1'", double** nonnull %[[i1]])
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[i2]])
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
