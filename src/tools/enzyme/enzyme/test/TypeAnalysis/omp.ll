; RUN: %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=caller -o /dev/null | FileCheck %s

source_filename = "lulesh.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.ident_t = type { i32, i32, i32, i32, i8* }

@0 = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
@1 = private unnamed_addr constant %struct.ident_t { i32 0, i32 514, i32 0, i32 0, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @0, i32 0, i32 0) }, align 8
@2 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @0, i32 0, i32 0) }, align 8

declare i8* @malloc(i64)

; Function Attrs: inlinehint nounwind uwtable mustprogress
define internal void @caller(i64 %length) #3 {
entry:
  %alloc = call i8* @malloc(i64 10000)
  %e_new = bitcast i8* %alloc to double*
  tail call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%struct.ident_t* nonnull @2, i32 2, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, i64, double*)* @.omp_outlined. to void (i32*, i32*, ...)*), i64 %length, double* %e_new)
  ret void
}

; Function Attrs: norecurse nounwind uwtable
define internal void @.omp_outlined.(i32* noalias nocapture readonly %.global_tid., i32* noalias nocapture readnone %.bound_tid., i64 %length, double* nocapture nonnull align 8 dereferenceable(8) %tmp) #4 {
entry:
  %.omp.lb = alloca i64, align 8
  %.omp.ub = alloca i64, align 8
  %.omp.stride = alloca i64, align 8
  %.omp.is_last = alloca i32, align 4
  %sub4 = add i64 %length, -1
  %cmp.not = icmp eq i64 %length, 0
  br i1 %cmp.not, label %omp.precond.end, label %omp.precond.then

omp.precond.then:                                 ; preds = %entry
  %0 = bitcast i64* %.omp.lb to i8*
  store i64 0, i64* %.omp.lb, align 8, !tbaa !3
  %1 = bitcast i64* %.omp.ub to i8*
  store i64 %sub4, i64* %.omp.ub, align 8, !tbaa !3
  %2 = bitcast i64* %.omp.stride to i8*
  store i64 1, i64* %.omp.stride, align 8, !tbaa !3
  %3 = bitcast i32* %.omp.is_last to i8*
  store i32 0, i32* %.omp.is_last, align 4, !tbaa !7
  %4 = load i32, i32* %.global_tid., align 4, !tbaa !7
  call void @__kmpc_for_static_init_8u(%struct.ident_t* nonnull @1, i32 %4, i32 34, i32* nonnull %.omp.is_last, i64* nonnull %.omp.lb, i64* nonnull %.omp.ub, i64* nonnull %.omp.stride, i64 1, i64 1)
  %5 = load i64, i64* %.omp.ub, align 8, !tbaa !3
  %cmp6 = icmp ugt i64 %5, %sub4
  %cond = select i1 %cmp6, i64 %sub4, i64 %5
  store i64 %cond, i64* %.omp.ub, align 8, !tbaa !3
  %6 = load i64, i64* %.omp.lb, align 8, !tbaa !3
  %add29 = add i64 %cond, 1
  %cmp730 = icmp ult i64 %6, %add29
  br i1 %cmp730, label %omp.inner.for.body, label %omp.loop.exit

omp.inner.for.body:                               ; preds = %omp.precond.then, %omp.inner.for.body
  %.omp.iv.031 = phi i64 [ %add11, %omp.inner.for.body ], [ %6, %omp.precond.then ]
  %arrayidx = getelementptr inbounds double, double* %tmp, i64 %.omp.iv.031
  %7 = load double, double* %arrayidx, align 8, !tbaa !9
  %call = call double @sqrt(double %7) #5
  store double %call, double* %arrayidx, align 8, !tbaa !9
  %add11 = add nuw i64 %.omp.iv.031, 1
  %8 = load i64, i64* %.omp.ub, align 8, !tbaa !3
  %add = add i64 %8, 1
  %cmp7 = icmp ult i64 %add11, %add
  br i1 %cmp7, label %omp.inner.for.body, label %omp.loop.exit

omp.loop.exit:                                    ; preds = %omp.inner.for.body, %omp.precond.then
  call void @__kmpc_for_static_fini(%struct.ident_t* nonnull @1, i32 %4)
  br label %omp.precond.end

omp.precond.end:                                  ; preds = %omp.loop.exit, %entry
  ret void
}

; Function Attrs: nounwind
declare dso_local void @__kmpc_for_static_init_8u(%struct.ident_t*, i32, i32, i32*, i64*, i64*, i64*, i64, i64) local_unnamed_addr #5

; Function Attrs: nofree nounwind willreturn mustprogress
declare dso_local double @sqrt(double) local_unnamed_addr #6

; Function Attrs: nounwind
declare void @__kmpc_for_static_fini(%struct.ident_t*, i32) local_unnamed_addr #5

; Function Attrs: nounwind
declare !callback !11 void @__kmpc_fork_call(%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) local_unnamed_addr #5

attributes #0 = { norecurse nounwind uwtable }
attributes #1 = { argmemonly }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}
!nvvm.annotations = !{}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"uwtable", i32 1}
!2 = !{!"clang version 13.0.0 (git@github.com:llvm/llvm-project 619bfe8bd23f76b22f0a53fedafbfc8c97a15f12)"}
!3 = !{!4, !4, i64 0}
!4 = !{!"long", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}
!7 = !{!8, !8, i64 0}
!8 = !{!"int", !5, i64 0}
!9 = !{!10, !10, i64 0}
!10 = !{!"double", !5, i64 0}
!11 = !{!12}
!12 = !{i64 2, i64 -1, i64 -1, i1 true}

; CHECK: caller - {} |{[-1]:Integer}:{}
; CHECK-NEXT: i64 %length: {[-1]:Integer}
; CHECK-NEXT: entry
; CHECK-NEXT:   %alloc = call i8* @malloc(i64 10000): {[-1]:Pointer, [-1,0]:Float@double}
; CHECK-NEXT:   %e_new = bitcast i8* %alloc to double*: {[-1]:Pointer, [-1,0]:Float@double}
; CHECK-NEXT:   tail call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%struct.ident_t* nonnull @2, i32 2, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, i64, double*)* @.omp_outlined. to void (i32*, i32*, ...)*), i64 %length, double* %e_new): {}
; CHECK-NEXT:   ret void: {}
