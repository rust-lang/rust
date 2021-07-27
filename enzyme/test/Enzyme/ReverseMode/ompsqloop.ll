; RUN: if [ %llvmver -ge 9 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -adce -loop-deletion -correlated-propagation -simplifycfg -S | FileCheck %s; fi

source_filename = "lulesh.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.ident_t = type { i32, i32, i32, i32, i8* }

@0 = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
@1 = private unnamed_addr constant %struct.ident_t { i32 0, i32 514, i32 0, i32 0, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @0, i32 0, i32 0) }, align 8
@2 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @0, i32 0, i32 0) }, align 8

; Function Attrs: norecurse nounwind uwtable mustprogress
define dso_local i32 @main(i32 %argc, i8** nocapture readnone %argv) local_unnamed_addr #0 {
entry:
  %data = alloca [100 x double], align 16
  %d_data = alloca [100 x double], align 16
  %0 = bitcast [100 x double]* %data to i8*
  %1 = bitcast [100 x double]* %d_data to i8*
  call void @_Z17__enzyme_autodiffPvS_S_m(i8* bitcast (void (double*, i64)* @_ZL16LagrangeLeapFrogPdm to i8*), i8* nonnull %0, i8* nonnull %1, i64 100) #5
  ret i32 0
}

declare dso_local void @_Z17__enzyme_autodiffPvS_S_m(i8*, i8*, i8*, i64) local_unnamed_addr #2

; Function Attrs: inlinehint nounwind uwtable mustprogress
define internal void @_ZL16LagrangeLeapFrogPdm(double* %e_new, i64 %length) #3 {
entry:
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


; CHECK: define internal void @augmented_.omp_outlined..1(i32* noalias nocapture readonly %.global_tid., i32* noalias nocapture readnone %.bound_tid., i64 %length, double* nocapture nonnull align 8 dereferenceable(8) %tmp, double* nocapture %"tmp'", double** %tape)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = load double*, double** %tape
; CHECK-NEXT:   %.omp.lb_smpl = alloca i64
; CHECK-NEXT:   %.omp.ub_smpl = alloca i64
; CHECK-NEXT:   %.omp.stride_smpl = alloca i64
; CHECK-NEXT:   %.omp.is_last = alloca i32, align 4
; CHECK-NEXT:   %sub4 = add i64 %length, -1
; CHECK-NEXT:   %cmp.not = icmp eq i64 %length, 0
; CHECK-NEXT:   br i1 %cmp.not, label %omp.precond.end, label %omp.precond.then

; CHECK: omp.precond.then:                                 ; preds = %entry
; CHECK-NEXT:   store i32 0, i32* %.omp.is_last, align 4, !tbaa !7
; CHECK-NEXT:   %1 = load i32, i32* %.global_tid., align 4, !tbaa !7
; CHECK-NEXT:   store i64 0, i64* %.omp.lb_smpl
; CHECK-NEXT:   store i64 %sub4, i64* %.omp.ub_smpl
; CHECK-NEXT:   store i64 1, i64* %.omp.stride_smpl
; CHECK-NEXT:   call void @__kmpc_for_static_init_8u(%struct.ident_t* nonnull @1, i32 %1, i32 34, i32* nonnull %.omp.is_last, i64* nocapture nonnull %.omp.lb_smpl, i64* nocapture nonnull %.omp.ub_smpl, i64* nocapture nonnull %.omp.stride_smpl, i64 1, i64 1)
; CHECK-NEXT:   %2 = load i64, i64* %.omp.lb_smpl
; CHECK-NEXT:   %3 = load i64, i64* %.omp.ub_smpl
; CHECK-NEXT:   %4 = load i64, i64* %.omp.lb_smpl
; CHECK-NEXT:   %cmp6 = icmp ugt i64 %3, %sub4
; CHECK-NEXT:   %cond = select i1 %cmp6, i64 %sub4, i64 %3
; CHECK-NEXT:   %add29 = add i64 %cond, 1
; CHECK-NEXT:   %cmp730 = icmp ult i64 %4, %add29
; CHECK-NEXT:   br i1 %cmp730, label %omp.inner.for.body, label %omp.loop.exit

; CHECK: omp.inner.for.body:                               ; preds = %omp.precond.then, %omp.inner.for.body
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %omp.inner.for.body ], [ 0, %omp.precond.then ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %5 = add i64 
;                            %4, %iv
; CHECK-NEXT:   %arrayidx = getelementptr inbounds double, double* %tmp, i64 %5
; CHECK-NEXT:   %6 = load double, double* %arrayidx, align 8, !tbaa !9
; CHECK-NEXT:   %call = call double @sqrt(double %6)
; CHECK-NEXT:   store double %call, double* %arrayidx, align 8, !tbaa !9
; CHECK-NEXT:   %7 = add nuw nsw i64 %iv, %2
; CHECK-NEXT:   %8 = getelementptr inbounds double, double* %0, i64 %7
; CHECK-NEXT:   store double %6, double* %8, align 8, !invariant.group !13
; CHECK-NEXT:   %add11 = add nuw i64 %5, 1
; CHECK-NEXT:   %add = add nuw i64 %cond, 1
; CHECK-NEXT:   %cmp7 = icmp ult i64 %add11, %add
; CHECK-NEXT:   br i1 %cmp7, label %omp.inner.for.body, label %omp.loop.exit

; CHECK: omp.loop.exit:                                    ; preds = %omp.inner.for.body, %omp.precond.then
; CHECK-NEXT:   call void @__kmpc_for_static_fini(%struct.ident_t* nonnull @1, i32 %1)
; CHECK-NEXT:   br label %omp.precond.end

; CHECK: omp.precond.end:                                  ; preds = %omp.loop.exit, %entry
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @diffe.omp_outlined.(i32* noalias nocapture readonly %.global_tid., i32* noalias nocapture readnone %.bound_tid., i64 %length, double* nocapture nonnull align 8 dereferenceable(8) %tmp, double* nocapture %"tmp'", double** %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %truetape = load double*, double** %tapeArg
; CHECK-NEXT:   %.omp.lb_smpl = alloca i64
; CHECK-NEXT:   %.omp.ub_smpl = alloca i64
; CHECK-NEXT:   %.omp.stride_smpl = alloca i64
; CHECK-NEXT:   %.omp.is_last = alloca i32, align 4
; CHECK-NEXT:   %sub4 = add i64 %length, -1
; CHECK-NEXT:   %cmp.not = icmp eq i64 %length, 0
; CHECK-NEXT:   br i1 %cmp.not, label %invertentry, label %omp.precond.then

; CHECK: omp.precond.then:                                 ; preds = %entry
; CHECK-NEXT:   store i32 0, i32* %.omp.is_last, align 4, !tbaa !7
; CHECK-NEXT:   %0 = load i32, i32* %.global_tid., align 4, !tbaa !7, !invariant.group !15
; CHECK-NEXT:   store i64 0, i64* %.omp.lb_smpl
; CHECK-NEXT:   store i64 %sub4, i64* %.omp.ub_smpl
; CHECK-NEXT:   store i64 1, i64* %.omp.stride_smpl
; CHECK-NEXT:   call void @__kmpc_for_static_init_8u(%struct.ident_t* nonnull @1, i32 %0, i32 34, i32* nonnull %.omp.is_last, i64* nocapture nonnull %.omp.lb_smpl, i64* nocapture nonnull %.omp.ub_smpl, i64* nocapture nonnull %.omp.stride_smpl, i64 1, i64 1)
; CHECK-NEXT:   %1 = load i64, i64* %.omp.lb_smpl
; CHECK-NEXT:   %2 = load i64, i64* %.omp.ub_smpl
; CHECK-NEXT:   %3 = load i64, i64* %.omp.lb_smpl
; CHECK-NEXT:   %cmp6 = icmp ugt i64 %2, %sub4
; CHECK-NEXT:   %cond = select i1 %cmp6, i64 %sub4, i64 %2
; CHECK-NEXT:   %add29 = add i64 %cond, 1
; CHECK-NEXT:   %cmp730 = icmp ult i64 %3, %add29
; CHECK-NEXT:   br i1 %cmp730, label %omp.inner.for.body, label %invertomp.precond.end

; CHECK: omp.inner.for.body:                               ; preds = %omp.precond.then, %omp.inner.for.body
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %omp.inner.for.body ], [ 0, %omp.precond.then ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %4 = add i64
;                            %3, %iv
; CHECK-NEXT:   %5 = add nuw nsw i64 %iv, %1
; CHECK-NEXT:   %6 = getelementptr inbounds double, double* %truetape, i64 %5
; CHECK-NEXT:   %7 = load double, double* %6, align 8, !invariant.group !19
; CHECK-NEXT:   %call = call double @sqrt(double %7)
; CHECK-NEXT:   %add11 = add nuw i64 %4, 1
; CHECK-NEXT:   %add = add nuw i64 %cond, 1
; CHECK-NEXT:   %cmp7 = icmp ult i64 %add11, %add
; CHECK-NEXT:   br i1 %cmp7, label %omp.inner.for.body, label %invertomp.precond.end

; CHECK: invertentry:                                      ; preds = %entry, %invertomp.precond.end, %invertomp.precond.then
; CHECK-NEXT:   ret void

; CHECK: invertomp.precond.then:                           ; preds = %invertomp.inner.for.body, %invertomp.loop.exit
; CHECK-NEXT:   %_unwrap = load i32, i32* %.global_tid., align 4, !tbaa !7, !invariant.group !15
; CHECK-NEXT:   call void @__kmpc_for_static_fini(%struct.ident_t* @1, i32 %_unwrap)
; CHECK-NEXT:   br label %invertentry

; CHECK: invertomp.inner.for.body:                         ; preds = %invertomp.loop.exit.loopexit, %incinvertomp.inner.for.body
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %_unwrap9, %invertomp.loop.exit.loopexit ], [ %19, %incinvertomp.inner.for.body ]
; CHECK-NEXT:   %_unwrap2 = load i64, i64* %.omp.lb_smpl
; CHECK-NEXT:   %_unwrap3 = add i64 %_unwrap2, %"iv'ac.0"
; CHECK-NEXT:   %"arrayidx'ipg_unwrap" = getelementptr inbounds double, double* %"tmp'", i64 %_unwrap3
; CHECK-NEXT:   %8 = load double, double* %"arrayidx'ipg_unwrap", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"arrayidx'ipg_unwrap", align 8
; CHECK-NEXT:   %_unwrap5 = load i64, i64* %.omp.lb_smpl
; CHECK-NEXT:   %9 = add nuw nsw i64 %"iv'ac.0", %_unwrap5
; CHECK-NEXT:   %10 = getelementptr inbounds double, double* %truetape, i64 %9
; CHECK-NEXT:   %11 = load double, double* %10, align 8, !invariant.group !20
; CHECK-NEXT:   %12 = call fast double @llvm.sqrt.f64(double %11)
; CHECK-NEXT:   %13 = fmul fast double 5.000000e-01, %8
; CHECK-NEXT:   %14 = fdiv fast double %13, %12
; CHECK-NEXT:   %15 = fcmp fast oeq double %11, 0.000000e+00
; CHECK-NEXT:   %16 = select fast i1 %15, double 0.000000e+00, double %14
; CHECK-NEXT:   %17 = atomicrmw fadd double* %"arrayidx'ipg_unwrap", double %16 monotonic
; CHECK-NEXT:   %18 = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %18, label %invertomp.precond.then, label %incinvertomp.inner.for.body

; CHECK: incinvertomp.inner.for.body:                      ; preds = %invertomp.inner.for.body
; CHECK-NEXT:   %19 = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertomp.inner.for.body

; CHECK: invertomp.loop.exit.loopexit:                     ; preds = %invertomp.loop.exit
; CHECK-NEXT:   %_unwrap7 = load i64, i64* %.omp.ub_smpl
; CHECK-NEXT:   %cmp6_unwrap = icmp ugt i64 %_unwrap7, %sub4
; CHECK-NEXT:   %cond_unwrap = select i1 %cmp6_unwrap, i64 %sub4, i64 %_unwrap7
; CHECK-NEXT:   %_unwrap8 = load i64, i64* %.omp.lb_smpl
; CHECK-NEXT:   %_unwrap9 = sub i64 %cond_unwrap, %_unwrap8
; CHECK-NEXT:   br label %invertomp.inner.for.body

; CHECK: invertomp.loop.exit:                              ; preds = %invertomp.precond.end
; CHECK-NEXT:   %_unwrap10 = load i64, i64* %.omp.lb_smpl
; CHECK-NEXT:   %_unwrap11 = load i64, i64* %.omp.ub_smpl
; CHECK-NEXT:   %cmp6_unwrap12 = icmp ugt i64 %_unwrap11, %sub4
; CHECK-NEXT:   %cond_unwrap13 = select i1 %cmp6_unwrap12, i64 %sub4, i64 %_unwrap11
; CHECK-NEXT:   %add29_unwrap = add i64 %cond_unwrap13, 1
; CHECK-NEXT:   %cmp730_unwrap = icmp ult i64 %_unwrap10, %add29_unwrap
; CHECK-NEXT:   br i1 %cmp730_unwrap, label %invertomp.loop.exit.loopexit, label %invertomp.precond.then

; CHECK: invertomp.precond.end:                            ; preds = %omp.inner.for.body, %omp.precond.then
; CHECK-NEXT:   br i1 %cmp.not, label %invertentry, label %invertomp.loop.exit
; CHECK-NEXT: }

