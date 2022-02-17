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
  call void @llvm.lifetime.start.p0i8(i64 800, i8* nonnull %0) #5
  %1 = bitcast [100 x double]* %d_data to i8*
  call void @llvm.lifetime.start.p0i8(i64 800, i8* nonnull %1) #5
  call void @_Z17__enzyme_autodiffPvS_S_m(i8* bitcast (void (double*, i64)* @_ZL16LagrangeLeapFrogPdm to i8*), i8* nonnull %0, i8* nonnull %1, i64 100) #5
  call void @llvm.lifetime.end.p0i8(i64 800, i8* nonnull %1) #5
  call void @llvm.lifetime.end.p0i8(i64 800, i8* nonnull %0) #5
  ret i32 0
}

; Function Attrs: argmemonly nofree nosync nounwind willreturn mustprogress
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

declare dso_local void @_Z17__enzyme_autodiffPvS_S_m(i8*, i8*, i8*, i64) local_unnamed_addr #2

; Function Attrs: inlinehint nounwind uwtable mustprogress
define internal void @_ZL16LagrangeLeapFrogPdm(double* %e_new, i64 %length) #3 {
entry:
  %e_new.addr = alloca double*, align 8
  store double* %e_new, double** %e_new.addr, align 8, !tbaa !3
  call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%struct.ident_t* nonnull @2, i32 2, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, i64, double**)* @.omp_outlined. to void (i32*, i32*, ...)*), i64 %length, double** nonnull %e_new.addr)
  ret void
}

; Function Attrs: argmemonly nofree nosync nounwind willreturn mustprogress
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: norecurse nounwind uwtable
define internal void @.omp_outlined.(i32* noalias nocapture readonly %.global_tid., i32* noalias nocapture readnone %.bound_tid., i64 %length, double** nocapture nonnull readonly align 8 dereferenceable(8) %e_new) #4 {
entry:
  %.pre = load double*, double** %e_new, align 8, !tbaa !3
  %.omp.lb = alloca i64, align 8
  %.omp.ub = alloca i64, align 8
  %.omp.stride = alloca i64, align 8
  %.omp.is_last = alloca i32, align 4
  %sub2 = add i64 %length, -1
  %0 = bitcast i64* %.omp.lb to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %0) #5
  store i64 0, i64* %.omp.lb, align 8, !tbaa !7
  %1 = bitcast i64* %.omp.ub to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1) #5
  store i64 %sub2, i64* %.omp.ub, align 8, !tbaa !7
  %2 = bitcast i64* %.omp.stride to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2) #5
  store i64 1, i64* %.omp.stride, align 8, !tbaa !7
  %3 = bitcast i32* %.omp.is_last to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %3) #5
  store i32 0, i32* %.omp.is_last, align 4, !tbaa !9
  %4 = load i32, i32* %.global_tid., align 4, !tbaa !9
  call void @__kmpc_for_static_init_8u(%struct.ident_t* nonnull @1, i32 %4, i32 34, i32* nonnull %.omp.is_last, i64* nonnull %.omp.lb, i64* nonnull %.omp.ub, i64* nonnull %.omp.stride, i64 1, i64 1)
  %5 = load i64, i64* %.omp.ub, align 8, !tbaa !7
  %cmp4 = icmp ugt i64 %5, %sub2
  %cond = select i1 %cmp4, i64 %sub2, i64 %5
  store i64 %cond, i64* %.omp.ub, align 8, !tbaa !7
  %6 = load i64, i64* %.omp.lb, align 8, !tbaa !7
  %add24 = add i64 %cond, 1
  %cmp525 = icmp ult i64 %6, %add24
  br i1 %cmp525, label %omp.inner.for.body.preheader, label %omp.loop.exit

omp.inner.for.body.preheader:                     ; preds = %omp.precond.then
  br label %omp.inner.for.body

omp.inner.for.body:                               ; preds = %omp.inner.for.body.preheader, %omp.inner.for.body
  %7 = phi double* [ %i9, %omp.inner.for.body ], [ %.pre, %omp.inner.for.body.preheader ]
  %.omp.iv.026 = phi i64 [ %add8, %omp.inner.for.body ], [ %6, %omp.inner.for.body.preheader ]
  %arrayidx = getelementptr inbounds double, double* %7, i64 %.omp.iv.026
  %i8 = load double, double* %arrayidx, align 8, !tbaa !11
  %call = call double @sqrt(double %i8) #5
  %i9 = load double*, double** %e_new, align 8, !tbaa !3
  %arrayidx7 = getelementptr inbounds double, double* %.pre, i64 %.omp.iv.026
  store double %call, double* %arrayidx7, align 8, !tbaa !11
  %add8 = add nuw i64 %.omp.iv.026, 1
  %i10 = load i64, i64* %.omp.ub, align 8, !tbaa !7
  %add = add i64 %i10, 1
  %cmp5 = icmp ult i64 %add8, %add
  br i1 %cmp5, label %omp.inner.for.body, label %omp.loop.exit

omp.loop.exit:                                    ; preds = %omp.inner.for.body, %omp.precond.then
  call void @__kmpc_for_static_fini(%struct.ident_t* nonnull @1, i32 %4)
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %3) #5
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2) #5
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1) #5
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %0) #5
  ret void
}

; Function Attrs: nounwind
declare dso_local void @__kmpc_for_static_init_8u(%struct.ident_t*, i32, i32, i32*, i64*, i64*, i64*, i64, i64) local_unnamed_addr #5

; Function Attrs: nofree nounwind willreturn mustprogress
declare dso_local double @sqrt(double) local_unnamed_addr #6

; Function Attrs: nounwind
declare void @__kmpc_for_static_fini(%struct.ident_t*, i32) local_unnamed_addr #5

; Function Attrs: nounwind
declare !callback !13 void @__kmpc_fork_call(%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) local_unnamed_addr #5

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}
!nvvm.annotations = !{}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"uwtable", i32 1}
!2 = !{!"clang version 13.0.0 (git@github.com:llvm/llvm-project 619bfe8bd23f76b22f0a53fedafbfc8c97a15f12)"}
!3 = !{!4, !4, i64 0}
!4 = !{!"any pointer", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}
!7 = !{!8, !8, i64 0}
!8 = !{!"long", !5, i64 0}
!9 = !{!10, !10, i64 0}
!10 = !{!"int", !5, i64 0}
!11 = !{!12, !12, i64 0}
!12 = !{!"double", !5, i64 0}
!13 = !{!14}
!14 = !{i64 2, i64 -1, i64 -1, i1 true}

; CHECK: define internal void @diffe.omp_outlined

; CHECK: %[[cmp:.+]] = icmp ne i64 %"iv'ac.0", 0
; CHECK-NEXT:  br i1 %[[cmp]], label %invertomp.inner.for.body_phirc, label %invertomp.inner.for.body_phimerge

; CHECK: invertomp.inner.for.body_phirc: 
; CHECK-NEXT:  %"i9'ipl_unwrap" = load double*, double** %"e_new'", align 8, !invariant.group !
; CHECK-NEXT:  br label %invertomp.inner.for.body_phimerge

; CHECK: invertomp.inner.for.body_phimerge:
; CHECK-NEXT:  %{{.*}} = phi double* [ %"i9'ipl_unwrap", %invertomp.inner.for.body_phirc ], [ %".pre'il_phi", %invertomp.inner.for.body ]
