; RUN: if [ %llvmver -ge 11 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=1 -mem2reg -instsimplify -adce -simplifycfg -S | FileCheck %s; fi

source_filename = "/home/ubuntu/LULESH-MPI-RAJA/lulesh-v2.0/RAJA/lulesh.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.ident_t = type { i32, i32, i32, i32, i8* }
@0 = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
@1 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 22, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @0, i32 0, i32 0) }, align 8
@2 = private unnamed_addr constant %struct.ident_t { i32 0, i32 514, i32 0, i32 22, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @0, i32 0, i32 0) }, align 8
@3 = private unnamed_addr constant %struct.ident_t { i32 0, i32 66, i32 0, i32 22, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @0, i32 0, i32 0) }, align 8

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture)

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture)

define void @caller(i8* %call18, i8* %call27) {
entry:
  call void @_Z17__enzyme_autodiffPvS_S_(i8* bitcast (void (i64**, double*, i64)* @_ZL16LagrangeLeapFrogP6Domain to i8*), i8* %call18, i8* %call18, i8* %call27, i64 10)
  ret void
}

declare i32 @__gxx_personality_v0(...)

declare void @_Z17__enzyme_autodiffPvS_S_(i8*, i8*, i8*, i8*, i64)

; Function Attrs: inlinehint nounwind uwtable
define internal void @_ZL16LagrangeLeapFrogP6Domain(i64** noalias %i12p, double* noalias %i13, i64 %a.val3) {
entry:
  %i12 = load i64*, i64** %i12p, align 8
  call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%struct.ident_t* nonnull @1, i32 3, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, i64, i64*, double*)* @.omp_outlined. to void (i32*, i32*, ...)*), i64 %a.val3, i64* nonnull %i12, double* %i13)
  ret void
}

; Function Attrs: alwaysinline norecurse nounwind uwtable
define internal void @.omp_outlined.(i32* noalias nocapture noundef readnone %.global_tid., i32* noalias nocapture noundef readnone %.bound_tid., i64 %.val3, i64* noalias %i12, double* noalias %i13) personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %.omp.lb.i.i = alloca i64, align 8
  %.omp.ub.i.i = alloca i64, align 8
  %.omp.stride.i.i = alloca i64, align 8
  %.omp.is_last.i.i = alloca i32, align 4
  %i4 = tail call i32 @__kmpc_global_thread_num(%struct.ident_t* nonnull @1)
  %sub9.i.i = add nsw i64 %.val3, -1
  store i64 0, i64* %.omp.lb.i.i, align 8
  store i64 %sub9.i.i, i64* %.omp.ub.i.i, align 8
  store i64 1, i64* %.omp.stride.i.i, align 8
  store i32 0, i32* %.omp.is_last.i.i, align 4
  invoke void @__kmpc_for_static_init_8(%struct.ident_t* nonnull @2, i32 %i4, i32 34, i32* nonnull %.omp.is_last.i.i, i64* nonnull %.omp.lb.i.i, i64* nonnull %.omp.ub.i.i, i64* nonnull %.omp.stride.i.i, i64 1, i64 1)
          to label %.noexc unwind label %terminate.lpad

.noexc:                                           ; preds = %entry
  %i9 = load i64, i64* %.omp.ub.i.i, align 8
  %cmp11.i.i = icmp sgt i64 %i9, %sub9.i.i
  %cond.i.i = select i1 %cmp11.i.i, i64 %sub9.i.i, i64 %i9
  store i64 %cond.i.i, i64* %.omp.ub.i.i, align 8
  %i10 = load i64, i64* %.omp.lb.i.i, align 8
  %cmp12.not3.i.i = icmp sgt i64 %i10, %cond.i.i
  br i1 %cmp12.not3.i.i, label %omp.loop.exit.i.i, label %omp.inner.for.body.lr.ph.i.i

omp.inner.for.body.lr.ph.i.i:                     ; preds = %.noexc
  br label %omp.inner.for.body.i.i

omp.inner.for.body.i.i:                           ; preds = %omp.inner.for.inc.i.i, %omp.inner.for.body.lr.ph.i.i
  %.omp.iv.04.i.i = phi i64 [ %i10, %omp.inner.for.body.lr.ph.i.i ], [ %add15.i.i, %omp.inner.for.inc.i.i ]
  %sub.i.i.i.i = load i64, i64* %i12, align 8
  br label %for.body.i.i.i

for.body.i.i.i:                                   ; preds = %for.body.i.i.i, %omp.inner.for.body.i.i
  %i.03.i.i.i = phi i64 [ %inc.i.i.i, %for.body.i.i.i ], [ 0, %omp.inner.for.body.i.i ]
  %inc.i.i.i = add nuw nsw i64 %i.03.i.i.i, 1
  %exitcond.not.i.i.i = icmp eq i64 %inc.i.i.i, %sub.i.i.i.i
  br i1 %exitcond.not.i.i.i, label %omp.inner.for.inc.i.i, label %for.body.i.i.i

omp.inner.for.inc.i.i:                            ; preds = %for.body.i.i.i
  %add.ptr.i.i.i.i.i = getelementptr inbounds double, double* %i13, i64 %.omp.iv.04.i.i
  store double 1.000000e+00, double* %add.ptr.i.i.i.i.i, align 8
  %add15.i.i = add i64 %.omp.iv.04.i.i, 1
  %exitcond.not.i.i = icmp eq i64 %.omp.iv.04.i.i, %cond.i.i
  br i1 %exitcond.not.i.i, label %omp.loop.exit.i.i, label %omp.inner.for.body.i.i

omp.loop.exit.i.i:                                ; preds = %omp.inner.for.inc.i.i, %.noexc
  call void @__kmpc_for_static_fini(%struct.ident_t* nonnull @2, i32 %i4)
  ret void

terminate.lpad:                                   ; preds = %entry
  %i16 = landingpad { i8*, i32 }
          catch i8* null
  unreachable
}

; Function Attrs: nounwind
declare void @__kmpc_fork_call(%struct.ident_t* nocapture readonly, i32, void (i32*, i32*, ...)* nocapture readonly, ...)

declare i32 @__kmpc_global_thread_num(%struct.ident_t* nocapture readonly)

declare void @__kmpc_for_static_init_8(%struct.ident_t*, i32, i32, i32*, i64*, i64*, i64*, i64, i64)

declare void @__kmpc_for_static_fini(%struct.ident_t* nocapture readonly, i32) 

; CHECK: diffe.omp_outlined
