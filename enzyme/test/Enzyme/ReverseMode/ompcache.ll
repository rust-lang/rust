; RUN: if [ %llvmver -ge 9 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.ident_t = type { i32, i32, i32, i32, i8* }

@str = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
@anon = private unnamed_addr constant %struct.ident_t { i32 0, i32 66, i32 0, i32 22, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @str, i32 0, i32 0) }, align 8

declare dso_local void @_Z17__enzyme_autodiffIdEvPFdPKT_mEiS2_PS0_m(...)

define void @caller(double* %a, double* %da, i64 %b) {
entry:
  tail call void (...) @_Z17__enzyme_autodiffIdEvPFdPKT_mEiS2_PS0_m(void (double**, i64)* nonnull @f, double* %a, double* %da, i64 %b)
  ret void
}

declare void @__kmpc_fork_call(%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...)

define internal void @f(double** %arg, i64 %arg1) {
bb:
  call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%struct.ident_t* nonnull @anon, i32 2, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, i64, double**)* @outlined to void (i32*, i32*, ...)*), i64 %arg1, double** %arg)
  ret void
}

define internal void @outlined(i32* noalias %arg, i32* noalias %arg1, i64 %i10, double** %arg4) {
bb:
  %i14 = icmp eq i64 %i10, 0
  br i1 %i14, label %bb56, label %bb17

bb17:                                             ; preds = %bb
  %i33 = load double*, double** %arg4, align 8
  store double 0.000000e+00, double* %i33, align 8
  br label %bb56

bb56:                                             ; preds = %bb55, %bb15
  ret void
}

; CHECK: define internal void @augmented_outlined.1(i32* noalias %arg, i32* noalias %arg1, i64 %i10, double** %arg4, double** %"arg4'", double*** %tape)
; CHECK-NEXT: bb:
; CHECK-NEXT:   %0 = load double**, double*** %tape
; CHECK-NEXT:   %1 = call i64 @omp_get_max_threads()
; CHECK-NEXT:   %2 = call i64 @omp_get_thread_num()
; CHECK-NEXT:   %3 = getelementptr inbounds double*, double** %0, i64 %2
; CHECK-NEXT:   %i14 = icmp eq i64 %i10, 0
; CHECK-NEXT:   br i1 %i14, label %bb56, label %bb17

; CHECK: bb17:                                             ; preds = %bb
; CHECK-NEXT:   %"i33'ipl" = load double*, double** %"arg4'"
; CHECK-NEXT:   store double* %"i33'ipl", double** %3
; CHECK-NEXT:   %i33 = load double*, double** %arg4
; CHECK-NEXT:   store double 0.000000e+00, double* %i33, align 8
; CHECK-NEXT:   br label %bb56

; CHECK: bb56:                                             ; preds = %bb17, %bb
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @diffeoutlined(i32* noalias %arg, i32* noalias %arg1, i64 %i10, double** %arg4, double** %"arg4'", double*** %tapeArg)
; CHECK-NEXT: bb:
; CHECK-NEXT:   %"i33'il_phi_fromtape" = load double**, double*** %tapeArg
; CHECK-NEXT:   %0 = call i64 @omp_get_thread_num() 
; CHECK-NEXT:   %i14 = icmp eq i64 %i10, 0
; CHECK-NEXT:   br i1 %i14, label %invertbb, label %invertbb17

; CHECK: invertbb:                                         ; preds = %bb, %invertbb17
; CHECK-NEXT:   ret void

; CHECK: invertbb17:                                       ; preds = %bb
; CHECK-NEXT:   %1 = getelementptr inbounds double*, double** %"i33'il_phi_fromtape", i64 %0
; CHECK-NEXT:   %2 = load double*, double** %1
; CHECK-NEXT:   store double 0.000000e+00, double* %2
; CHECK-NEXT:   br label %invertbb
; CHECK-NEXT: }
