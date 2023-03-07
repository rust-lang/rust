; RUN: if [ %llvmver -ge 9 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -adce -simplifycfg -S | FileCheck %s; fi

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
  %data = alloca double, align 8
  %d_data = alloca double, align 8
  call void @_Z17__enzyme_autodiffPvS_S_m(i8* bitcast (void (double*, double*)* @_ZL16LagrangeLeapFrogPdm to i8*), double* %data, double* %d_data, double* %data, double* %d_data)
  ret i32 0
}

declare dso_local void @_Z17__enzyme_autodiffPvS_S_m(i8*, double*, double*, double*, double*)

; Function Attrs: inlinehint nounwind uwtable mustprogress
define internal void @_ZL16LagrangeLeapFrogPdm(double* nocapture readonly noalias %e_new, double* noalias nocapture %out) #3 {
entry:
  tail call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%struct.ident_t* nonnull @2, i32 2, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, double*, double*)* @.omp_outlined. to void (i32*, i32*, ...)*), double* %e_new, double* %out)
  ret void
}


declare i64 @omp_get_thread_num()

declare void @julia.write_barrier(double* nocapture) readnone

; Function Attrs: norecurse nounwind uwtable
define internal void @.omp_outlined.(i32* noalias nocapture readonly %.global_tid., i32* noalias nocapture readnone %.bound_tid., double* readonly nocapture noalias %x, double* noalias nocapture %out) #4 {
entry:
  %m = alloca double, align 16
  ; A fake barrier here is added to prevent %m from being mem2reg'd away
  call void @julia.write_barrier(double* %m)
  %prev = load double, double* %x, align 8
  store double %prev, double* %m, align 8
  %t = call i64 @omp_get_thread_num()
  %gep = getelementptr inbounds double, double* %out, i64 %t
  %mload = load double, double* %m, align 8
  store double %mload, double* %gep, align 8
  ret void
}

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

; CHECK: define internal void @diffe.omp_outlined.(i32* noalias nocapture readonly %.global_tid., i32* noalias nocapture readnone %.bound_tid., double* noalias nocapture readonly %x, double* nocapture %"x'", double* noalias nocapture %out, double* nocapture %"out'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"m'ai" = alloca double, i64 1, align 16
; CHECK-NEXT:   %0 = bitcast double* %"m'ai" to i8*
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull dereferenceable(8) dereferenceable_or_null(8) %0, i8 0, i64 8, i1 false)
; CHECK-NEXT:   call void @julia.write_barrier(double* %"m'ai")
; CHECK-NEXT:   %t = call i64 @omp_get_thread_num()
; CHECK-NEXT:   %"gep'ipg" = getelementptr inbounds double, double* %"out'", i64 %t
; CHECK-NEXT:   %[[i0:.+]] = load double, double* %"gep'ipg", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"gep'ipg", align 8
; CHECK-NEXT:   %[[i1:.+]] = load double, double* %"m'ai", align 8
; CHECK-NEXT:   %[[i2:.+]] = fadd fast double %[[i1]], %[[i0]]
; CHECK-NEXT:   store double %[[i2]], double* %"m'ai", align 8
; CHECK-NEXT:   %[[i3:.+]] = load double, double* %"m'ai", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"m'ai", align 8
; CHECK-NEXT:   %[[i4:.+]] = atomicrmw fadd double* %"x'", double %[[i3]] monotonic
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
