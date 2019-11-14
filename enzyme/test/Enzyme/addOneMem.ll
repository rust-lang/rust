; RUN: %opt < %s %loadEnzyme -enzyme -enzyme_preopt=false -inline -mem2reg -instsimplify -gvn -dse -dse -S | FileCheck %s

; __attribute__((noinline))
; void addOneMem(double *x) {
;     *x += 1;
; }
; 
; void test_derivative(double *x, double *xp) {
;   __builtin_autodiff(addOneMem, x, xp);
; }

; Function Attrs: noinline norecurse nounwind uwtable
define dso_local void @addOneMem(double* nocapture %x) {
entry:
  %0 = load double, double* %x, align 8, !tbaa !2
  %add = fadd fast double %0, 1.000000e+00
  store double %add, double* %x, align 8, !tbaa !2
  ret void
}

; Function Attrs: nounwind uwtable
define dso_local void @test_derivative(double* %x, double* %xp) local_unnamed_addr {
entry:
  %0 = tail call double (void (double*)*, ...) @__enzyme_autodiff(void (double*)* nonnull @addOneMem, double* %x, double* %xp)
  ret void
}

; Function Attrs: nounwind
declare double @__enzyme_autodiff(void (double*)*, ...)

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0}
!3 = !{!"double", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}


; CHECK: define dso_local void @test_derivative(double* %x, double* %xp) 
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = load double, double* %x, align 8, !tbaa !2
; CHECK-NEXT:   %[[plus1:.+]] = fadd fast double %0, 1.000000e+00
; CHECK-NEXT:   store double %[[plus1]], double* %x, align 8, !tbaa !2
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
