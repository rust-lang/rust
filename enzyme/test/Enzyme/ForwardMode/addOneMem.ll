; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

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
  %0 = tail call double (void (double*)*, ...) @__enzyme_fwddiff(void (double*)* nonnull @addOneMem, double* %x, double* %xp)
  ret void
}

; Function Attrs: nounwind
declare double @__enzyme_fwddiff(void (double*)*, ...)

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0}
!3 = !{!"double", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}

; CHECK: define internal void @fwddiffeaddOneMem(double* nocapture %x, double* nocapture %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"'ipl" = load double, double* %"x'", align 8, !tbaa !2
; CHECK-NEXT:   %0 = load double, double* %x, align 8, !tbaa !2
; CHECK-NEXT:   %add = fadd fast double %0, 1.000000e+00
; CHECK-NEXT:   store double %add, double* %x, align 8, !tbaa !2
; CHECK-NEXT:   store double %"'ipl", double* %"x'", align 8, !tbaa !2
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
