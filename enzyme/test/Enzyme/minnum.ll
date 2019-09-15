; RUN: opt < %s %loadEnzyme -enzyme -enzyme_preopt=false -mem2reg -early-cse -simplifycfg -S | FileCheck %s

define double @tester(double %x, double %y) {
entry:
  %0 = tail call double @llvm.minnum.f64(double %x, double %y)
  ret double %0
}

define double @test_derivative(double %x, double %y) {
entry:
  %0 = tail call double (double (double, double)*, ...) @__enzyme_autodiff(double (double, double)* nonnull @tester, double %x, double %y)
  ret double %0
}

declare double @llvm.minnum.f64(double, double)

declare double @__enzyme_autodiff(double (double, double)*, ...)

; CHECK: define internal {{(dso_local )?}}{ double, double } @diffetester(double %x, double %y, double %[[differet:.+]])
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[cmp:.+]] = fcmp fast olt double %x, %y
; CHECK-NEXT:   %[[diffex:.+]] = select i1 %[[cmp]], double %[[differet]], double 0.000000e+00
; CHECK-NEXT:   %[[diffey:.+]] = select i1 %[[cmp]], double 0.000000e+00, double %[[differet]]
; CHECK-NEXT:   %[[iv0:.+]] = insertvalue { double, double } undef, double %[[diffex]], 0
; CHECK-NEXT:   %[[iv1:.+]] = insertvalue { double, double } %[[iv0]], double %[[diffey]], 1
; CHECK-NEXT:   ret { double, double } %[[iv1]]
; CHECK-NEXT: }
