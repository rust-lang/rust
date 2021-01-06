; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

; Function Attrs: noinline nounwind readnone uwtable
define double @tester(double %x, double %y) {
entry:
  %0 = tail call fast double @llvm.copysign.f64(double %x, double %y)
  ret double %0
}

define double @test_derivative(double %x, double %y) {
entry:
  %0 = tail call double (double (double, double)*, ...) @__enzyme_autodiff(double (double, double)* nonnull @tester, double %x, double %y)
  ret double %0
}

declare double @llvm.copysign.f64(double, double)

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double, double)*, ...)

; CHECK: define internal {{(dso_local )?}}{ double, double } @diffetester(double %x, double %y, double %[[differet:.+]])
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call fast double @llvm.copysign.f64(double 1.000000e+00, double %x)
; CHECK-NEXT:   %1 = call fast double @llvm.copysign.f64(double 1.000000e+00, double %y)
; CHECK-NEXT:   %2 = fmul fast double %0, %1
; CHECK-NEXT:   %3 = fmul fast double %2, %[[differet]]
; CHECK-NEXT:   %4 = insertvalue { double, double } undef, double %3, 0
; CHECK-NEXT:   %5 = insertvalue { double, double } %4, double 0.000000e+00, 1
; CHECK-NEXT:   ret { double, double } %5
; CHECK-NEXT: }
