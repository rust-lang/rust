; RUN: opt < %s %loadEnzyme -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

; Function Attrs: noinline nounwind readnone uwtable
define double @tester(double %x) {
entry:
  ret double 1.000000e+00
}

define double @test_derivative(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_autodiff(double (double)* nonnull @tester, double %x)
  ret double %0
}

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double)*, ...)

; CHECK: define internal {{(dso_local )?}}{ double } @diffetester(double %x, double %[[differet:.+]])
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret { double } zeroinitializer
; CHECK-NEXT: }
