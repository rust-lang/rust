; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -inline -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

; Function Attrs: noinline nounwind readnone uwtable
define double @tester(double %x, double %y) {
entry:
  %0 = fadd fast double %x, %y
  ret double %0
}

define { double, double } @test_derivative(double %x, double %y) {
entry:
  %0 = tail call { double, double } (double (double, double)*, ...) @__enzyme_autodiff(double (double, double)* nonnull @tester, double %x, double %y)
  ret { double, double } %0
}

; Function Attrs: nounwind
declare { double, double } @__enzyme_autodiff(double (double, double)*, ...)

; CHECK: define { double, double } @test_derivative(double %x, double %y)
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret { double, double } { double 1.000000e+00, double 1.000000e+00 }
; CHECK-NEXT: }
