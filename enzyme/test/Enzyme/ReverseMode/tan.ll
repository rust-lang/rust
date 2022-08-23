; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x) {
entry:
  %0 = tail call fast double @tan(double %x)
  ret double %0
}

define double @test_derivative(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_autodiff(double (double)* nonnull @tester, double %x)
  ret double %0
}

; Function Attrs: nounwind readnone speculatable
declare double @tan(double)

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double)*, ...)

; CHECK: define internal { double } @diffetester(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call fast double @tan(double %x) 
; CHECK-NEXT:   %1 = fmul fast double %0, %0
; CHECK-NEXT:   %2 = fadd fast double 1.000000e+00, %1
; CHECK-NEXT:   %3 = fmul fast double %differeturn, %2
; CHECK-NEXT:   %4 = insertvalue { double } undef, double %3, 0
; CHECK-NEXT:   ret { double } %4
; CHECK-NEXT: }
