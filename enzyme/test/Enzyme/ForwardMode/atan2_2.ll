; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -instsimplify -simplifycfg -adce -S | FileCheck %s

define double @tester2(double %y) {
entry:
  %call = call double @atan2(double %y, double 2.000000e+00)
  ret double %call
}

define double @test_derivative(double %y, double %x) {
entry:
  %0 = tail call double (...) @__enzyme_fwddiff(double (double)* nonnull @tester2, double %y, double 1.000000e+00)
  ret double %0
}

declare double @atan2(double, double)

; Function Attrs: nounwind
declare double @__enzyme_fwddiff(...)

; CHECK-LABEL: define internal double @fwddiffetester2(
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = fmul fast double %y, %y
; CHECK-NEXT:   %1 = fadd fast double 4.000000e+00, %0
; CHECK-NEXT:   %2 = fmul fast double %"y'", 2.000000e+00
; CHECK-NEXT:   %3 = fdiv fast double %2, %1
; CHECK-NEXT:   ret double %3
; CHECK-NEXT: }

