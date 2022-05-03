; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -instsimplify -simplifycfg -adce -S | FileCheck %s

define double @tester2(double %x) {
entry:
  %call = call double @atan2(double 2.000000e+00, double %x)
  ret double %call
}

define double @test_derivative(double %y, double %x) {
entry:
  %0 = tail call double (...) @__enzyme_fwddiff(double (double)* nonnull @tester2, double %x, double 1.000000e+00)
  ret double %0
}

declare double @atan2(double, double)

; Function Attrs: nounwind
declare double @__enzyme_fwddiff(...)

; CHECK-LABEL: define internal double @fwddiffetester2(
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = fmul fast double %x, %x
; CHECK-NEXT:   %1 = fadd fast double %0, 4.000000e+00
; CHECK-NEXT:   %2 = fmul fast double %"x'", 2.000000e+00
; CHECK-NEXT:   %3 = fsub fast double 0.000000e+00, %2
; CHECK-NEXT:   %4 = fdiv fast double %3, %1
; CHECK-NEXT:   ret double %4
; CHECK-NEXT: }

