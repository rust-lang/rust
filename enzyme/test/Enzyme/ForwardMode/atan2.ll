; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -instsimplify -simplifycfg -adce -S | FileCheck %s

; Function Attrs: nounwind readnone uwtable
define double @tester(double %y, double %x) {
entry:
  %call = call double @atan2(double %y, double %x)
  ret double %call
}

define double @test_derivative(double %y, double %x) {
entry:
  %0 = tail call double (...) @__enzyme_fwddiff(double (double, double)* nonnull @tester, double %y, double 1.000000e+00, double %x, double 1.000000e+00)
  ret double %0
}

declare double @atan2(double, double)

; Function Attrs: nounwind
declare double @__enzyme_fwddiff(...)

; CHECK-LABEL: define internal double @fwddiffetester(
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = fmul fast double %y, %y
; CHECK-NEXT:   %1 = fmul fast double %x, %x
; CHECK-NEXT:   %2 = fadd fast double %1, %0
; CHECK-NEXT:   %3 = fmul fast double %"y'", %x
; CHECK-NEXT:   %4 = fmul fast double %"x'", %y
; CHECK-NEXT:   %5 = fsub fast double %3, %4
; CHECK-NEXT:   %6 = fdiv fast double %5, %2
; CHECK-NEXT:   ret double %6
; CHECK-NEXT: }

