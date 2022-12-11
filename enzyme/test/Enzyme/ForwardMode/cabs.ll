; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x, double %y) {
entry:
  %call = call double @cabs(double %x, double %y)
  ret double %call
}

define double @test_derivative(double %x, double %y) {
entry:
  %0 = tail call double (double (double, double)*, ...) @__enzyme_fwddiff(double (double, double)* nonnull @tester, double %x, double 1.0, double %y, double 1.0)
  ret double %0
}

declare double @cabs(double, double)

; Function Attrs: nounwind
declare double @__enzyme_fwddiff(double (double, double)*, ...)


; CHECK: define internal double @fwddiffetester(double %x, double %"x'", double %y, double %"y'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call fast double @cabs(double %x, double %y)
; CHECK-NEXT:   %1 = fdiv fast double %"x'", %0
; CHECK-NEXT:   %2 = fmul fast double %x, %1
; CHECK-NEXT:   %3 = fdiv fast double %"y'", %0
; CHECK-NEXT:   %4 = fmul fast double %y, %3
; CHECK-NEXT:   %5 = fadd fast double %2, %4
; CHECK-NEXT:   ret double %5
; CHECK-NEXT: }
