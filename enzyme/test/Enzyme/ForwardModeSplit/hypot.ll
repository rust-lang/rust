; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -instsimplify -simplifycfg -adce -S | FileCheck %s

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x, double %y) {
entry:
  %call = call double @hypot(double %x, double %y)
  ret double %call
}

define double @tester2(double %x) {
entry:
  %call = call double @hypot(double %x, double 2.000000e+00)
  ret double %call
}

define double @test_derivative(double %x, double %y) {
entry:
  %0 = tail call double (...) @__enzyme_fwdsplit(double (double, double)* nonnull @tester, double %x, double 1.000000e+00, double %y, double 1.000000e+00, i8* null)
  %1 = tail call double (...) @__enzyme_fwdsplit(double (double)* nonnull @tester2, double %x, double 1.000000e+00, i8* null)
  ret double %0
}

declare double @hypot(double, double)

; Function Attrs: nounwind
declare double @__enzyme_fwdsplit(...)

; CHECK-LABEL: define internal double @fwddiffetester(
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call fast double @hypot(double %x, double %y)
; CHECK-NEXT:   %1 = fmul fast double %x, %"x'"
; CHECK-NEXT:   %2 = fmul fast double %y, %"y'"
; CHECK-NEXT:   %3 = fadd fast double %1, %2
; CHECK-NEXT:   %4 = fdiv fast double %3, %0
; CHECK-NEXT:   ret double %4
; CHECK-NEXT: }

; CHECK-LABEL: define internal double @fwddiffetester2(
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call fast double @hypot(double %x, double 2.000000e+00)
; CHECK-NEXT:   %1 = fmul fast double %x, %"x'"
; CHECK-NEXT:   %2 = fdiv fast double %1, %0
; CHECK-NEXT:   ret double %2
; CHECK-NEXT: }

