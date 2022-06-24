; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -early-cse -instsimplify -simplifycfg -adce -S | FileCheck %s

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
; CHECK-DAG:   %[[a0:.+]] = fmul fast double %y, %y
; CHECK-DAG:   %[[a1:.+]] = fadd fast double 4.000000e+00, %[[a0]]
; CHECK-DAG:   %[[a2:.+]] = fmul fast double %"y'", 2.000000e+00
; CHECK-DAG:   %[[a3:.+]] = fdiv fast double %[[a2]], %[[a1]]
; CHECK-NEXT:   ret double %3
; CHECK-NEXT: }

