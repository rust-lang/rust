; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -instsimplify -simplifycfg -adce -S | FileCheck %s

define double @tester2(double %x) {
entry:
  %call = call double @hypot(double %x, double 2.000000e+00)
  ret double %call
}

define double @test_derivative(double %x, double %y) {
entry:
  %0 = tail call double (...) @__enzyme_fwddiff(double (double)* nonnull @tester2, double %x, double 1.000000e+00)
  ret double %0
}

declare double @hypot(double, double)

; Function Attrs: nounwind
declare double @__enzyme_fwddiff(...)

; CHECK: define internal double @fwddiffetester2(
; CHECK-NEXT: entry:
; CHECK-DAG:   %[[a1:.+]] = fmul fast double %"x'", %x
; CHECK-DAG:   %[[a0:.+]] = call fast double @hypot(double %x, double 2.000000e+00)
; CHECK-DAG:   %[[a2:.+]] = fdiv fast double %[[a1]], %[[a0]]
; CHECK-DAG:   ret double %[[a2]]
