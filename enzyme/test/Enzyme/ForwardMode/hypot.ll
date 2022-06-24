; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -instsimplify -simplifycfg -early-cse -adce -S | FileCheck %s

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x, double %y) {
entry:
  %call = call double @hypot(double %x, double %y)
  ret double %call
}

define double @test_derivative(double %x, double %y) {
entry:
  %0 = tail call double (...) @__enzyme_fwddiff(double (double, double)* nonnull @tester, double %x, double 1.000000e+00, double %y, double 1.000000e+00)
  ret double %0
}

declare double @hypot(double, double)

; Function Attrs: nounwind
declare double @__enzyme_fwddiff(...)

; CHECK: define internal double @fwddiffetester(
; CHECK-NEXT: entry:
; CHECK-DAG:   %[[a1:.+]] = fmul fast double %"x'", %x
; CHECK-DAG:   %[[a0:.+]] = call fast double @hypot(double %x, double %y)
; CHECK-DAG:   %[[a2:.+]] = fmul fast double %"y'", %y
; CHECK-DAG:   %[[a40:.+]] = fdiv fast double %[[a1]], %[[a0]]
; CHECK-DAG:   %[[a41:.+]] = fdiv fast double %[[a2]], %[[a0]]
; CHECK-DAG:   %[[a3:.+]] = fadd fast double %[[a40]], %[[a41]]
; CHECK-DAG:   ret double %[[a3]]

