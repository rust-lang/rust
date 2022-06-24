; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -early-cse -instcombine -simplifycfg -adce -S | FileCheck %s

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
; CHECK-DAG:   %[[a3:.+]] = fmul fast double %"y'", %x
; CHECK-DAG:    %[[a1:.+]] = fmul fast double %x, %x
; CHECK-DAG:    %[[a0:.+]] = fmul fast double %y, %y
; CHECK-DAG:   %[[a2:.+]] = fadd fast double %[[a1]], %[[a0]]
; CHECK-DAG:   %[[a4:.+]] = fmul fast double %"x'", %y
; CHECK-DAG:   %[[a5:.+]] = fsub fast double %[[a3]], %[[a4]]
; CHECK-DAG:   %[[a6:.+]] = fdiv fast double %[[a5]], %[[a2]]
; CHECK-NEXT:   ret double %[[a6]]
; CHECK-NEXT: }

