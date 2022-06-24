; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -early-cse -instsimplify -simplifycfg -adce -S | FileCheck %s

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
; CHECK-DAG:   %[[a2:.+]] = fmul fast double %"x'", 2.000000e+00
; CHECK-DAG:   %[[a0:.+]] = fmul fast double %x, %x
; CHECK-DAG:   %[[a1:.+]] = fadd fast double %[[a0]], 4.000000e+00
; CHECK-DAG:   %[[a4:.+]] = fdiv fast double %[[a2]], %[[a1]]
; CHECK-DAG:   %[[a7:.+]] = {{(fneg fast double)|(fsub fast double \-0.000000e\+00,)}} %[[a4]]
; CHECK-NEXT:   ret double %[[a7]]
; CHECK-NEXT: }

