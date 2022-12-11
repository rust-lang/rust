; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x, double %y, double %z) {
entry:
  %0 = tail call fast double @llvm.fmuladd.f64(double %x, double %y, double %z)
  ret double %0
}

define double @test_derivative(double %x, double %y, double %z) {
entry:
  %0 = tail call double (double (double, double, double)*, ...) @__enzyme_fwddiff(double (double, double, double)* nonnull @tester, double %x, double %x, double %y, double %y, double %z, double %z)
  ret double %0
}

declare double @llvm.fmuladd.f64(double %a, double %b, double %c)

; Function Attrs: nounwind
declare double @__enzyme_fwddiff(double (double, double, double)*, ...)

; CHECK: define internal double @fwddiffetester(double %x, double %"x'", double %y, double %"y'", double %z, double %"z'")
; CHECK-NEXT: entry:
; CHECK-DAG:   %[[i0:.+]] = fmul fast double %y, %"x'"
; CHECK-DAG:   %[[i1:.+]] = fmul fast double %x, %"y'"
; CHECK-NEXT:   %2 = fadd fast double %[[i1]], %[[i0]]
; CHECK-NEXT:   %3 = fadd fast double %2, %"z'"
; CHECK-NEXT:   ret double %3
; CHECK-NEXT: }
