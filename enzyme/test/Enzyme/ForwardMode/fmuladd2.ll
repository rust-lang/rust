; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -gvn -simplifycfg -instcombine -S | FileCheck %s

; Function Attrs: nounwind readnone uwtable
define double @tester(double* %yp) {
entry:
  %y = load double, double* %yp, align 8
  %0 = tail call fast double @llvm.fmuladd.f64(double 1.000000e+00, double %y, double 0.000000e+00)
  ret double %0
}

define double @test_derivative(double* %yp, double* %dyp) {
entry:
  %0 = tail call double (double (double*)*, ...) @__enzyme_fwddiff(double (double*)* nonnull @tester, double* %yp, double* %dyp)
  ret double %0
}

declare double @llvm.fmuladd.f64(double %a, double %b, double %c)

; Function Attrs: nounwind
declare double @__enzyme_fwddiff(double (double*)*, ...)

; CHECK: define internal double @fwddiffetester(double* %yp, double* %"yp'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"y'ipl" = load double, double* %"yp'", align 8
; CHECK-NEXT:   ret double %"y'ipl"
; CHECK-NEXT: }
