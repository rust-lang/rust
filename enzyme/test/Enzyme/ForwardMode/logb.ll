; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

declare double @__enzyme_fwddiff(i8*, ...)
declare double @logb(double)

define double @test(double %x) {
entry:
  %call = call double @logb(double %x)
  ret double %call
}

define double @test_derivative(double %x) {
entry:
  %call = call double (i8*, ...) @__enzyme_fwddiff(i8* bitcast (double (double)* @test to i8*), double %x, double 1.0)
  ret double %call
}


; CHECK: define internal double @fwddiffetest(double %x, double %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret double 0.000000e+00
; CHECK-NEXT: }
