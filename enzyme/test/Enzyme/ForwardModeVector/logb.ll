; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -simplifycfg -S | FileCheck %s

declare [3 x double] @__enzyme_fwddiff(i8*, ...)
declare double @logb(double)

define double @test(double %x) {
entry:
  %call = call double @logb(double %x)
  ret double %call
}

define [3 x double] @test_derivative(double %x) {
entry:
  %call = call [3 x double] (i8*, ...) @__enzyme_fwddiff(i8* bitcast (double (double)* @test to i8*), metadata !"enzyme_width", i64 3, double %x, double 1.0, double 0.1, double 10.0)
  ret [3 x double] %call
}


; CHECK: define internal [3 x double] @fwddiffe3test(double %x, [3 x double] %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret [3 x double] zeroinitializer
; CHECK-NEXT: }