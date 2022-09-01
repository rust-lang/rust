; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -simplifycfg -instsimplify -S | FileCheck %s

%struct.Gradients = type { double, double }

define double @f(double %x, i1 %c) {
entry:
  %v = select i1 %c, double 0.000000e+00, double 1.000000e+00
  ret double %v
}

define double @tester(double %x, double %y) {
entry:
  %c = call double @f(double %x, i1 true)
  %mul = fmul double %c, %y
  ret double %mul
}

define %struct.Gradients @test_derivative(double %x, double %y){
entry:
  %call = call %struct.Gradients (double (double, double)*, ...) @__enzyme_fwddiff(double (double, double)* nonnull @tester, metadata !"enzyme_width", i64 2, double %x, double 1.000000e+00, double 0.000000e+00, double %y, double 0.000000e+00, double 1.000000e+00)
  ret %struct.Gradients %call
}

declare %struct.Gradients @__enzyme_fwddiff(double (double, double)*, ...)

; CHECK: define internal { double, [2 x double] } @fwddiffe2f(double %x, [2 x double] %"x'", i1 %c)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %v = select i1 %c, double 0.000000e+00, double 1.000000e+00
; CHECK-NEXT:   %0 = insertvalue { double, [2 x double] } undef, double %v, 0
; CHECK-NEXT:   %1 = insertvalue { double, [2 x double] } %0, [2 x double] zeroinitializer, 1
; CHECK-NEXT:   ret { double, [2 x double] } %1
; CHECK-NEXT: }
