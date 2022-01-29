; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -simplifycfg -adce -instsimplify -S | FileCheck %s

%struct.Gradients = type { float, float }
%struct.ExtGradients = type { double, double }

; Function Attrs: nounwind
declare %struct.ExtGradients @__enzyme_fwddiff(double (float)*, ...)

define double @tester(float %x) {
entry:
  %y = fpext float %x to double
  ret double %y
}

define %struct.ExtGradients @test_derivative(float %x) {
entry:
  %0 = tail call %struct.ExtGradients (double (float)*, ...) @__enzyme_fwddiff(double (float)* nonnull @tester,  metadata !"enzyme_width", i64 2, float %x, float 1.0, float 2.0)
  ret %struct.ExtGradients %0
}


; CHECK: define internal [2 x double] @fwddiffe2tester(float %x, [2 x float] %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = extractvalue [2 x float] %"x'", 0
; CHECK-NEXT:   %1 = fpext float %0 to double
; CHECK-NEXT:   %2 = insertvalue [2 x double] undef, double %1, 0
; CHECK-NEXT:   %3 = extractvalue [2 x float] %"x'", 1
; CHECK-NEXT:   %4 = fpext float %3 to double
; CHECK-NEXT:   %5 = insertvalue [2 x double] %2, double %4, 1
; CHECK-NEXT:   ret [2 x double] %5
; CHECK-NEXT: }