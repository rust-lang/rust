; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -early-cse -simplifycfg -S | FileCheck %s

%struct.Gradients = type { double, double }

; Function Attrs: nounwind
declare %struct.Gradients @__enzyme_fwddiff(double (double, double)*, ...)

; Function Attrs: noinline nounwind readnone uwtable
define double @tester(double %x, double %y) {
entry:
  %0 = fmul fast double %x, %y
  ret double %0
}

define %struct.Gradients @test_derivative(double %x, double %y) {
entry:
  %0 = tail call %struct.Gradients (double (double, double)*, ...) @__enzyme_fwddiff(double (double, double)* nonnull @tester, metadata !"enzyme_width", i64 2, double %x, double 1.0, double 0.0, double %y, double 0.0, double 1.0)
  ret %struct.Gradients %0
}


; CHECK: define internal [2 x double] @fwddiffe2tester(double %x, [2 x double] %"x'", double %y, [2 x double] %"y'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = extractvalue [2 x double] %"x'", 0
; CHECK-NEXT:   %1 = extractvalue [2 x double] %"y'", 0
; CHECK-NEXT:   %2 = fmul fast double %0, %y
; CHECK-NEXT:   %3 = fmul fast double %1, %x
; CHECK-NEXT:   %4 = fadd fast double %2, %3
; CHECK-NEXT:   %5 = insertvalue [2 x double] undef, double %4, 0
; CHECK-NEXT:   %6 = extractvalue [2 x double] %"x'", 1
; CHECK-NEXT:   %7 = extractvalue [2 x double] %"y'", 1
; CHECK-NEXT:   %8 = fmul fast double %6, %y
; CHECK-NEXT:   %9 = fmul fast double %7, %x
; CHECK-NEXT:   %10 = fadd fast double %8, %9
; CHECK-NEXT:   %11 = insertvalue [2 x double] %5, double %10, 1
; CHECK-NEXT:   ret [2 x double] %11
; CHECK-NEXT: }