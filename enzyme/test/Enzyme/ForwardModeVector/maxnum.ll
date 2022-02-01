; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -early-cse -simplifycfg -S | FileCheck %s

%struct.Gradients = type { double, double }

; Function Attrs: nounwind
declare %struct.Gradients @__enzyme_fwddiff(double (double, double)*, ...)

; Function Attrs: noinline nounwind readnone uwtable
define double @tester(double %x, double %y) {
entry:
  %0 = tail call double @llvm.maxnum.f64(double %x, double %y)
  ret double %0
}

define %struct.Gradients @test_derivative(double %x, double %y) {
entry:
  %0 = tail call %struct.Gradients (double (double, double)*, ...) @__enzyme_fwddiff(double (double, double)* nonnull @tester, metadata !"enzyme_width", i64 2, double %x, double 1.0, double 0.0, double %y, double 0.0, double 1.0)
  ret %struct.Gradients %0
}

; Function Attrs: nounwind readnone speculatable
declare double @llvm.maxnum.f64(double, double)


; CHECK: define internal [2 x double] @fwddiffe2tester(double %x, [2 x double] %"x'", double %y, [2 x double] %"y'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = fcmp fast olt double %x, %y
; CHECK-NEXT:   %1 = extractvalue [2 x double] %"x'", 0
; CHECK-NEXT:   %2 = extractvalue [2 x double] %"y'", 0
; CHECK-NEXT:   %3 = select {{(fast )?}}i1 %0, double %1, double %2
; CHECK-NEXT:   %4 = insertvalue [2 x double] undef, double %3, 0
; CHECK-NEXT:   %5 = extractvalue [2 x double] %"x'", 1
; CHECK-NEXT:   %6 = extractvalue [2 x double] %"y'", 1
; CHECK-NEXT:   %7 = select {{(fast )?}}i1 %0, double %5, double %6
; CHECK-NEXT:   %8 = insertvalue [2 x double] %4, double %7, 1
; CHECK-NEXT:   ret [2 x double] %8
; CHECK-NEXT: }