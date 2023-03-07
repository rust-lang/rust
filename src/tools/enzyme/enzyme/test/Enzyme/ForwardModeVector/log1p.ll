; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -gvn -simplifycfg -instcombine -S | FileCheck %s

%struct.Gradients = type { double, double, double }

; Function Attrs: nounwind
declare %struct.Gradients @__enzyme_fwddiff(double (double)*, ...)

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x) {
entry:
  %0 = tail call double @log1p(double %x)
  ret double %0
}

define %struct.Gradients @test_derivative(double %x) {
entry:
  %0 = tail call %struct.Gradients (double (double)*, ...) @__enzyme_fwddiff(double (double)* nonnull @tester, metadata !"enzyme_width", i64 3, double %x, double 1.0, double 2.0, double 3.0)
  ret %struct.Gradients %0
}

; Function Attrs: nounwind readnone speculatable
declare double @log1p(double)

; CHECK: define internal [3 x double] @fwddiffe3tester(double %x, [3 x double] %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = fadd fast double %x, 1.000000e+00
; CHECK-NEXT:   %1 = extractvalue [3 x double] %"x'", 0
; CHECK-NEXT:   %2 = fdiv fast double %1, %0
; CHECK-NEXT:   %3 = insertvalue [3 x double] undef, double %2, 0
; CHECK-NEXT:   %4 = extractvalue [3 x double] %"x'", 1
; CHECK-NEXT:   %5 = fdiv fast double %4, %0
; CHECK-NEXT:   %6 = insertvalue [3 x double] %3, double %5, 1
; CHECK-NEXT:   %7 = extractvalue [3 x double] %"x'", 2
; CHECK-NEXT:   %8 = fdiv fast double %7, %0
; CHECK-NEXT:   %9 = insertvalue [3 x double] %6, double %8, 2
; CHECK-NEXT:   ret [3 x double] %9
; CHECK-NEXT: }
