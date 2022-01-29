; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

%struct.Gradients = type { double, double, double }

; Function Attrs: nounwind
declare %struct.Gradients @__enzyme_fwddiff(double (double)*, ...)

; Function Attrs: noinline nounwind readnone uwtable
define double @tester(double %x) {
entry:
  %agg1 = insertvalue [3 x double] undef, double %x, 0
  %mul = fmul double %x, %x
  %agg2 = insertvalue [3 x double] %agg1, double %mul, 1
  %add = fadd double %mul, 2.0
  %agg3 = insertvalue [3 x double] %agg2, double %add, 2
  %res = extractvalue [3 x double] %agg2, 1
  ret double %res
}

define %struct.Gradients @test_derivative(double %x) {
entry:
  %0 = tail call %struct.Gradients (double (double)*, ...) @__enzyme_fwddiff(double (double)* nonnull @tester, metadata !"enzyme_width", i64 3, double %x, double 1.0, double 2.0, double 3.0)
  ret %struct.Gradients %0
}


; CHECK: define internal [3 x double] @fwddiffe3tester(double %x, [3 x double] %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = extractvalue [3 x double] %"x'", 0
; CHECK-NEXT:   %1 = extractvalue [3 x double] %"x'", 0
; CHECK-NEXT:   %2 = fmul fast double %0, %x
; CHECK-NEXT:   %3 = fmul fast double %1, %x
; CHECK-NEXT:   %4 = fadd fast double %2, %3
; CHECK-NEXT:   %5 = extractvalue [3 x double] %"x'", 1
; CHECK-NEXT:   %6 = extractvalue [3 x double] %"x'", 1
; CHECK-NEXT:   %7 = fmul fast double %5, %x
; CHECK-NEXT:   %8 = fmul fast double %6, %x
; CHECK-NEXT:   %9 = fadd fast double %7, %8
; CHECK-NEXT:   %10 = extractvalue [3 x double] %"x'", 2
; CHECK-NEXT:   %11 = extractvalue [3 x double] %"x'", 2
; CHECK-NEXT:   %12 = fmul fast double %10, %x
; CHECK-NEXT:   %13 = fmul fast double %11, %x
; CHECK-NEXT:   %14 = fadd fast double %12, %13
; CHECK-NEXT:   %15 = insertvalue [3 x double] undef, double %4, 0
; CHECK-NEXT:   %16 = insertvalue [3 x double] %15, double %9, 1
; CHECK-NEXT:   %17 = insertvalue [3 x double] %16, double %14, 2
; CHECK-NEXT:   ret [3 x double] %17
; CHECK-NEXT: }