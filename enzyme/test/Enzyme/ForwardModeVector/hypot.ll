; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -instsimplify -simplifycfg -adce -S | FileCheck %s

%struct.Gradients = type { double, double, double }

define double @tester(double %x, double %y) {
entry:
  %call = call double @hypot(double %x, double %y)
  ret double %call
}

define double @tester2(double %x) {
entry:
  %call = call double @hypot(double %x, double 2.000000e+00)
  ret double %call
}


define %struct.Gradients @test_derivative(double %x, double %y) {
entry:
  %0 = tail call %struct.Gradients (...) @__enzyme_fwddiff(double (double, double)* nonnull @tester, metadata !"enzyme_width", i64 3,  double %x, double 1.0, double 2.0, double 3.0, double %y, double 1.0, double 2.0, double 3.0)
  %1 = tail call %struct.Gradients (...) @__enzyme_fwddiff(double (double)* nonnull @tester2, metadata !"enzyme_width", i64 3, double %x, double 1.0, double 2.0, double 3.0)
  ret %struct.Gradients %0
}

declare double @hypot(double, double)

declare %struct.Gradients @__enzyme_fwddiff(...)

; CHECK: define internal [3 x double] @fwddiffe3tester(double %x, [3 x double] %"x'", double %y, [3 x double] %"y'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call fast double @hypot(double %x, double %y)
; CHECK-NEXT:   %1 = extractvalue [3 x double] %"x'", 0
; CHECK-NEXT:   %2 = extractvalue [3 x double] %"y'", 0
; CHECK-NEXT:   %3 = fmul fast double %x, %1
; CHECK-NEXT:   %4 = fmul fast double %y, %2
; CHECK-NEXT:   %5 = fadd fast double %3, %4
; CHECK-NEXT:   %6 = fdiv fast double %5, %0
; CHECK-NEXT:   %7 = insertvalue [3 x double] undef, double %6, 0
; CHECK-NEXT:   %8 = extractvalue [3 x double] %"x'", 1
; CHECK-NEXT:   %9 = extractvalue [3 x double] %"y'", 1
; CHECK-NEXT:   %10 = fmul fast double %x, %8
; CHECK-NEXT:   %11 = fmul fast double %y, %9
; CHECK-NEXT:   %12 = fadd fast double %10, %11
; CHECK-NEXT:   %13 = fdiv fast double %12, %0
; CHECK-NEXT:   %14 = insertvalue [3 x double] %7, double %13, 1
; CHECK-NEXT:   %15 = extractvalue [3 x double] %"x'", 2
; CHECK-NEXT:   %16 = extractvalue [3 x double] %"y'", 2
; CHECK-NEXT:   %17 = fmul fast double %x, %15
; CHECK-NEXT:   %18 = fmul fast double %y, %16
; CHECK-NEXT:   %19 = fadd fast double %17, %18
; CHECK-NEXT:   %20 = fdiv fast double %19, %0
; CHECK-NEXT:   %21 = insertvalue [3 x double] %14, double %20, 2
; CHECK-NEXT:   ret [3 x double] %21
; CHECK-NEXT: }

; CHECK: define internal [3 x double] @fwddiffe3tester2(double %x, [3 x double] %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call fast double @hypot(double %x, double 2.000000e+00)
; CHECK-NEXT:   %1 = extractvalue [3 x double] %"x'", 0
; CHECK-NEXT:   %2 = fmul fast double %x, %1
; CHECK-NEXT:   %3 = fdiv fast double %2, %0
; CHECK-NEXT:   %4 = insertvalue [3 x double] undef, double %3, 0
; CHECK-NEXT:   %5 = extractvalue [3 x double] %"x'", 1
; CHECK-NEXT:   %6 = fmul fast double %x, %5
; CHECK-NEXT:   %7 = fdiv fast double %6, %0
; CHECK-NEXT:   %8 = insertvalue [3 x double] %4, double %7, 1
; CHECK-NEXT:   %9 = extractvalue [3 x double] %"x'", 2
; CHECK-NEXT:   %10 = fmul fast double %x, %9
; CHECK-NEXT:   %11 = fdiv fast double %10, %0
; CHECK-NEXT:   %12 = insertvalue [3 x double] %8, double %11, 2
; CHECK-NEXT:   ret [3 x double] %12
; CHECK-NEXT: }


