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
; CHECK-NEXT:   %0 = extractvalue [3 x double] %"x'", 0
; CHECK-NEXT:   %1 = fmul fast double %0, %x
; CHECK-NEXT:   %2 = extractvalue [3 x double] %"x'", 1
; CHECK-NEXT:   %3 = fmul fast double %2, %x
; CHECK-NEXT:   %4 = extractvalue [3 x double] %"x'", 2
; CHECK-NEXT:   %5 = fmul fast double %4, %x
; CHECK-NEXT:   %6 = call fast double @hypot(double %x, double %y)
; CHECK-NEXT:   %7 = fdiv fast double %1, %6
; CHECK-NEXT:   %8 = fdiv fast double %3, %6
; CHECK-NEXT:   %9 = fdiv fast double %5, %6
; CHECK-NEXT:   %10 = extractvalue [3 x double] %"y'", 0
; CHECK-NEXT:   %11 = fmul fast double %10, %y
; CHECK-NEXT:   %12 = extractvalue [3 x double] %"y'", 1
; CHECK-NEXT:   %13 = fmul fast double %12, %y
; CHECK-NEXT:   %14 = extractvalue [3 x double] %"y'", 2
; CHECK-NEXT:   %15 = fmul fast double %14, %y
; CHECK-NEXT:   %16 = call fast double @hypot(double %x, double %y)
; CHECK-NEXT:   %17 = fdiv fast double %11, %16
; CHECK-NEXT:   %18 = fdiv fast double %13, %16
; CHECK-NEXT:   %19 = fdiv fast double %15, %16
; CHECK-NEXT:   %20 = fadd fast double %7, %17
; CHECK-NEXT:   %21 = insertvalue [3 x double] undef, double %20, 0
; CHECK-NEXT:   %22 = fadd fast double %8, %18
; CHECK-NEXT:   %23 = insertvalue [3 x double] %21, double %22, 1
; CHECK-NEXT:   %24 = fadd fast double %9, %19
; CHECK-NEXT:   %25 = insertvalue [3 x double] %23, double %24, 2
; CHECK-NEXT:   ret [3 x double] %25
; CHECK-NEXT: }

; CHECK: define internal [3 x double] @fwddiffe3tester2(double %x, [3 x double] %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = extractvalue [3 x double] %"x'", 0
; CHECK-NEXT:   %1 = fmul fast double %0, %x
; CHECK-NEXT:   %2 = extractvalue [3 x double] %"x'", 1
; CHECK-NEXT:   %3 = fmul fast double %2, %x
; CHECK-NEXT:   %4 = extractvalue [3 x double] %"x'", 2
; CHECK-NEXT:   %5 = fmul fast double %4, %x
; CHECK-NEXT:   %6 = call fast double @hypot(double %x, double 2.000000e+00)
; CHECK-NEXT:   %7 = fdiv fast double %1, %6
; CHECK-NEXT:   %8 = insertvalue [3 x double] undef, double %7, 0
; CHECK-NEXT:   %9 = fdiv fast double %3, %6
; CHECK-NEXT:   %10 = insertvalue [3 x double] %8, double %9, 1
; CHECK-NEXT:   %11 = fdiv fast double %5, %6
; CHECK-NEXT:   %12 = insertvalue [3 x double] %10, double %11, 2
; CHECK-NEXT:   ret [3 x double] %12
; CHECK-NEXT: }


