; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -instsimplify -simplifycfg -adce -S | FileCheck %s

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x, double %y) {
entry:
  %call = call double @cabs(double %x, double %y)
  ret double %call
}

define [3 x double] @test_derivative(double %x, double %y) {
entry:
  %0 = tail call [3 x double] (double (double, double)*, ...) @__enzyme_fwddiff(double (double, double)* nonnull @tester, metadata !"enzyme_width", i64 3, double %x, double 1.0, double 1.3, double 2.0, double %y, double 1.0, double 0.0, double 2.0)
  ret [3 x double] %0
}

declare double @cabs(double, double)

; Function Attrs: nounwind
declare [3 x double] @__enzyme_fwddiff(double (double, double)*, ...)


; CHECK: define internal [3 x double] @fwddiffe3tester(double %x, [3 x double] %"x'", double %y, [3 x double] %"y'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call fast double @cabs(double %x, double %y)
; CHECK-NEXT:   %1 = extractvalue [3 x double] %"x'", 0
; CHECK-NEXT:   %2 = extractvalue [3 x double] %"y'", 0
; CHECK-NEXT:   %3 = fdiv fast double %1, %0
; CHECK-NEXT:   %4 = fmul fast double %x, %3
; CHECK-NEXT:   %5 = fdiv fast double %2, %0
; CHECK-NEXT:   %6 = fmul fast double %y, %5
; CHECK-NEXT:   %7 = fadd fast double %4, %6
; CHECK-NEXT:   %8 = insertvalue [3 x double] undef, double %7, 0
; CHECK-NEXT:   %9 = extractvalue [3 x double] %"x'", 1
; CHECK-NEXT:   %10 = extractvalue [3 x double] %"y'", 1
; CHECK-NEXT:   %11 = fdiv fast double %9, %0
; CHECK-NEXT:   %12 = fmul fast double %x, %11
; CHECK-NEXT:   %13 = fdiv fast double %10, %0
; CHECK-NEXT:   %14 = fmul fast double %y, %13
; CHECK-NEXT:   %15 = fadd fast double %12, %14
; CHECK-NEXT:   %16 = insertvalue [3 x double] %8, double %15, 1
; CHECK-NEXT:   %17 = extractvalue [3 x double] %"x'", 2
; CHECK-NEXT:   %18 = extractvalue [3 x double] %"y'", 2
; CHECK-NEXT:   %19 = fdiv fast double %17, %0
; CHECK-NEXT:   %20 = fmul fast double %x, %19
; CHECK-NEXT:   %21 = fdiv fast double %18, %0
; CHECK-NEXT:   %22 = fmul fast double %y, %21
; CHECK-NEXT:   %23 = fadd fast double %20, %22
; CHECK-NEXT:   %24 = insertvalue [3 x double] %16, double %23, 2
; CHECK-NEXT:   ret [3 x double] %24
; CHECK-NEXT: }