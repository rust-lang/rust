; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -instsimplify -simplifycfg -adce -S | FileCheck %s

; Function Attrs: nounwind readnone willreturn
declare double @cabs([2 x double]) #7

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x, double %y) {
entry:
  %agg0 = insertvalue [2 x double] undef, double %x, 0
  %agg1 = insertvalue [2 x double] %agg0, double %y, 1
  %call = call double @cabs([2 x double] %agg1)
  ret double %call
}

define [3 x double] @test_derivative(double %x, double %y) {
entry:
  %0 = tail call [3 x double] (double (double, double)*, ...) @__enzyme_fwddiff(double (double, double)* nonnull @tester, metadata !"enzyme_width", i64 3, double %x, double 1.0, double 1.3, double 2.0, double %y, double 1.0, double 0.0, double 2.0)
  ret [3 x double] %0
}

; Function Attrs: nounwind
declare [3 x double] @__enzyme_fwddiff(double (double, double)*, ...)


; CHECK: define internal [3 x double] @fwddiffe3tester(double %x, [3 x double] %"x'", double %y, [3 x double] %"y'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = extractvalue [3 x double] %"x'", 0
; CHECK-NEXT:   %1 = extractvalue [3 x double] %"x'", 1
; CHECK-NEXT:   %2 = extractvalue [3 x double] %"x'", 2
; CHECK-NEXT:   %agg0 = insertvalue [2 x double] undef, double %x, 0
; CHECK-NEXT:   %3 = extractvalue [3 x double] %"y'", 0
; CHECK-NEXT:   %4 = extractvalue [3 x double] %"y'", 1
; CHECK-NEXT:   %5 = extractvalue [3 x double] %"y'", 2
; CHECK-NEXT:   %agg1 = insertvalue [2 x double] %agg0, double %y, 1
; CHECK-NEXT:   %6 = call fast double @cabs([2 x double] %agg1)
; CHECK-NEXT:   %7 = fdiv fast double %0, %6
; CHECK-NEXT:   %8 = fmul fast double %x, %7
; CHECK-NEXT:   %9 = fdiv fast double %3, %6
; CHECK-NEXT:   %10 = fmul fast double %y, %9
; CHECK-NEXT:   %11 = fadd fast double %8, %10
; CHECK-NEXT:   %12 = insertvalue [3 x double] undef, double %11, 0
; CHECK-NEXT:   %13 = fdiv fast double %1, %6
; CHECK-NEXT:   %14 = fmul fast double %x, %13
; CHECK-NEXT:   %15 = fdiv fast double %4, %6
; CHECK-NEXT:   %16 = fmul fast double %y, %15
; CHECK-NEXT:   %17 = fadd fast double %14, %16
; CHECK-NEXT:   %18 = insertvalue [3 x double] %12, double %17, 1
; CHECK-NEXT:   %19 = fdiv fast double %2, %6
; CHECK-NEXT:   %20 = fmul fast double %x, %19
; CHECK-NEXT:   %21 = fdiv fast double %5, %6
; CHECK-NEXT:   %22 = fmul fast double %y, %21
; CHECK-NEXT:   %23 = fadd fast double %20, %22
; CHECK-NEXT:   %24 = insertvalue [3 x double] %18, double %23, 2
; CHECK-NEXT:   ret [3 x double] %24
; CHECK-NEXT: }