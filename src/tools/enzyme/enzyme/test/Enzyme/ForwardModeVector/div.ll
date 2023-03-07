; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -early-cse -instsimplify -simplifycfg -S | FileCheck %s

%struct.Gradients = type { double, double }

; Function Attrs: nounwind
declare %struct.Gradients @__enzyme_fwddiff(double (double,double)*, ...)

; Function Attrs: noinline nounwind readnone uwtable
define double @tester(double %x, double %y) {
entry:
  %0 = fdiv fast double %x, %y
  ret double %0
}

define %struct.Gradients @test_derivative(double %x, double %y) {
entry:
  %0 = tail call %struct.Gradients (double (double, double)*, ...) @__enzyme_fwddiff(double (double, double)* nonnull @tester, metadata !"enzyme_width", i64 2, double %x, double 0.0, double 1.0, double %y, double 1.0, double 0.0)
  ret %struct.Gradients %0
}


; CHECK: define internal [2 x double] @fwddiffe2tester(double %x, [2 x double] %"x'", double %y, [2 x double] %"y'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = extractvalue [2 x double] %"x'", 0
; CHECK-NEXT:   %1 = extractvalue [2 x double] %"y'", 0
; CHECK-NEXT:   %2 = fmul fast double %0, %y
; CHECK-NEXT:   %3 = fmul fast double %x, %1
; CHECK-NEXT:   %4 = fsub fast double %2, %3
; CHECK-NEXT:   %5 = extractvalue [2 x double] %"x'", 1
; CHECK-NEXT:   %6 = extractvalue [2 x double] %"y'", 1
; CHECK-NEXT:   %7 = fmul fast double %5, %y
; CHECK-NEXT:   %8 = fmul fast double %x, %6
; CHECK-NEXT:   %9 = fsub fast double %7, %8
; CHECK-NEXT:   %10 = fmul fast double %y, %y
; CHECK-NEXT:   %11 = fdiv fast double %4, %10
; CHECK-NEXT:   %12 = insertvalue [2 x double] undef, double %11, 0
; CHECK-NEXT:   %13 = fdiv fast double %9, %10
; CHECK-NEXT:   %14 = insertvalue [2 x double] %12, double %13, 1
; CHECK-NEXT:   ret [2 x double] %14
; CHECK-NEXT: }