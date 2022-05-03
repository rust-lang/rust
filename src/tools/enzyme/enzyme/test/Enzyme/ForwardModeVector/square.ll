; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -simplifycfg -early-cse -S | FileCheck %s

%struct.Gradients = type { double, double, double }

; Function Attrs: nounwind
declare %struct.Gradients @__enzyme_fwddiff(double (double)*, ...)

define double @square(double %x) {
entry:
  %mul = fmul fast double %x, %x
  ret double %mul
}

define %struct.Gradients @dsquare(double %x) {
entry:
  %0 = tail call %struct.Gradients (double (double)*, ...) @__enzyme_fwddiff(double (double)* nonnull @square, metadata !"enzyme_width", i64 3, double %x, double 1.0, double 10.0, double 100.0)
  ret %struct.Gradients %0
}


; CHECK: define internal [3 x double] @fwddiffe3square(double %x, [3 x double] %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = extractvalue [3 x double] %"x'", 0
; CHECK-NEXT:   %1 = fmul fast double %0, %x
; CHECK-NEXT:   %2 = fadd fast double %1, %1
; CHECK-NEXT:   %3 = insertvalue [3 x double] undef, double %2, 0
; CHECK-NEXT:   %4 = extractvalue [3 x double] %"x'", 1
; CHECK-NEXT:   %5 = fmul fast double %4, %x
; CHECK-NEXT:   %6 = fadd fast double %5, %5
; CHECK-NEXT:   %7 = insertvalue [3 x double] %3, double %6, 1
; CHECK-NEXT:   %8 = extractvalue [3 x double] %"x'", 2
; CHECK-NEXT:   %9 = fmul fast double %8, %x
; CHECK-NEXT:   %10 = fadd fast double %9, %9
; CHECK-NEXT:   %11 = insertvalue [3 x double] %7, double %10, 2
; CHECK-NEXT:   ret [3 x double] %11
; CHECK-NEXT: }