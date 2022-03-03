; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -instsimplify -simplifycfg -adce -S | FileCheck %s

%struct.Gradients = type { double, double }

; Function Attrs: nounwind
declare %struct.Gradients @__enzyme_fwddiff(double (double)*, ...)

declare double @erfi(double)

define double @tester(double %x) {
entry:
  %call = call double @erfi(double %x)
  ret double %call
}

define %struct.Gradients @test_derivative(double %x) {
entry:
  %0 = tail call %struct.Gradients (double (double)*, ...) @__enzyme_fwddiff(double (double)* nonnull @tester, metadata !"enzyme_width", i64 2, double %x, double 1.0, double 2.0)
  ret %struct.Gradients %0
}


; CHECK: define internal [2 x double] @fwddiffe2tester(double %x, [2 x double] %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = fmul fast double %x, %x
; CHECK-NEXT:   %1 = call fast double @llvm.exp.f64(double %0)
; CHECK-NEXT:   %2 = fmul fast double %1, 0x3FF20DD750429B6D
; CHECK-NEXT:   %3 = extractvalue [2 x double] %"x'", 0
; CHECK-NEXT:   %4 = fmul fast double %2, %3
; CHECK-NEXT:   %5 = insertvalue [2 x double] undef, double %4, 0
; CHECK-NEXT:   %6 = extractvalue [2 x double] %"x'", 1
; CHECK-NEXT:   %7 = fmul fast double %2, %6
; CHECK-NEXT:   %8 = insertvalue [2 x double] %5, double %7, 1
; CHECK-NEXT:   ret [2 x double] %8
; CHECK-NEXT: }