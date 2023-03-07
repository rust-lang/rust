; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -instsimplify -simplifycfg -adce -S | FileCheck %s

%struct.Gradients = type { double, double, double }

; Function Attrs: nounwind
declare %struct.Gradients @__enzyme_fwddiff(double (double)*, ...)

declare double @erf(double)

define double @tester(double %x) {
entry:
  %call = call double @erf(double %x)
  ret double %call
}

define %struct.Gradients @test_derivative(double %x) {
entry:
  %0 = tail call %struct.Gradients (double (double)*, ...) @__enzyme_fwddiff(double (double)* nonnull @tester, metadata !"enzyme_width", i64 3, double %x, double 1.0, double 2.0, double 3.0)
  ret %struct.Gradients %0
}


; CHECK: define internal [3 x double] @fwddiffe3tester(double %x, [3 x double] %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = fmul fast double %x, %x
; CHECK-NEXT:   %1 = {{(fsub fast double \-?0.000000e\+00,|fneg fast double)}} %0
; CHECK-NEXT:   %2 = call fast double @llvm.exp.f64(double %1)
; CHECK-NEXT:   %3 = fmul fast double %2, 0x3FF20DD750429B6D
; CHECK-NEXT:   %4 = extractvalue [3 x double] %"x'", 0
; CHECK-NEXT:   %5 = fmul fast double %3, %4
; CHECK-NEXT:   %6 = insertvalue [3 x double] undef, double %5, 0
; CHECK-NEXT:   %7 = extractvalue [3 x double] %"x'", 1
; CHECK-NEXT:   %8 = fmul fast double %3, %7
; CHECK-NEXT:   %9 = insertvalue [3 x double] %6, double %8, 1
; CHECK-NEXT:   %10 = extractvalue [3 x double] %"x'", 2
; CHECK-NEXT:   %11 = fmul fast double %3, %10
; CHECK-NEXT:   %12 = insertvalue [3 x double] %9, double %11, 2
; CHECK-NEXT:   ret [3 x double] %12
; CHECK-NEXT: }