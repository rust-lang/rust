; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

%struct.Gradients = type { double, double, double }

; Function Attrs: noinline nounwind readnone uwtable
define double @tester(double %x) {
entry:
  %cstx = bitcast double %x to i64 
  %negx = xor i64 %cstx, -9223372036854775808
  %csty = bitcast i64 %negx to double
  ret double %csty
}

define %struct.Gradients @test_derivative(double %x, double %dx1, double %dx2, double %dx3) {
entry:
  %0 = tail call %struct.Gradients (double (double)*, ...) @__enzyme_fwddiff(double (double)* nonnull @tester, metadata !"enzyme_width", i64 3, double %x, double %dx1, double %dx2, double %dx3)
  ret %struct.Gradients %0
}

; Function Attrs: nounwind
declare %struct.Gradients @__enzyme_fwddiff(double (double)*, ...)

; CHECK: define internal [3 x double] @fwddiffe3tester(double %x, [3 x double] %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = extractvalue [3 x double] %"x'", 0
; CHECK-NEXT:   %1 = extractvalue [3 x double] %"x'", 1
; CHECK-NEXT:   %2 = extractvalue [3 x double] %"x'", 2
; CHECK-NEXT:   %3 = {{(fsub fast double \-0.000000e\+00,|fneg fast double)}} %0
; CHECK-NEXT:   %4 = {{(fsub fast double \-0.000000e\+00,|fneg fast double)}} %1
; CHECK-NEXT:   %5 = {{(fsub fast double \-0.000000e\+00,|fneg fast double)}} %2
; CHECK-NEXT:   %6 = insertvalue [3 x double] undef, double %3, 0
; CHECK-NEXT:   %7 = insertvalue [3 x double] %6, double %4, 1
; CHECK-NEXT:   %8 = insertvalue [3 x double] %7, double %5, 2
; CHECK-NEXT:   ret [3 x double] %8
; CHECK-NEXT: }