; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

%struct.Gradients = type { double, double, double, double, double, double }

; Function Attrs: noinline nounwind readnone uwtable
define <2 x double> @tester(<2 x double> %x) {
entry:
  %cstx = bitcast <2 x double> %x to <2 x i64>
  %negx = xor <2 x i64> %cstx, <i64 -9223372036854775808, i64 -9223372036854775808>
  %csty = bitcast <2 x i64> %negx to <2 x double>
  ret <2 x double> %csty
}

define %struct.Gradients @test_derivative(<2 x double> %x, <2  x double> %dx1, <2 x double> %dx2, <2 x double> %dx3) {
entry:
  %0 = tail call %struct.Gradients (<2 x double> (<2 x double>)*, ...) @__enzyme_fwddiff(<2 x double> (<2 x double>)* nonnull @tester, metadata !"enzyme_width", i64 3, <2 x double> %x, <2 x double> %dx1, <2 x double> %dx2, <2 x double> %dx3)
  ret %struct.Gradients %0
}

; Function Attrs: nounwind
declare %struct.Gradients @__enzyme_fwddiff(<2 x double> (<2 x double>)*, ...)


; CHECK: define internal [3 x <2 x double>] @fwddiffe3tester(<2 x double> %x, [3 x <2 x double>] %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = extractvalue [3 x <2 x double>] %"x'", 0
; CHECK-NEXT:   %1 = extractvalue [3 x <2 x double>] %"x'", 1
; CHECK-NEXT:   %2 = extractvalue [3 x <2 x double>] %"x'", 2
; CHECK-NEXT:   %3 = {{(fsub fast <2 x double> <double \-0.000000e\+00, double \-0.000000e\+00>,|fneg fast <2 x double>)}} %0
; CHECK-NEXT:   %4 = {{(fsub fast <2 x double> <double \-0.000000e\+00, double \-0.000000e\+00>,|fneg fast <2 x double>)}} %1
; CHECK-NEXT:   %5 = {{(fsub fast <2 x double> <double \-0.000000e\+00, double \-0.000000e\+00>,|fneg fast <2 x double>)}} %2
; CHECK-NEXT:   %6 = insertvalue [3 x <2 x double>] undef, <2 x double> %3, 0
; CHECK-NEXT:   %7 = insertvalue [3 x <2 x double>] %6, <2 x double> %4, 1
; CHECK-NEXT:   %8 = insertvalue [3 x <2 x double>] %7, <2 x double> %5, 2
; CHECK-NEXT:   ret [3 x <2 x double>] %8
; CHECK-NEXT: }