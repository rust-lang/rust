; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

; Function Attrs: noinline nounwind readnone uwtable
define <2 x double> @tester(<2 x double> %x) {
entry:
  %cstx = bitcast <2 x double> %x to <2 x i64>
  %negx = xor <2 x i64> %cstx, <i64 -9223372036854775808, i64 0>
  %csty = bitcast <2 x i64> %negx to <2 x double>
  ret <2 x double> %csty
}

define <2 x double> @test_derivative(<2 x double> %x, <2 x double> %dx) {
entry:
  %0 = tail call <2 x double> (<2 x double> (<2 x double>)*, ...) @__enzyme_fwddiff(<2 x double> (<2 x double>)* nonnull @tester, <2 x double> %x, <2 x double> %dx)
  ret <2 x double> %0
}

; Function Attrs: nounwind
declare <2 x double> @__enzyme_fwddiff(<2 x double> (<2 x double>)*, ...)

; CHECK: define internal <2 x double> @fwddiffetester(<2 x double> %x, <2 x double> %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = bitcast <2 x double> %"x'" to <2 x i64>
; CHECK-NEXT:   %1 = extractelement <2 x i64> %0, i64 0
; CHECK-NEXT:   %2 = bitcast i64 %1 to double
; CHECK-NEXT:   %3 = {{(fsub fast double -?0.000000e\+00,|fneg fast double)}} %2
; CHECK-NEXT:   %4 = bitcast double %3 to i64
; CHECK-NEXT:   %5 = insertelement <2 x i64> undef, i64 %4, i64 0
; CHECK-NEXT:   %6 = extractelement <2 x i64> %0, i64 1
; CHECK-NEXT:   %7 = insertelement <2 x i64> %5, i64 %6, i64 1
; CHECK-NEXT:   %8 = bitcast <2 x i64> %7 to <2 x double>
; CHECK-NEXT:   ret <2 x double> %8
; CHECK-NEXT: }
