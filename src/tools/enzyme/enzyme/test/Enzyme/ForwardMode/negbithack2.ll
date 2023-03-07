; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -instsimplify -S | FileCheck %s
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(instsimplify)" -enzyme-preopt=false -S | FileCheck %s

; Function Attrs: noinline nounwind readnone uwtable
define <2 x double> @tester(<2 x double> %x) {
entry:
  %cstx = bitcast <2 x double> %x to <2 x i64>
  %negx = xor <2 x i64> %cstx, <i64 -9223372036854775808, i64 -9223372036854775808>
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
; CHECK-NEXT:   %0 = {{(fsub fast <2 x double> <double \-?0.000000e\+00, double \-?0.000000e\+00>,|fneg fast <2 x double>)}} %"x'"
; CHECK-NEXT:   ret <2 x double> %0
; CHECK-NEXT: }
