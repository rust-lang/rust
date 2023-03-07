; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -instsimplify -S | FileCheck %s
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(instsimplify)" -enzyme-preopt=false -S | FileCheck %s

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
; CHECK-NEXT:   %[[i0:.+]] = bitcast <2 x double> %"x'" to <2 x i64>
; CHECK-NEXT:   %[[i1:.+]] = extractelement <2 x i64> %[[i0]], i64 0
; CHECK-NEXT:   %[[i2:.+]] = bitcast i64 %[[i1]] to double
; CHECK-NEXT:   %[[i3:.+]] = {{(fsub fast double -?0.000000e\+00,|fneg fast double)}} %[[i2]]
; CHECK-NEXT:   %[[i4:.+]] = bitcast double %[[i3]] to i64
; CHECK-NEXT:   %[[i5:.+]] = insertelement <2 x i64> undef, i64 %[[i4]], i64 0
; CHECK-NEXT:   %[[i6:.+]] = extractelement <2 x i64> %[[i0]], i64 1
; CHECK-NEXT:   %[[i7:.+]] = insertelement <2 x i64> %[[i5]], i64 %[[i6]], i64 1
; CHECK-NEXT:   %[[i8:.+]] = bitcast <2 x i64> %[[i7]] to <2 x double>
; CHECK-NEXT:   ret <2 x double> %[[i8]]
; CHECK-NEXT: }
