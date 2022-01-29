; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

%struct.Gradients = type { double, double, double, double, double, double }

; Function Attrs: noinline nounwind readnone uwtable
define <2 x double> @tester(<2 x double> %x) {
entry:
  %cstx = bitcast <2 x double> %x to <2 x i64>
  %negx = xor <2 x i64> %cstx, <i64 -9223372036854775808, i64 0>
  %csty = bitcast <2 x i64> %negx to <2 x double>
  ret <2 x double> %csty
}

define %struct.Gradients @test_derivative(<2 x double> %x, <2 x double> %dx1, <2 x double> %dx2, <2 x double> %dx3) {
entry:
  %0 = tail call %struct.Gradients (<2 x double> (<2 x double>)*, ...) @__enzyme_fwddiff(<2 x double> (<2 x double>)* nonnull @tester, metadata !"enzyme_width", i64 3, <2 x double> %x, <2 x double> %dx1, <2 x double> %dx2, <2 x double> %dx3)
  ret %struct.Gradients %0
}

; Function Attrs: nounwind
declare %struct.Gradients @__enzyme_fwddiff(<2 x double> (<2 x double>)*, ...)


; CHECK: define internal [3 x <2 x double>] @fwddiffe3tester(<2 x double> %x, [3 x <2 x double>] %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = extractvalue [3 x <2 x double>] %"x'", 0
; CHECK-NEXT:   %1 = bitcast <2 x double> %0 to <2 x i64>
; CHECK-NEXT:   %2 = extractvalue [3 x <2 x double>] %"x'", 1
; CHECK-NEXT:   %3 = bitcast <2 x double> %2 to <2 x i64>
; CHECK-NEXT:   %4 = extractvalue [3 x <2 x double>] %"x'", 2
; CHECK-NEXT:   %5 = bitcast <2 x double> %4 to <2 x i64>
; CHECK-NEXT:   %6 = extractelement <2 x i64> %1, i64 0
; CHECK-NEXT:   %7 = bitcast i64 %6 to double
; CHECK-NEXT:   %8 = {{(fsub fast double \-0.000000e\+00,|fneg fast double)}} %7
; CHECK-NEXT:   %9 = bitcast double %8 to i64
; CHECK-NEXT:   %10 = insertelement <2 x i64> undef, i64 %9, i64 0
; CHECK-NEXT:   %11 = extractelement <2 x i64> %1, i64 1
; CHECK-NEXT:   %12 = insertelement <2 x i64> %10, i64 %11, i64 1
; CHECK-NEXT:   %13 = extractelement <2 x i64> %3, i64 0
; CHECK-NEXT:   %14 = bitcast i64 %13 to double
; CHECK-NEXT:   %15 = {{(fsub fast double \-0.000000e\+00,|fneg fast double)}} %14
; CHECK-NEXT:   %16 = bitcast double %15 to i64
; CHECK-NEXT:   %17 = insertelement <2 x i64> undef, i64 %16, i64 0
; CHECK-NEXT:   %18 = extractelement <2 x i64> %3, i64 1
; CHECK-NEXT:   %19 = insertelement <2 x i64> %17, i64 %18, i64 1
; CHECK-NEXT:   %20 = extractelement <2 x i64> %5, i64 0
; CHECK-NEXT:   %21 = bitcast i64 %20 to double
; CHECK-NEXT:   %22 = {{(fsub fast double \-0.000000e\+00,|fneg fast double)}} %21
; CHECK-NEXT:   %23 = bitcast double %22 to i64
; CHECK-NEXT:   %24 = insertelement <2 x i64> undef, i64 %23, i64 0
; CHECK-NEXT:   %25 = extractelement <2 x i64> %5, i64 1
; CHECK-NEXT:   %26 = insertelement <2 x i64> %24, i64 %25, i64 1
; CHECK-NEXT:   %27 = bitcast <2 x i64> %12 to <2 x double>
; CHECK-NEXT:   %28 = insertvalue [3 x <2 x double>] undef, <2 x double> %27, 0
; CHECK-NEXT:   %29 = bitcast <2 x i64> %19 to <2 x double>
; CHECK-NEXT:   %30 = insertvalue [3 x <2 x double>] %28, <2 x double> %29, 1
; CHECK-NEXT:   %31 = bitcast <2 x i64> %26 to <2 x double>
; CHECK-NEXT:   %32 = insertvalue [3 x <2 x double>] %30, <2 x double> %31, 2
; CHECK-NEXT:   ret [3 x <2 x double>] %32
; CHECK-NEXT: }