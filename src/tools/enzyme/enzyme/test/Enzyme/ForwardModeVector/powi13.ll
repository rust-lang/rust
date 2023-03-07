; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

%struct.Gradients = type { double, double, double }

; Function Attrs: nounwind
declare %struct.Gradients @__enzyme_fwddiff(double (double, i32)*, ...)

; Function Attrs: noinline nounwind readnone uwtable
define double @tester(double %x, i32 %y) {
entry:
  %0 = tail call fast double @llvm.powi.f64.i32(double %x, i32 %y)
  ret double %0
}

define %struct.Gradients @test_derivative(double %x, i32 %y) {
entry:
  %0 = tail call %struct.Gradients (double (double, i32)*, ...) @__enzyme_fwddiff(double (double, i32)* nonnull @tester, metadata !"enzyme_width", i64 3, double %x, double 1.0, double 2.0, double 3.0, i32 %y)
  ret %struct.Gradients %0
}

; Function Attrs: nounwind readnone speculatable
declare double @llvm.powi.f64.i32(double, i32)


; CHECK: define internal [3 x double] @fwddiffe3tester(double %x, [3 x double] %"x'", i32 %y)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = sub i32 %y, 1
; CHECK-NEXT:   %1 = call fast double @llvm.powi.f64{{(\.i32)?}}(double %x, i32 %0)
; CHECK-NEXT:   %2 = sitofp i32 %y to double
; CHECK-NEXT:   %3 = icmp eq i32 0, %y
; CHECK-NEXT:   %4 = extractvalue [3 x double] %"x'", 0
; CHECK-NEXT:   %5 = fmul fast double %4, %1
; CHECK-NEXT:   %6 = fmul fast double %5, %2
; CHECK-NEXT:   %7 = select {{(fast )?}}i1 %3, double 0.000000e+00, double %6
; CHECK-NEXT:   %8 = insertvalue [3 x double] undef, double %7, 0
; CHECK-NEXT:   %9 = extractvalue [3 x double] %"x'", 1
; CHECK-NEXT:   %10 = fmul fast double %9, %1
; CHECK-NEXT:   %11 = fmul fast double %10, %2
; CHECK-NEXT:   %12 = select {{(fast )?}}i1 %3, double 0.000000e+00, double %11
; CHECK-NEXT:   %13 = insertvalue [3 x double] %8, double %12, 1
; CHECK-NEXT:   %14 = extractvalue [3 x double] %"x'", 2
; CHECK-NEXT:   %15 = fmul fast double %14, %1
; CHECK-NEXT:   %16 = fmul fast double %15, %2
; CHECK-NEXT:   %17 = select {{(fast )?}}i1 %3, double 0.000000e+00, double %16
; CHECK-NEXT:   %18 = insertvalue [3 x double] %13, double %17, 2
; CHECK-NEXT:   ret [3 x double] %18
; CHECK-NEXT }
