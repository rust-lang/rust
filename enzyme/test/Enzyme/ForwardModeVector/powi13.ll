; RUN: if [ %llvmver -ge 13 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi

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


; CHECK: define internal [3 x double] @fwddiffe3tester(double %x, [3 x double] %"x'", i32 %y) #1 {
; CHECK-NEXT entry:
; CHECK-NEXT   %0 = sub i32 %y, 1
; CHECK-NEXT   %1 = call fast double @llvm.powi.f64.i32(double %x, i32 %0)
; CHECK-NEXT   %2 = sitofp i32 %y to double
; CHECK-NEXT   %3 = extractvalue [3 x double] %"x'", 0
; CHECK-NEXT   %4 = fmul fast double %3, %1
; CHECK-NEXT   %5 = fmul fast double %4, %2
; CHECK-NEXT   %6 = insertvalue [3 x double] undef, double %5, 0
; CHECK-NEXT   %7 = extractvalue [3 x double] %"x'", 1
; CHECK-NEXT   %8 = fmul fast double %7, %1
; CHECK-NEXT   %9 = fmul fast double %8, %2
; CHECK-NEXT   %10 = insertvalue [3 x double] %6, double %9, 1
; CHECK-NEXT   %11 = extractvalue [3 x double] %"x'", 2
; CHECK-NEXT   %12 = fmul fast double %11, %1
; CHECK-NEXT   %13 = fmul fast double %12, %2
; CHECK-NEXT   %14 = insertvalue [3 x double] %10, double %13, 2
; CHECK-NEXT   ret [3 x double] %14
; CHECK-NEXT }