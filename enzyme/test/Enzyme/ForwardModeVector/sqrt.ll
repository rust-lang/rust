; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -O3 -S | FileCheck %s

%struct.Gradients = type { double, double, double }

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x) {
entry:
  %0 = tail call fast double @llvm.sqrt.f64(double %x)
  ret double %0
}

define %struct.Gradients @test_derivative(double %x) {
entry:
  %0 = tail call %struct.Gradients (double (double)*, ...) @__enzyme_fwddiff(double (double)* nonnull @tester, metadata !"enzyme_width", i64 3, double %x, double 1.0, double 2.0, double 3.0)
  ret %struct.Gradients %0
}

; Function Attrs: nounwind readnone speculatable
declare double @llvm.sqrt.f64(double)

; Function Attrs: nounwind
declare %struct.Gradients @__enzyme_fwddiff(double (double)*, ...)


; CHECK: define %struct.Gradients @test_derivative(double %x)
; CHECK-NEXT: entry
; CHECK-NEXT:   %0 = tail call fast double @llvm.sqrt.f64(double %x)
; CHECK-NEXT:   %1 = fdiv fast double 5.000000e-01, %0
; CHECK-NEXT:   %2 = fcmp fast oeq double %x, 0.000000e+00
; CHECK-NEXT:   %3 = select {{(fast )?}}i1 %2, double 0.000000e+00, double %1
; CHECK-NEXT:   %4 = fdiv fast double 1.000000e+00, %0
; CHECK-NEXT:   %5 = select {{(fast )?}}i1 %2, double 0.000000e+00, double %4
; CHECK-NEXT:   %6 = fdiv fast double 1.500000e+00, %0
; CHECK-NEXT:   %7 = select {{(fast )?}}i1 %2, double 0.000000e+00, double %6
; CHECK-NEXT:   %8 = insertvalue %struct.Gradients zeroinitializer, double %3, 0
; CHECK-NEXT:   %9 = insertvalue %struct.Gradients %8, double %5, 1
; CHECK-NEXT:   %10 = insertvalue %struct.Gradients %9, double %7, 2
; CHECK-NEXT:   ret %struct.Gradients %10
; CHECK-NEXT: }