; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -O3 -S | FileCheck %s

%struct.Gradients = type { double, double }

; Function Attrs: nounwind
declare %struct.Gradients @__enzyme_fwddiff(double (double)*, ...)

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x) {
entry:
  %0 = tail call fast double @llvm.exp2.f64(double %x)
  ret double %0
}

define %struct.Gradients @test_derivative(double %x) {
entry:
  %0 = tail call %struct.Gradients (double (double)*, ...) @__enzyme_fwddiff(double (double)* nonnull @tester, metadata !"enzyme_width", i64 2, double %x, double 1.0, double 2.5)
  ret %struct.Gradients %0
}

; Function Attrs: nounwind readnone speculatable
declare double @llvm.exp2.f64(double)


; CHECK: define %struct.Gradients @test_derivative(double %x)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = tail call fast double @llvm.exp2.f64(double %x)
; CHECK-NEXT:   %1 = fmul fast double %0, 0x3FE62E42FEFA39EF
; CHECK-NEXT:   %2 = fmul fast double %0, 0x3FFBB9D3BEB8C86B
; CHECK-NEXT:   %3 = insertvalue %struct.Gradients zeroinitializer, double %1, 0
; CHECK-NEXT:   %4 = insertvalue %struct.Gradients %3, double %2, 1
; CHECK-NEXT:   ret %struct.Gradients %4
; CHECK-NEXT: }