; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -O3 -S | FileCheck %s

; Function Attrs: nounwind
declare [4 x double] @__enzyme_batch(...)

; Function Attrs: nounwind readnone speculatable
declare double @llvm.log.f64(double)

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x) {
entry:
  %0 = tail call fast double @llvm.log.f64(double %x)
  ret double %0
}

define [4 x double] @test_derivative(double %x1, double %x2, double %x3, double %x4) {
entry:
  %0 = tail call [4 x double] (...) @__enzyme_batch(double (double)* nonnull @tester, metadata !"enzyme_width", i64 4, metadata !"enzyme_vector", double %x1, double %x2, double %x3, double %x4)
  ret [4 x double] %0
}


; CHECK: define [4 x double] @test_derivative(double %x1, double %x2, double %x3, double %x4)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = tail call fast double @llvm.log.f64(double %x1)
; CHECK-NEXT:   %1 = tail call fast double @llvm.log.f64(double %x2)
; CHECK-NEXT:   %2 = tail call fast double @llvm.log.f64(double %x3)
; CHECK-NEXT:   %3 = tail call fast double @llvm.log.f64(double %x4)
; CHECK-NEXT:   %mrv.i = insertvalue [4 x double] undef, double %0, 0
; CHECK-NEXT:   %mrv1.i = insertvalue [4 x double] %mrv.i, double %1, 1
; CHECK-NEXT:   %mrv2.i = insertvalue [4 x double] %mrv1.i, double %2, 2
; CHECK-NEXT:   %mrv3.i = insertvalue [4 x double] %mrv2.i, double %3, 3
; CHECK-NEXT:   ret [4 x double] %mrv3.i
; CHECK-NEXT: }