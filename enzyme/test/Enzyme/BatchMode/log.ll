; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify)" -enzyme-preopt=false -S | FileCheck %s

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

; CHECK: define internal [4 x double] @batch_tester([4 x double] %x) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %unwrap.x0 = extractvalue [4 x double] %x, 0
; CHECK-NEXT:   %unwrap.x1 = extractvalue [4 x double] %x, 1
; CHECK-NEXT:   %unwrap.x2 = extractvalue [4 x double] %x, 2
; CHECK-NEXT:   %unwrap.x3 = extractvalue [4 x double] %x, 3
; CHECK-NEXT:   %0 = tail call fast double @llvm.log.f64(double %unwrap.x0)
; CHECK-NEXT:   %1 = tail call fast double @llvm.log.f64(double %unwrap.x1)
; CHECK-NEXT:   %2 = tail call fast double @llvm.log.f64(double %unwrap.x2)
; CHECK-NEXT:   %3 = tail call fast double @llvm.log.f64(double %unwrap.x3)
; CHECK-NEXT:   %mrv = insertvalue [4 x double] {{(undef|poison)}}, double %0, 0
; CHECK-NEXT:   %mrv1 = insertvalue [4 x double] %mrv, double %1, 1
; CHECK-NEXT:   %mrv2 = insertvalue [4 x double] %mrv1, double %2, 2
; CHECK-NEXT:   %mrv3 = insertvalue [4 x double] %mrv2, double %3, 3
; CHECK-NEXT:   ret [4 x double] %mrv3
; CHECK-NEXT: }
