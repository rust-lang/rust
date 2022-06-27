; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -S | FileCheck %s

define void @addOneMem(double* nocapture %x) {
entry:
  %0 = load double, double* %x, align 8
  %add = fadd fast double %0, 1.000000e+00
  store double %add, double* %x, align 8
  ret void
}

define void @test(double* %x1, double* %x2, double* %x3, double* %x4) {
entry:
  tail call void (...) @__enzyme_batch(void (double*)* nonnull @addOneMem, metadata !"enzyme_width", i64 4, metadata !"enzyme_vector", double* %x1, double* %x2, double* %x3, double* %x4)
  ret void
}

declare void @__enzyme_batch(...)


; CHECK: define internal void @batch_addOneMem([4 x double*] %x)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %unwrap.x0 = extractvalue [4 x double*] %x, 0
; CHECK-NEXT:   %unwrap.x1 = extractvalue [4 x double*] %x, 1
; CHECK-NEXT:   %unwrap.x2 = extractvalue [4 x double*] %x, 2
; CHECK-NEXT:   %unwrap.x3 = extractvalue [4 x double*] %x, 3
; CHECK-NEXT:   %0 = load double, double* %unwrap.x0, align 8
; CHECK-NEXT:   %1 = load double, double* %unwrap.x1, align 8
; CHECK-NEXT:   %2 = load double, double* %unwrap.x2, align 8
; CHECK-NEXT:   %3 = load double, double* %unwrap.x3, align 8
; CHECK-NEXT:   %add0 = fadd fast double %0, 1.000000e+00
; CHECK-NEXT:   %add1 = fadd fast double %1, 1.000000e+00
; CHECK-NEXT:   %add2 = fadd fast double %2, 1.000000e+00
; CHECK-NEXT:   %add3 = fadd fast double %3, 1.000000e+00
; CHECK-NEXT:   store double %add0, double* %unwrap.x0, align 8
; CHECK-NEXT:   store double %add3, double* %unwrap.x3, align 8
; CHECK-NEXT:   store double %add2, double* %unwrap.x2, align 8
; CHECK-NEXT:   store double %add1, double* %unwrap.x1, align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }