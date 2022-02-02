; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -inline -mem2reg -instsimplify -gvn -dse -dse -S | FileCheck %s

define void @addOneMem(double* nocapture %x) {
entry:
  %0 = load double, double* %x, align 8
  %add = fadd double %0, 1.000000e+00
  store double %add, double* %x, align 8
  ret void
}

define void @test_derivative(double* %x, double* %xp1, double* %xp2, double* %xp3) {
entry:
  call void (void (double*)*, ...) @__enzyme_fwddiff(void (double*)* nonnull @addOneMem, metadata !"enzyme_width", i64 3, double* %x, double* %xp1, double* %xp2, double* %xp3)
  ret void
}

declare void @__enzyme_fwddiff(void (double*)*, ...)


; CHECK: define void @test_derivative(double* %x, double* %xp1, double* %xp2, double* %xp3)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = load double, double* %x, align 8
; CHECK-NEXT:   %1 = load double, double* %xp1, align 8
; CHECK-NEXT:   %2 = load double, double* %xp2, align 8
; CHECK-NEXT:   %3 = load double, double* %xp3, align 8
; CHECK-NEXT:   %add.i = fadd double %0, 1.000000e+00
; CHECK-NEXT:   store double %add.i, double* %x, align 8
; CHECK-NEXT:   store double %1, double* %xp1, align 8
; CHECK-NEXT:   store double %2, double* %xp2, align 8
; CHECK-NEXT:   store double %3, double* %xp3, align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }