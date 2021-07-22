; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -O3 -S | FileCheck %s

declare i32 @putchar(i32) 

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x) {
entry:
  %tmp = call i32 @putchar(i32 32)
  %0 = tail call fast double @llvm.exp.f64(double %x)
  ret double %0
}

define double @test_derivative(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_autodiff(double (double)* nonnull @tester, double %x)
  ret double %0
}

; Function Attrs: nounwind readnone speculatable
declare double @llvm.exp.f64(double)

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double)*, ...)

; CHECK: define double @test_derivative(double %x)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %tmp.i = tail call i32 @putchar(i32 32)
; CHECK-NEXT:   %0 = tail call fast double @llvm.exp.f64(double %x)
; CHECK-NEXT:   ret double %0
; CHECK-NEXT: }
