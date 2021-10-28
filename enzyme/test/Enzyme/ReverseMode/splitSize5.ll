; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

; Function Attrs: noinline nounwind readnone uwtable
define double @tester(double* %x) {
entry:
  %gep = getelementptr double, double* %x, i32 1
  %y = load double, double* %x
  %z = load double, double* %gep
  %res = fadd fast double %y, %z
  ret double %res
}

define void @test_derivative(double* %x, double* %dx) {
entry:
  %size = call i64 (double (double*)*, ...) @__enzyme_augmentsize(double (double*)* nonnull @tester, metadata !"enzyme_dup")
  %cache = alloca i8, i64 %size, align 1
  call void (double (double*)*, ...) @__enzyme_augmentfwd(double (double*)* nonnull @tester, metadata !"enzyme_allocated", i64 %size, metadata !"enzyme_tape", i8* %cache, double* %x, double* %dx)
  tail call void (double (double*)*, ...) @__enzyme_reverse(double (double*)* nonnull @tester, metadata !"enzyme_allocated", i64 %size, metadata !"enzyme_tape", i8* %cache, double* %x, double* %dx)
  ret void
}

; Function Attrs: nounwind
declare void @__enzyme_augmentfwd(double (double*)*, ...)
declare i64 @__enzyme_augmentsize(double (double*)*, ...)
declare void @__enzyme_reverse(double (double*)*, ...)

; CHECK: define void @test_derivative(double* %x, double* %dx)
; CHECK-NEXT: entry:
; CHECK-NEXT:  %0 = call fast double @augmented_tester(double* %x, double* %dx)
; CHECK-NEXT:  call void @diffetester(double* %x, double* %dx, double 1.000000e+00)
; CHECK-NEXT:  ret void
; CHECK-NEXT:}
