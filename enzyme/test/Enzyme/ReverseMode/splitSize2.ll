; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

; Function Attrs: noinline nounwind readnone uwtable
define void @tester(double* %x) {
entry:
  %gep = getelementptr double, double* %x, i32 1
  %y = load double, double* %x
  %z = load double, double* %gep
  %res = fmul fast double %y, %z
  store double %res, double* %x
  ret void
}

define void @test_derivative(double* %x, double* %dx) {
entry:
  %size = call i64 (void (double*)*, ...) @__enzyme_augmentsize(void (double*)* nonnull @tester, metadata !"enzyme_dup")
  %cache = alloca i8, i64 %size, align 1
  tail call void (void (double*)*, ...) @__enzyme_augmentfwd(void (double*)* nonnull @tester, metadata !"enzyme_allocated", i64 %size, metadata !"enzyme_tape", i8* %cache, double* %x, double* %dx)
  tail call void (void (double*)*, ...) @__enzyme_reverse(void (double*)* nonnull @tester, metadata !"enzyme_allocated", i64 %size, metadata !"enzyme_tape", i8* %cache, double* %x, double* %dx)
  ret void
}

; Function Attrs: nounwind
declare void @__enzyme_augmentfwd(void (double*)*, ...)
declare i64 @__enzyme_augmentsize(void (double*)*, ...)
declare void @__enzyme_reverse(void (double*)*, ...)

; CHECK: define void @test_derivative(double* %x, double* %dx)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %cache = alloca i8, i64 16
; CHECK-NEXT:   %0 = call { double, double } @augmented_tester(double* %x, double* %dx)
; CHECK-NEXT:   %1 = bitcast i8* %cache to { double, double }*
; CHECK-NEXT:   store { double, double } %0, { double, double }* %1
; CHECK-NEXT:   %2 = bitcast i8* %cache to { double, double }*
; CHECK-NEXT:   %3 = load { double, double }, { double, double }* %2
; CHECK-NEXT:   call void @diffetester(double* %x, double* %dx, { double, double } %3)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
