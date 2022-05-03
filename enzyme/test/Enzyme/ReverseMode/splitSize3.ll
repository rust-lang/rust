; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

; Function Attrs: noinline nounwind readnone uwtable
define double* @tester(double* %x) {
entry:
  %gep = getelementptr double, double* %x, i32 1
  %y = load double, double* %x
  %z = load double, double* %gep
  %res = fmul fast double %y, %z
  store double %res, double* %x
  ret double* %x
}

declare void @printOutput({ double*, double* } %res)

define void @test_derivative(double* %x, double* %dx) {
entry:
  %size = call i64 (double* (double*)*, ...) @__enzyme_augmentsize(double* (double*)* nonnull @tester, metadata !"enzyme_dup")
  %cache = alloca i8, i64 %size, align 1
  %aug_out = tail call { double*, double* } (double* (double*)*, ...) @__enzyme_augmentfwd(double* (double*)* nonnull @tester, metadata !"enzyme_allocated", i64 %size, metadata !"enzyme_tape", i8* %cache, double* %x, double* %dx)
  call void @printOutput({ double*, double* } %aug_out)
  tail call void (double* (double*)*, ...) @__enzyme_reverse(double* (double*)* nonnull @tester, metadata !"enzyme_allocated", i64 %size, metadata !"enzyme_tape", i8* %cache, double* %x, double* %dx)
  ret void
}

; Function Attrs: nounwind
declare { double*, double* } @__enzyme_augmentfwd(double* (double*)*, ...)
declare i64 @__enzyme_augmentsize(double* (double*)*, ...)
declare void @__enzyme_reverse(double* (double*)*, ...)

; CHECK: define void @test_derivative(double* %x, double* %dx)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %cache = alloca i8, i64 16
; CHECK-NEXT:   %0 = call { { double, double }, double*, double* } @augmented_tester(double* %x, double* %dx)
; CHECK-NEXT:   %1 = extractvalue { { double, double }, double*, double* } %0, 0
; CHECK-NEXT:   %2 = bitcast i8* %cache to { double, double }*
; CHECK-NEXT:   store { double, double } %1, { double, double }* %2
; CHECK-NEXT:   %3 = extractvalue { { double, double }, double*, double* } %0, 1
; CHECK-NEXT:   %4 = insertvalue { double*, double* } undef, double* %3, 0
; CHECK-NEXT:   %5 = extractvalue { { double, double }, double*, double* } %0, 2
; CHECK-NEXT:   %6 = insertvalue { double*, double* } %4, double* %5, 1
; CHECK-NEXT:   call void @printOutput({ double*, double* } %6)
; CHECK-NEXT:   %7 = bitcast i8* %cache to { double, double }*
; CHECK-NEXT:   %8 = load { double, double }, { double, double }* %7
; CHECK-NEXT:   call void @diffetester(double* %x, double* %dx, { double, double } %8)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
