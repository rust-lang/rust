; RUN: opt < %s %loadEnzyme -enzyme -mem2reg -instsimplify -simplifycfg -early-cse -S | FileCheck %s

; #include <math.h>
; #include <stdio.h>
; 
; __attribute__((noinline))
; void store(double *x, double *y) {
;   unsigned long long *xl = (unsigned long long*)x;
;   unsigned long long *yl = (unsigned long long*)y;
;   *yl = *xl;
; }
; 
; 
; void test_derivative(double* x, double *xp, double* y, double* yp) {
;   __builtin_autodiff(store, x, xp, y, yp);
; }

; Function Attrs: noinline norecurse nounwind uwtable
define dso_local void @store(double* nocapture readonly %x, double* nocapture %y) #0 {
entry:
  %0 = bitcast double* %x to i64*
  %1 = bitcast double* %y to i64*
  %2 = load i64, i64* %0, align 8, !tbaa !2
  store i64 %2, i64* %1, align 8, !tbaa !2
  ret void
}

; Function Attrs: nounwind uwtable
define dso_local void @test_derivative(double* %x, double* %xp, double* %y, double* %yp) local_unnamed_addr #1 {
entry:
  %0 = tail call double (void (double*, double*)*, ...) @__enzyme_autodiff(void (double*, double*)* nonnull @store, double* %x, double* %xp, double* %y, double* %yp)
  ret void
}

; Function Attrs: nounwind
declare double @__enzyme_autodiff(void (double*, double*)*, ...) #2

attributes #0 = { noinline norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #2 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0}
!3 = !{!"long long", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}

; CHECK: define internal {} @diffestore(double* nocapture readonly %x, double* %"x'", double* nocapture %y, double* %"y'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = bitcast double* %x to i64*
; CHECK-NEXT:   %1 = bitcast double* %y to i64*
; CHECK-NEXT:   %2 = load i64, i64* %0
; CHECK-NEXT:   store i64 %2, i64* %1
; CHECK-NEXT:   %"'ipc" = bitcast double* %"y'" to i64*
; CHECK-NEXT:   %3 = load i64, i64* %"'ipc"
; CHECK-NEXT:   store i64 0, i64* %"'ipc"
; CHECK-NEXT:   %"'ipc2" = bitcast double* %"x'" to i64*
; CHECK-NEXT:   %4 = load i64, i64* %"'ipc2"
; CHECK-NEXT:   %5 = bitcast i64 %3 to double
; CHECK-NEXT:   %6 = bitcast i64 %4 to double
; CHECK-NEXT:   %7 = fadd fast double %6, %5
; CHECK-NEXT:   %8 = bitcast double %7 to i64
; CHECK-NEXT:   store i64 %8, i64* %"'ipc2"
; CHECK-NEXT:   ret {} undef
; CHECK-NEXT: }

