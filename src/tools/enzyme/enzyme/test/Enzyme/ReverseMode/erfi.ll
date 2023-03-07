; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -instsimplify -simplifycfg -adce -S | FileCheck %s

declare double @erfi(double)

define double @tester(double %x) {
entry:
  %call = call double @erfi(double %x)
  ret double %call
}

define double @test_derivative(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_autodiff(double (double)* nonnull @tester, double %x)
  ret double %0
}

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double)*, ...)

; CHECK: define internal { double } @diffetester(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:    %0 = fmul fast double %x, %x
; CHECK-NEXT:    %1 = call fast double @llvm.exp.f64(double %0)
; CHECK-NEXT:    %2 = fmul fast double %1, 0x3FF20DD750429B6D
; CHECK-NEXT:    %3 = fmul fast double %2, %differeturn
; CHECK-NEXT:    %4 = insertvalue { double } undef, double %3, 0
; CHECK-NEXT:    ret { double } %4
; CHECK-NEXT: }
