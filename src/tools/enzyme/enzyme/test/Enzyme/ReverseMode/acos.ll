; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x) {
entry:
  %0 = tail call fast double @acos(double %x)
  ret double %0
}

define double @test_derivative(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_autodiff(double (double)* nonnull @tester, double %x)
  ret double %0
}

; Function Attrs: nounwind readnone speculatable
declare double @acos(double)

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double)*, ...)

; CHECK: define internal { double } @diffetester(double %x, double %differeturn) 
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = fmul fast double %x, %x
; CHECK-NEXT:   %1 = fsub fast double 1.000000e+00, %0
; CHECK-NEXT:   %2 = call fast double @llvm.sqrt.f64(double %1)
; CHECK-NEXT:   %3 = fdiv fast double %differeturn, %2
; CHECK-NEXT:   %4 = {{(fsub fast double \-?0.000000e\+00,|fneg fast double)}} %3
; CHECK-NEXT:   %5 = insertvalue { double } undef, double %4, 0
; CHECK-NEXT:   ret { double } %5
; CHECK-NEXT: }
