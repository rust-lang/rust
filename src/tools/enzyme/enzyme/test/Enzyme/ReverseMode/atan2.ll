; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -early-cse -instsimplify -simplifycfg -adce -S | FileCheck %s

; Function Attrs: nounwind readnone uwtable
define double @tester(double %y, double %x) {
entry:
  %call = call double @atan2(double %y, double %x)
  ret double %call
}

define double @tester2(double %y) {
entry:
  %call = call double @atan2(double %y, double 2.000000e+00)
  ret double %call
}

define double @test_derivative(double %y, double %x) {
entry:
  %0 = tail call double (...) @__enzyme_autodiff(double (double, double)* nonnull @tester, double %y, double %x)
  %1 = tail call double (...) @__enzyme_autodiff(double (double)* nonnull @tester2, double %y)
  ret double %0
}

declare double @atan2(double, double)

; Function Attrs: nounwind
declare double @__enzyme_autodiff(...)

; CHECK: define internal { double, double } @diffetester(double %y, double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-DAG:    %[[a0:.+]] = fmul fast double %y, %y
; CHECK-DAG:    %[[a1:.+]] = fmul fast double %x, %x
; CHECK-DAG:   %[[a2:.+]] = fadd fast double %[[a1]], %[[a0]]
; CHECK-DAG:   %[[a3:.+]] = fmul fast double %differeturn, %x
; CHECK-DAG:   %[[a4:.+]] = fdiv fast double %[[a3]], %[[a2]]
; CHECK-DAG:   %[[a5:.+]] = fmul fast double %differeturn, %y
; CHECK-DAG:   %[[a6:.+]] = fdiv fast double %[[a5]], %[[a2]]
; CHECK-DAG:   %[[a7:.+]] = {{(fneg fast double)|(fsub fast double (-)?0.000000e\+00,)}} %[[a6]]
; CHECK-DAG:   %[[a8:.+]] = insertvalue { double, double } undef, double %4, 0
; CHECK-DAG:   %[[a9:.+]] = insertvalue { double, double } %[[a8]], double %[[a7]], 1
; CHECK-DAG:   ret { double, double } %[[a9]]
; CHECK-NEXT: }

; CHECK: define internal { double } @diffetester2(double %y, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-DAG:   %[[a0:.+]] = fmul fast double %y, %y
; CHECK-DAG:   %[[a1:.+]] = fadd fast double 4.000000e+00, %[[a0]]
; CHECK-DAG:   %[[a2:.+]] = fmul fast double %differeturn, 2.000000e+00
; CHECK-DAG:   %[[a3:.+]] = fdiv fast double %[[a2:.+]], %[[a1]]
; CHECK-DAG:   %[[a4:.+]] = insertvalue { double } undef, double %[[a3]], 0
; CHECK-NEXT:   ret { double } %[[a4]]
; CHECK-NEXT: }
