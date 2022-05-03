; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -instsimplify -simplifycfg -adce -S | FileCheck %s

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x, double %y) {
entry:
  %call = call double @hypot(double %x, double %y)
  ret double %call
}

define double @tester2(double %x) {
entry:
  %call = call double @hypot(double %x, double 2.000000e+00)
  ret double %call
}

define double @test_derivative(double %x, double %y) {
entry:
  %0 = tail call double (...) @__enzyme_autodiff(double (double, double)* nonnull @tester, double %x, double %y)
  %1 = tail call double (...) @__enzyme_autodiff(double (double)* nonnull @tester2, double %x)
  ret double %0
}

declare double @hypot(double, double)

; Function Attrs: nounwind
declare double @__enzyme_autodiff(...)

; CHECK: define internal { double, double } @diffetester(double %x, double %y, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call fast double @hypot(double %x, double %y)
; CHECK-NEXT:   %1 = fmul fast double %differeturn, %x
; CHECK-NEXT:   %2 = fdiv fast double %1, %0
; CHECK-NEXT:   %3 = fmul fast double %differeturn, %y
; CHECK-NEXT:   %4 = fdiv fast double %3, %0
; CHECK-NEXT:   %5 = insertvalue { double, double } undef, double %2, 0
; CHECK-NEXT:   %6 = insertvalue { double, double } %5, double %4, 1
; CHECK-NEXT:   ret { double, double } %6
; CHECK-NEXT: }

; CHECK: define internal { double } @diffetester2(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call fast double @hypot(double %x, double 2.000000e+00)
; CHECK-NEXT:   %1 = fmul fast double %differeturn, %x
; CHECK-NEXT:   %2 = fdiv fast double %1, %0
; CHECK-NEXT:   %3 = insertvalue { double } undef, double %2, 0
; CHECK-NEXT:   ret { double } %3
; CHECK-NEXT: }
