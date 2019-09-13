; RUN: opt < %s %loadEnzyme -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

; Function Attrs: noinline nounwind readnone uwtable
define double @tester(double %x, double %y) {
entry:
  %0 = fdiv fast double %x, %y
  ret double %0
}

define double @test_derivative(double %x, double %y) {
entry:
  %0 = tail call double (double (double, double)*, ...) @__enzyme_autodiff(double (double, double)* nonnull @tester, double %x, double %y)
  ret double %0
}

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double, double)*, ...)

; CHECK: define internal {{(dso_local )?}}{ double, double } @diffetester(double %x, double %y, double %[[differet:.+]])
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[diffex:.+]] = fdiv fast double %[[differet]], %y
; CHECK-NEXT:   %[[xdivy:.+]] = fdiv fast double %x, %y
; CHECK-NEXT:   %[[xdivydret:.+]] = fmul fast double %[[differet]], %[[xdivy]]
; CHECK-NEXT:   %[[xdivy2:.+]] = fdiv fast double %[[xdivydret]], %y
; CHECK-NEXT:   %[[mxdivy2:.+]] = fsub fast double -0.000000e+00, %[[xdivy2]]
; CHECK-NEXT:   %[[res1:.+]] = insertvalue { double, double } undef, double %[[diffex]], 0
; CHECK-NEXT:   %[[res2:.+]] = insertvalue { double, double } %[[res1:.+]], double %[[mxdivy2]], 1
; CHECK-NEXT:   ret { double, double } %[[res2]]
; CHECK-NEXT: }
