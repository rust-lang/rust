; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

; Function Attrs: norecurse nounwind readnone uwtable
define dso_local double @max(double %x, double %y) #0 {
entry:
  %cmp = fcmp fast ogt double %x, %y
  %cond = select i1 %cmp, double %x, double %y
  ret double %cond
}

; Function Attrs: nounwind uwtable
define dso_local double @test_derivative(double %x, double %y) local_unnamed_addr #1 {
entry:
  %0 = tail call double (double (double, double)*, ...) @__enzyme_autodiff(double (double, double)* nonnull @max, double %x, double %y)
  ret double %0
}

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double, double)*, ...)

; CHECK: define internal {{(dso_local )?}}{ double, double } @diffemax(double %x, double %y, double {{(noundef )?}}%[[differet:.+]])
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[cmp:.+]] = fcmp fast ogt double %x, %y
; CHECK-NEXT:   %[[diffex:.+]] = select {{(fast )?}}i1 %[[cmp]], double %[[differet]], double 0.000000e+00
; CHECK-NEXT:   %[[diffey:.+]] = select {{(fast )?}}i1 %[[cmp]], double 0.000000e+00, double %[[differet]]
; CHECK-NEXT:   %[[insert1:.+]] = insertvalue { double, double } undef, double %[[diffex]], 0
; CHECK-NEXT:   %[[result:.+]] = insertvalue { double, double } %[[insert1]], double %[[diffey]], 1
; CHECK-NEXT:   ret { double, double } %[[result]]
; CHECK-NEXT: }
