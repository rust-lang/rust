; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

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
  %0 = tail call double (double (double, double)*, ...) @__enzyme_fwddiff(double (double, double)* nonnull @max, double %x, double 1.0, double %y, double 1.0)
  ret double %0
}

; Function Attrs: nounwind
declare double @__enzyme_fwddiff(double (double, double)*, ...)


; CHECK: define internal double @fwddiffemax(double %x, double %"x'", double %y, double %"y'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %cmp = fcmp fast ogt double %x, %y
; CHECK-NEXT:   %[[i0:.+]] = select {{(fast )?}}i1 %cmp, double %"x'", double %"y'"
; CHECK-NEXT:   ret double %[[i0]]
; CHECK-NEXT: }
