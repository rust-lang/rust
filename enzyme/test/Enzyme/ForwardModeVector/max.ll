; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

%struct.Gradients = type { double, double }

; Function Attrs: nounwind
declare %struct.Gradients @__enzyme_fwddiff(double (double, double)*, ...)

; Function Attrs: norecurse nounwind readnone uwtable
define dso_local double @max(double %x, double %y) #0 {
entry:
  %cmp = fcmp fast ogt double %x, %y
  %cond = select i1 %cmp, double %x, double %y
  ret double %cond
}

; Function Attrs: nounwind uwtable
define dso_local %struct.Gradients @test_derivative(double %x, double %y) local_unnamed_addr #1 {
entry:
  %0 = tail call %struct.Gradients (double (double, double)*, ...) @__enzyme_fwddiff(double (double, double)* nonnull @max, metadata !"enzyme_width", i64 2, double %x, double 1.0, double 0.0, double %y, double 0.0, double 1.0)
  ret %struct.Gradients %0
}


; CHECK: define internal [2 x double] @fwddiffe2max(double %x, [2 x double] %"x'", double %y, [2 x double] %"y'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %cmp = fcmp fast ogt double %x, %y
; CHECK-NEXT:   %0 = select {{(fast )?}}i1 %cmp, [2 x double] %"x'", [2 x double] %"y'"
; CHECK-NEXT:   ret [2 x double] %0
; CHECK-NEXT: }