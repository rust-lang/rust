; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

; Function Attrs: noinline nounwind readnone uwtable
define double @tester(double %x) {
entry:
  tail call void @myprint(double %x)
  ret double %x
}

define double @test_derivative(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_autodiff(double (double)* nonnull @tester, double %x)
  ret double %0
}

declare void @myprint(double %x) #0

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double)*, ...)

attributes #0 = { "enzyme_inactive" }

; CHECK: define internal {{(dso_local )?}}{ double } @diffetester(double %x, double %[[differet:.+]])
; CHECK-NEXT: entry:
; CHECK-NEXT:   tail call void @myprint(double %x)
; CHECK-NEXT:   %0 = insertvalue { double } undef, double %differeturn, 0
; CHECK-NEXT:   ret { double } %0
; CHECK-NEXT: }
