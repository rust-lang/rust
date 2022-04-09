; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

; Function Attrs: noinline nounwind readnone uwtable
define double @tester(double %x) {
entry:
  tail call void @myprint(double %x) #0
  ret double %x
}

define double @test_derivative(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_fwdsplit(double (double)* nonnull @tester, double %x, double 1.0, i8* null)
  ret double %0
}

declare void @myprint(double %x)

; Function Attrs: nounwind
declare double @__enzyme_fwdsplit(double (double)*, ...)

attributes #0 = { "enzyme_inactive" }

; CHECK: define internal {{(dso_local )?}}double @fwddiffetester(double %x, double %"x'", i8* %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret double %"x'"
; CHECK-NEXT: }
