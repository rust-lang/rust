; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -simplifycfg -adce -instsimplify -S | FileCheck %s

define double @tester(float %x) {
entry:
  %y = fpext float %x to double
  ret double %y
}

define double @test_derivative(float %x) {
entry:
  %0 = tail call double (double (float)*, ...) @__enzyme_fwdsplit(double (float)* nonnull @tester, float %x, float 1.0, i8* null)
  ret double %0
}

declare double @__enzyme_fwdsplit(double (float)*, ...)

; CHECK: define internal {{(dso_local )?}}double @fwddiffetester(float %x, float %"x'", i8* %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[i0:.+]] = fpext float %"x'" to double
; CHECK-NEXT:   ret double %[[i0]]
; CHECK-NEXT: }
