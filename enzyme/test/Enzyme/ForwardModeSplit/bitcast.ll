; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -simplifycfg -adce -instsimplify -S | FileCheck %s

define double @tester(double %x) {
entry:
  %y = bitcast double %x to i64
  %z = bitcast i64 %y to double
  ret double %z
}

define double @test_derivative(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_fwdsplit(double (double)* nonnull @tester, double %x, double 1.0, i8* null)
  ret double %0
}

declare double @__enzyme_fwdsplit(double (double)*, ...)

; CHECK: define internal {{(dso_local )?}}double @fwddiffetester(double %x, double %"x'", i8* %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret double %"x'"
; CHECK-NEXT: }
