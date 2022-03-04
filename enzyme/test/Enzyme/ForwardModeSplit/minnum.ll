; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -early-cse -simplifycfg -S | FileCheck %s

define double @tester(double %x, double %y) {
entry:
  %0 = tail call double @llvm.minnum.f64(double %x, double %y)
  ret double %0
}

define double @test_derivative(double %x, double %y) {
entry:
  %0 = tail call double (double (double, double)*, ...) @__enzyme_fwdsplit(double (double, double)* nonnull @tester, double %x, double 1.0, double %y, double 1.0, i8* null)
  ret double %0
}

declare double @llvm.minnum.f64(double, double)

declare double @__enzyme_fwdsplit(double (double, double)*, ...)

; CHECK: define internal {{(dso_local )?}}double @fwddiffetester(double %x, double %"x'", double %y, double %"y'", i8* %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   tail call void @free(i8* nonnull %tapeArg)
; CHECK-NEXT:   %0 = fcmp fast olt double %x, %y
; CHECK-NEXT:   %1 = select{{( fast)?}} i1 %0, double %"x'", double %"y'"
; CHECK-NEXT:   ret double %1
; CHECK-NEXT: }
