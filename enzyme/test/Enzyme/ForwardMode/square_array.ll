; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -simplifycfg -early-cse -S | FileCheck %s

define { double, double } @squared(double %x) {
entry:
  %mul = fmul double %x, %x
  %mul2 = fmul double %mul, %x
  %.fca.0.insert = insertvalue { double, double } undef, double %mul, 0
  %.fca.1.insert = insertvalue { double, double } %.fca.0.insert, double %mul2, 1
  ret { double, double } %.fca.1.insert
}

define { double, double } @dsquared(double %x) {
entry:
  %call = call { double, double } (i8*, ...) @__enzyme_fwddiff(i8* bitcast ({ double, double } (double)* @squared to i8*), double %x, double 1.0)
  ret { double, double } %call
}

declare { double, double } @__enzyme_fwddiff(i8*, ...)



; CHECK: define internal {{(dso_local )?}}{ double, double } @fwddiffesquared(double %x, double %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %mul = fmul double %x, %x
; CHECK-NEXT:   %0 = fmul fast double %"x'", %x
; CHECK-NEXT:   %1 = fadd fast double %0, %0
; CHECK-NEXT:   %2 = fmul fast double %1, %x
; CHECK-NEXT:   %3 = fmul fast double %"x'", %mul
; CHECK-NEXT:   %4 = fadd fast double %2, %3
; CHECK-NEXT:   %5 = insertvalue { double, double } zeroinitializer, double %1, 0
; CHECK-NEXT:   %6 = insertvalue { double, double } %5, double %4, 1
; CHECK-NEXT:   ret { double, double } %6
; CHECK-NEXT: }
