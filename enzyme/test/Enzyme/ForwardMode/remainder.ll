; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x, double %y) {
entry:
  %0 = tail call fast double @remainder(double %x, double %y)
  ret double %0
}

define double @test_derivative1(double %x, double %y) {
entry:
  %0 = tail call double (double (double, double)*, ...) @__enzyme_fwddiff(double (double, double)* nonnull @tester, metadata !"enzyme_const", double %x, double %y, double 1.0)
  ret double %0
}

define double @test_derivative2(double %x, double %y) {
entry:
  %0 = tail call double (double (double, double)*, ...) @__enzyme_fwddiff(double (double, double)* nonnull @tester, double %x, double 1.0, metadata !"enzyme_const", double %y)
  ret double %0
}

; Function Attrs: nounwind readnone speculatable
declare double @remainder(double, double)

; Function Attrs: nounwind
declare double @__enzyme_fwddiff(double (double, double)*, ...)

; CHECK: define internal double @fwddiffetester(double %x, double %y, double %"y'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = {{(fneg fast double|fsub fast double \-0\.000000e\+00,)}} %"y'"
; CHECK-NEXT:   %1 = fdiv fast double %x, %y
; CHECK-NEXT:   %2 = call fast double @llvm.round.f64(double %1)
; CHECK-NEXT:   %3 = fmul fast double %0, %2
; CHECK-NEXT:   ret double %3
; CHECK-NEXT: }

; CHECK: define internal double @fwddiffetester.1(double %x, double %"x'", double %y)
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret double %"x'"
; CHECK-NEXT: }
