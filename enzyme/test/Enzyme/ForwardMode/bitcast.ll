; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -instsimplify -S | FileCheck %s
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(instsimplify)" -enzyme-preopt=false -S | FileCheck %s

define double @tester(double %x) {
entry:
  %y = bitcast double %x to i64
  %z = bitcast i64 %y to double
  ret double %z
}

define double @test_derivative(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_fwddiff(double (double)* nonnull @tester, double %x, double 1.0)
  ret double %0
}

declare double @__enzyme_fwddiff(double (double)*, ...)

; CHECK: define internal {{(dso_local )?}}double @fwddiffetester(double %x, double %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret double %"x'"
; CHECK-NEXT: }
