; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -instsimplify -S | FileCheck %s
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(instsimplify)" -enzyme-preopt=false -S | FileCheck %s

; Function Attrs: noinline nounwind readnone uwtable
define double @tester(double %x) {
entry:
  %cstx = bitcast double %x to i64 
  %negx = xor i64 %cstx, -9223372036854775808
  %csty = bitcast i64 %negx to double
  ret double %csty
}

define double @test_derivative(double %x, double %dx) {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_fwddiff(double (double)* nonnull @tester, double %x, double %dx)
  ret double %0
}

; Function Attrs: nounwind
declare double @__enzyme_fwddiff(double (double)*, ...)

; CHECK: define internal double @fwddiffetester(double %x, double %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = {{(fsub fast double \-?0.000000e\+00,|fneg fast double)}} %"x'"
; CHECK-NEXT:   ret double %0
; CHECK-NEXT: }
