; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -instsimplify -S | FileCheck %s
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(instsimplify)" -enzyme-preopt=false -S | FileCheck %s

; Function Attrs: noinline nounwind readnone uwtable
define double @tester(double %x) {
entry:
  %agg1 = insertvalue [3 x double] undef, double %x, 0
  %mul = fmul double %x, %x
  %agg2 = insertvalue [3 x double] %agg1, double %mul, 1
  %add = fadd double %mul, 2.0
  %agg3 = insertvalue [3 x double] %agg2, double %add, 2
  %res = extractvalue [3 x double] %agg2, 1
  ret double %res
}

define double @test_derivative(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_fwddiff(double (double)* nonnull @tester, double %x, double 1.0)
  ret double %0
}

; Function Attrs: nounwind
declare double @__enzyme_fwddiff(double (double)*, ...)

; CHECK: define internal double @fwddiffetester(double %x, double %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = fmul fast double %"x'", %x
; CHECK-NEXT:   %1 = fmul fast double %"x'", %x
; CHECK-NEXT:   %2 = fadd fast double %0, %1
; CHECK-NEXT:   ret double %2
; CHECK-NEXT: }
