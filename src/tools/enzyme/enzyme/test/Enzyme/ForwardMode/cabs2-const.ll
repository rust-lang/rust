; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -instsimplify -adce -S | FileCheck %s
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(instsimplify,adce)" -enzyme-preopt=false -S | FileCheck %s

; Function Attrs: nounwind readnone willreturn
declare double @cabs([2 x double])

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x, double %y) {
entry:
  %agg0 = insertvalue [2 x double] undef, double %x, 0
  %agg1 = insertvalue [2 x double] %agg0, double %y, 1
  %call = call double @cabs([2 x double] %agg1)
  ret double %call
}

define double @test_derivative(double %x, double %y) {
entry:
  %0 = tail call double (double (double, double)*, ...) @__enzyme_fwddiff(double (double, double)* nonnull @tester, metadata !"enzyme_const", double %x, double %y, double 1.0)
  ret double %0
}

; Function Attrs: nounwind
declare double @__enzyme_fwddiff(double (double, double)*, ...)


; CHECK: define internal double @fwddiffetester(double %x, double %y, double %"y'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %agg0 = insertvalue [2 x double] undef, double %x, 0
; CHECK-NEXT:   %"agg1'ipiv" = insertvalue [2 x double] zeroinitializer, double %"y'", 1
; CHECK-NEXT:   %agg1 = insertvalue [2 x double] %agg0, double %y, 1
; CHECK-NEXT:   %0 = call fast double @cabs([2 x double] %agg1)
; CHECK-NEXT:   %1 = extractvalue [2 x double] %"agg1'ipiv", 0
; CHECK-NEXT:   %2 = fdiv fast double %1, %0
; CHECK-NEXT:   %3 = fmul fast double %x, %2
; CHECK-NEXT:   %4 = fdiv fast double %"y'", %0
; CHECK-NEXT:   %5 = fmul fast double %y, %4
; CHECK-NEXT:   %6 = fadd fast double %3, %5
; CHECK-NEXT:   ret double %6
; CHECK-NEXT: }
