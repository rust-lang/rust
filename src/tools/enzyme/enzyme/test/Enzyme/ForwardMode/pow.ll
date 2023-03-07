; RUN: %opt < %s %loadEnzyme -enzyme -instsimplify -enzyme-preopt=false -S | FileCheck %s
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(instsimplify)" -enzyme-preopt=false -S | FileCheck %s

; Function Attrs: noinline nounwind readnone uwtable
define double @tester(double %x, double %y) {
entry:
  %0 = tail call fast double @llvm.pow.f64(double %x, double %y)
  ret double %0
}

define double @test_derivative(double %x, double %y) {
entry:
  %0 = tail call double (double (double, double)*, ...) @__enzyme_fwddiff(double (double, double)* nonnull @tester, double %x, double 1.0, double %y, double 1.0)
  ret double %0
}

; Function Attrs: nounwind readnone speculatable
declare double @llvm.pow.f64(double, double)

; Function Attrs: nounwind
declare double @__enzyme_fwddiff(double (double, double)*, ...)

; CHECK: define internal {{(dso_local )?}}double @fwddiffetester(double %x, double %"x'", double %y, double %"y'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[i0:.+]] = fsub fast double %y, 1.000000e+00
; CHECK-NEXT:   %[[i1:.+]] = call fast double @llvm.pow.f64(double %x, double %[[i0]])
; CHECK-NEXT:   %[[i2:.+]] = fmul fast double %y, %[[i1]]
; CHECK-NEXT:   %[[dx:.+]] = fmul fast double %[[i2]], %"x'"
; CHECK-NEXT:   %[[i3:.+]] = call fast double @llvm.pow.f64(double %x, double %y)
; CHECK-NEXT:   %[[i4:.+]] = call fast double @llvm.log.f64(double %x)
; CHECK-DAG:    %[[i5:.+]] = fmul fast double %[[i3]], %[[i4]]
; CHECK-NEXT:   %[[dy:.+]] = fmul fast double %[[i5]], %"y'"
; CHECK-DAG:    %[[i6:.+]] = fadd fast double %[[dx]], %[[dy]]
; CHECK-NEXT:   ret double %[[i6]]
; CHECK-NEXT: }
