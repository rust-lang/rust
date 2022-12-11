; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

; Function Attrs: noinline nounwind readnone uwtable
define double @tester(double %x, double %y) {
entry:
  %0 = frem fast double %x, %y
  ret double %0
}

define double @test_derivative(double %x, double %y) {
entry:
  %0 = tail call double (double (double, double)*, ...) @__enzyme_fwddiff(double (double, double)* nonnull @tester, double %x, double 1.0, double %y, double 1.0)
  ret double %0
}

; Function Attrs: nounwind
declare double @__enzyme_fwddiff(double (double, double)*, ...)


; CHECK: define internal double @fwddiffetester(double %x, double %"x'", double %y, double %"y'") 
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[i0:.+]] = fdiv fast double %x, %y
; CHECK-NEXT:   %[[i1:.+]] = call fast double @llvm.fabs.f64(double %[[i0]])
; CHECK-NEXT:   %[[i2:.+]] = call fast double @llvm.floor.f64(double %[[i1]])
; CHECK-NEXT:   %[[i3:.+]] = call fast double @llvm.copysign.f64(double %[[i2]], double %[[i0]])
; CHECK-NEXT:   %[[i4:.+]] = {{(fsub fast double \-?0.000000e\+00,|fneg fast double)}} %[[i3]]
; CHECK-NEXT:   %[[i5:.+]] = fmul fast double %"y'", %[[i4]]
; CHECK-NEXT:   %[[i6:.+]] = fadd fast double %"x'", %[[i5]]
; CHECK-NEXT:   ret double %[[i6]]
; CHECK-NEXT: }
