; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x) {
entry:
  %0 = tail call fast double @Faddeeva_erfcx_re(double %x)
  ret double %0
}

define double @test_derivative(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_autodiff(double (double)* nonnull @tester, double %x)
  ret double %0
}

declare double @Faddeeva_erfcx_re(double)

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double)*, ...)

; CHECK: define internal { double } @diffetester(double %x, double %differeturn) 
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[i0:.+]] = call fast double @Faddeeva_erfcx_re(double %x) 
; CHECK-NEXT:   %[[i1:.+]] = fmul fast double %x, %[[i0]]
; CHECK-NEXT:   %[[i2:.+]] = fsub fast double %[[i1]], 0x3FE20DD750429B6D
; CHECK-NEXT:   %[[i3:.+]] = fmul fast double 2.000000e+00, %[[i2]]
; CHECK-NEXT:   %[[i4:.+]] = fmul fast double %differeturn, %[[i3]]
; CHECK-NEXT:   %[[i5:.+]] = insertvalue { double } undef, double %[[i4]], 0
; CHECK-NEXT:   ret { double } %[[i5]]
; CHECK-NEXT: }
