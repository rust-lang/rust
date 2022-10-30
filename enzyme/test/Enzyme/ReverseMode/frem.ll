; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -early-cse -instsimplify -simplifycfg -S | FileCheck %s

; Function Attrs: noinline nounwind readnone uwtable
define double @tester(double %x, double %y) {
entry:
  %0 = frem fast double %x, %y
  ret double %0
}

define double @test_derivative(double %x, double %y) {
entry:
  %0 = tail call double (double (double, double)*, ...) @__enzyme_autodiff(double (double, double)* nonnull @tester, double %x, double %y)
  ret double %0
}

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double, double)*, ...)

; CHECK: define internal { double, double } @diffetester(double %x, double %y, double %differeturn) 
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[i0:.+]] = fdiv fast double %x, %y
; CHECK-NEXT:   %[[i1:.+]] = call fast double @llvm.fabs.f64(double %[[i0]])
; CHECK-NEXT:   %[[i2:.+]] = call fast double @llvm.floor.f64(double %[[i1]])
; CHECK-NEXT:   %[[i3:.+]] = call fast double @llvm.copysign.f64(double %[[i2]], double %[[i0]])
; CHECK-NEXT:   %[[i4:.+]] = {{(fsub fast double \-?0.000000e\+00,|fneg fast double)}} %[[i3]]
; CHECK-NEXT:   %[[i5:.+]] = fmul fast double %differeturn, %[[i4]]
; CHECK-NEXT:   %[[i6:.+]] = insertvalue { double, double } undef, double %differeturn, 0
; CHECK-NEXT:   %[[i7:.+]] = insertvalue { double, double } %[[i6]], double %[[i5]], 1
; CHECK-NEXT:   ret { double, double } %[[i7]]
; CHECK-NEXT: }
