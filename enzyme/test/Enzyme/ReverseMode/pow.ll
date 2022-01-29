; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

; Function Attrs: noinline nounwind readnone uwtable
define double @tester(double %x, double %y) {
entry:
  %0 = tail call fast double @llvm.pow.f64(double %x, double %y)
  ret double %0
}

define double @test_derivative(double %x, double %y) {
entry:
  %0 = tail call double (double (double, double)*, ...) @__enzyme_autodiff(double (double, double)* nonnull @tester, double %x, double %y)
  ret double %0
}

; Function Attrs: nounwind readnone speculatable
declare double @llvm.pow.f64(double, double)

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double, double)*, ...)

; CHECK: define internal {{(dso_local )?}}{ double, double } @diffetester(double %x, double %y, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[ym1:.+]] = fsub fast double %y, 1.000000e+00
; CHECK-NEXT:   %[[newpow:.+]] = call fast double @llvm.pow.f64(double %x, double %[[ym1]])
; CHECK-NEXT:   %[[newpowdret:.+]] = fmul fast double %differeturn, %[[newpow]]
; CHECK-NEXT:   %[[dx:.+]] = fmul fast double %[[newpowdret]], %y
; CHECK-NEXT:   %[[origpow:.+]] = call fast double @llvm.pow.f64(double %x, double %y)
; CHECK-DAG:    %[[logy:.+]] = call fast double @llvm.log.f64(double %x)
; CHECK-DAG:    %[[origpowdret:.+]] = fmul fast double %differeturn, %[[origpow]]
; CHECK-NEXT:   %[[dy:.+]] = fmul fast double %[[origpowdret]], %[[logy]]
; CHECK-NEXT:   %[[interres:.+]] = insertvalue { double, double } undef, double %[[dx:.+]], 0
; CHECK-NEXT:   %[[finalres:.+]] = insertvalue { double, double } %[[interres]], double %[[dy:.+]], 1
; CHECK-NEXT:   ret { double, double } %[[finalres]]
; CHECK-NEXT: }
