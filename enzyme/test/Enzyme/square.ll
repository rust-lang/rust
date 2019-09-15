; RUN: opt < %s %loadEnzyme -enzyme -enzyme_preopt=false -mem2reg -simplifycfg -instcombine -S | FileCheck %s

; source code
; double square(double x) {
;     return x * x;
; }
; 
; double dsquare(double x) {
;     return __builtin_autodiff(square, x);
; }

define double @square(double %x) {
entry:
  %mul = fmul fast double %x, %x
  ret double %mul
}

define double @dsquare(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_autodiff(double (double)* nonnull @square, double %x)
  ret double %0
}

declare double @__enzyme_autodiff(double (double)*, ...) 

; CHECK: define internal {{(dso_local )?}}{ double } @diffesquare(double %x, double %[[differet:.+]])
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[x2:.+]] = fadd fast double %x, %x
; CHECK-NEXT:   %[[result:.+]] = fmul fast double %[[x2]], %[[differet]]
; CHECK-NEXT:   %[[iv:.+]] = insertvalue { double } undef, double %[[result]]
; CHECK-NEXT:   ret { double } %[[iv]]
; CHECK-NEXT: }
