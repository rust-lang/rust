; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -simplifycfg -early-cse -S | FileCheck %s

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
; CHECK-NEXT:   %[[x2:.+]] = fmul fast double %[[differet]], %x
; CHECK-NEXT:   %[[result:.+]] = fadd fast double %[[x2]], %[[x2]]
; CHECK-NEXT:   %[[iv:.+]] = insertvalue { double } undef, double %[[result]]
; CHECK-NEXT:   ret { double } %[[iv]]
; CHECK-NEXT: }
