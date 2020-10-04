; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

; __attribute__((noinline))
; double add2(double x) {
;     return 2 + x;
; }
; 
; __attribute__((noinline))
; double add4(double x) {
;     return add2(x) + 2;
; }
; 
; double dadd4(double x) {
;     return __builtin_autodiff(add4, x);
; }

define dso_local double @add2(double %x) #0 {
entry:
  %add = fadd fast double %x, 2.000000e+00
  ret double %add
}

define dso_local double @add4(double %x) #0 {
entry:
  %call = tail call fast double @add2(double %x)
  %add = fadd fast double %call, 2.000000e+00
  ret double %add
}

define dso_local double @dadd4(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_autodiff(double (double)* nonnull @add4, double %x)
  ret double %0
}

attributes #0 = { readnone }

declare double @__enzyme_autodiff(double (double)*, ...)

; CHECK: define internal {{(dso_local )?}}{ double } @diffeadd4(double %x, double %[[differet:.+]])
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call { double } @diffeadd2(double %x, double %[[differet]])
; CHECK-NEXT:   ret { double } %0
; CHECK-NEXT: }

; CHECK: define internal {{(dso_local )?}}{ double } @diffeadd2(double %x, double %[[differet:.+]])
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[result2:.+]] = insertvalue { double } undef, double %[[differet]], 0
; CHECK-NEXT:   ret { double } %[[result2]]
; CHECK-NEXT: }
