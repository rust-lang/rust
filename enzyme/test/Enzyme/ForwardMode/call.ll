; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

; extern double __enzyme_fwddiff(double (double), double, double);

; __attribute__((noinline))
; double add2(double x) {
;     return 2 + x;
; }

; __attribute__((noinline))
; double add4(double x) {
;     return add2(x) + 2;
; }

; double dadd4(double x) {
;     return __enzyme_fwddiff(add4, x, 1.0);
; }


define dso_local double @add2(double %x) {
entry:
  %add = fadd double %x, 2.000000e+00
  ret double %add
}

define dso_local double @add4(double %x) {
entry:
  %call = call double @add2(double %x)
  %add = fadd double %call, 2.000000e+00
  ret double %add
}

define dso_local double @dadd4(double %x) {
entry:
  %call = call double @__enzyme_fwddiff(double (double)* nonnull @add4, double %x, double 1.000000e+00)
  ret double %call
}

declare dso_local double @__enzyme_fwddiff(double (double)*, double, double)



; CHECK: define internal {{(dso_local )?}}{ double } @diffeadd4(double %x, double %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call { double } @diffeadd2(double %x, double %"x'")
; CHECK-NEXT:   ret { double } %0
; CHECK-NEXT: }

; CHECK: define internal {{(dso_local )?}}{ double } @diffeadd2(double %x, double %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = insertvalue { double } undef, double %"x'", 0
; CHECK-NEXT:   ret { double } %0
; CHECK-NEXT: }
