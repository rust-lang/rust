; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

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



; CHECK: define internal {{(dso_local )?}}double @fwddiffeadd4(double %x, double %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call fast double @fwddiffeadd2(double %x, double %"x'")
; CHECK-NEXT:   ret double %0
; CHECK-NEXT: }

; CHECK: define internal {{(dso_local )?}}double @fwddiffeadd2(double %x, double %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret double %"x'"
; CHECK-NEXT: }
