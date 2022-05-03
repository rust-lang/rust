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

define internal { {}, double } @augment_add2(double %x) {
entry:
  %add = fadd fast double %x, 2.000000e+00
  %struct1 = insertvalue { {}, double } undef, double %add, 1
  ret { {}, double } %struct1
}

define internal { double } @gradient_add2(double %x, double %differet, {} %tapeArg) {
entry:
  %struct1 = insertvalue { double } undef, double %differet, 0
  ret { double } %struct1
}

define internal { double, double } @fr_add2(double %x, double %differet) {
entry:
  %add = fadd fast double %x, 2.000000e+00
  %struct1 = insertvalue { double, double } undef, double %differet, 1
  %struct2 = insertvalue { double, double } %struct1, double %add, 0
  ret { double, double } %struct2
}

declare !enzyme_augment !{{ {}, double } (double)* @augment_add2} !enzyme_gradient !{{ double } (double, double, {})* @gradient_add2} double @add2(double %x)
; entry:
;   %add = fadd fast double %x, 2.000000e+00
;   ret double %add
; }

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

; CHECK: define internal { double } @diffeadd4(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call { double } @fixgradient_add2(double %x, double %differeturn)
; CHECK-NEXT:   ret { double } %0
; CHECK-NEXT: }

; CHECK: define internal { double } @fixgradient_add2(double %arg0, double %arg1)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call { {}, double } @augment_add2(double %arg0)
; CHECK-NEXT:   %1 = call { double } @gradient_add2(double %arg0, double %arg1, {} undef)
; CHECK-NEXT:   ret { double } %1
; CHECK-NEXT: }
