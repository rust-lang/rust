; RUN: %opt < %s %loadEnzyme -enzyme -enzyme_preopt=false -mem2reg -sroa -simplifycfg -instcombine -adce -S | FileCheck %s

; this check is done to ensure that we cannot do forward/reverse for f since it is used by g

define double @caller(double %M) {
entry:
  %call11 = call fast double @__enzyme_autodiff(i8* bitcast (double (double)* @todiff to i8*), double %M)
  ret double %call11
}

declare dso_local double @__enzyme_autodiff(i8*, double)

define linkonce_odr dso_local double @todiff(double %lhs) {
entry:
  %call = call double @f(double %lhs)
  %res = call double @g(double %call)
  ret double %res
}

define double @f(double %xpr) {
entry:
  ret double %xpr
}

define double @g(double %this) {
entry:
  ret double %this
}

; CHECK: define internal { double } @diffetodiff(double %lhs, double %differeturn) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %call_augmented = call { {}, double } @augmented_f(double %lhs)
; CHECK-NEXT:   %0 = extractvalue { {}, double } %call_augmented, 1
; CHECK-NEXT:   %1 = call { double } @diffeg(double %0, double %differeturn)
; CHECK-NEXT:   %2 = extractvalue { double } %1, 0
; CHECK-NEXT:   %3 = call { double } @diffef(double %lhs, double %2, {} undef)
; CHECK-NEXT:   %4 = extractvalue { double } %3, 0
; CHECK-NEXT:   %5 = insertvalue { double } undef, double %4, 0
; CHECK-NEXT:   ret { double } %5
; CHECK-NEXT: }

; CHECK: define internal { double } @diffeg(double %this, double %differeturn) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = insertvalue { double } undef, double %differeturn, 0
; CHECK-NEXT:   ret { double } %0
; CHECK-NEXT: }

; CHECK: define internal { {}, double } @augmented_f(double %xpr) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %.fca.1.insert = insertvalue { {}, double } undef, double %xpr, 1
; CHECK-NEXT:   ret { {}, double } %.fca.1.insert
; CHECK-NEXT: }

; CHECK: define internal { double } @diffef(double %xpr, double %differeturn, {} %tapeArg) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = insertvalue { double } undef, double %differeturn, 0
; CHECK-NEXT:   ret { double } %0
; CHECK-NEXT: }
