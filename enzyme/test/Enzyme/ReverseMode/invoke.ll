; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -simplifycfg -adce -S | FileCheck %s

define double @sq(double %x) {
entry:
  %0 = fmul fast double %x, %x
  ret double %0
}

declare i32 @__gxx_personality_v0(...)

; Function Attrs: norecurse ssp uwtable
define double @caller(double %x) personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)  {
  %res = invoke double (...) @_Z17__enzyme_autodiffz(double (double)* nonnull @sq, double %x)
          to label %eblock unwind label %cblock

eblock:
  ret double %res

cblock:
  %lp = landingpad { i8*, i32 }
          cleanup
  ret double 0.000000e+00
}

declare double @_Z17__enzyme_autodiffz(...)

; CHECK: define double @caller(double %x)
; CHECK-NEXT: eblock:
; CHECK-NEXT:   %0 = call { double } @diffesq(double %x, double 1.000000e+00)
; CHECK-NEXT:   %1 = extractvalue { double } %0, 0
; CHECK-NEXT:   ret double %1
; CHECK-NEXT: }