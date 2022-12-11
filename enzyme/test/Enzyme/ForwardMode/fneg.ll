; RUN: if [ %llvmver -ge 10 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 10 ]; then %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s; fi

; extern double __enzyme_fwddiff(void*, double, double);
;
; double fneg(double x) {
;     return -x;
; }
; 
; double dfneg(double x) {
;     return __enzyme_fwddiff((void*)fneg, x, 1.0);
; }


define double @fneg(double %x) {
  %fneg = fneg double %x
  ret double %fneg
}

define double @dfneg(double %x) {
  %1 = call double @__enzyme_fwddiff(double (double)* @fneg, double %x, double 1.0)
  ret double %1
}

declare double @__enzyme_fwddiff(double (double)*, double, double)


; CHECK: define internal double @fwddiffefneg(double %x, double %"x'")
; CHECK-NEXT:   %1 = fneg fast double %"x'"
; CHECK-NEXT:   ret double %1
; CHECK-NEXT: }
