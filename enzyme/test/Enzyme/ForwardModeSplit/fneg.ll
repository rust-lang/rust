; RUN: if [ %llvmver -ge 10 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi

; extern double __enzyme_fwdsplit(void*, double, double);
;
; double fneg(double x) {
;     return -x;
; }
; 
; double dfneg(double x) {
;     return __enzyme_fwdsplit((void*)fneg, x, 1.0);
; }


define double @fneg(double %x) {
  %fneg = fneg double %x
  ret double %fneg
}

define double @dfneg(double %x) {
  %1 = call double @__enzyme_fwdsplit(double (double)* @fneg, double %x, double 1.0, i8* null)
  ret double %1
}

declare double @__enzyme_fwdsplit(double (double)*, double, double, i8*)


; CHECK: define internal double @fwddiffefneg(double %x, double %"x'", i8* %tapeArg)
; CHECK-NEXT:   %1 = fneg fast double %"x'"
; CHECK-NEXT:   ret double %1
; CHECK-NEXT: }
