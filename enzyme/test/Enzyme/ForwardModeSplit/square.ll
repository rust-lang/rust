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
  %0 = tail call double (double (double)*, ...) @__enzyme_fwdsplit(double (double)* nonnull @square, double %x, double 1.0, i8* null)
  ret double %0
}

declare double @__enzyme_fwdsplit(double (double)*, ...) 

; CHECK: define internal {{(dso_local )?}}double @fwddiffesquare(double %x, double %"x'", i8* %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   tail call void @free(i8* nonnull %tapeArg)
; CHECK-NEXT:   %0 = fmul fast double %"x'", %x
; CHECK-NEXT:   %1 = fadd fast double %0, %0
; CHECK-NEXT:   ret double %1
; CHECK-NEXT: }
