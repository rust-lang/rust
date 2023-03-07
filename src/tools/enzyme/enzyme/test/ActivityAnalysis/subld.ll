; RUN: %opt < %s %loadEnzyme -print-activity-analysis -activity-analysis-func=submalloced -o /dev/null | FileCheck %s

define double* @f(double** nocapture readonly %a0) readonly {
entry:
  %a2 = load double*, double** %a0, align 8
  ret double* %a2
}

define double @submalloced(double* %a0) {
entry: 
  %p3 = alloca double*, align 8
  store double* %a0, double** %p3, align 8
  %a4 = call double* @f(double** nonnull %p3)
  %r = load double, double* %a4, align 8
  ret double %r
}

; CHECK: double* %a0: icv:0
; CHECK-NEXT: entry
; CHECK-NEXT:   %p3 = alloca double*, align 8: icv:0 ici:1
; CHECK-NEXT:   store double* %a0, double** %p3, align 8: icv:1 ici:0
; CHECK-NEXT:   %a4 = call double* @f(double** nonnull %p3): icv:0 ici:1
; CHECK-NEXT:   %r = load double, double* %a4, align 8: icv:0 ici:0
; CHECK-NEXT:   ret double %r: icv:1 ici:1
