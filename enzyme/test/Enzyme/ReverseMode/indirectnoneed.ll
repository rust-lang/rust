; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -adce -loop-deletion -correlated-propagation -simplifycfg -S | FileCheck %s

declare void @__enzyme_autodiff(...)
define void @caller(double** %outp, double** %doutp, double** %inp, double** %dinp) {
entry:
  call void (...) @__enzyme_autodiff(i8* bitcast (void (double**, double**)* @f to i8*), metadata !"enzyme_dupnoneed", double** %outp, double** %doutp, double** %inp, double** %dinp)
  ret void
}

; Function Attrs: noinline nounwind uwtable
define internal void @f(double** %outp, double** %inp) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %inc, %loop ]
  %inc = add i64 %i, 1
  %cmp = icmp eq i64 %inc, 10
  %out = load double*, double** %outp, align 8, !alias.scope !11, !noalias !101
  %in = load double*, double** %inp, align 8, !alias.scope !12, !noalias !102
  %v = load double, double* %in, align 8, !alias.scope !13, !noalias !103
  %v2 = fmul double %v, %v
  store double %v2, double* %out, align 8, !alias.scope !14, !noalias !104
  br i1 %cmp, label %exit, label %loop
  ; %tmp5.i.i29 = load i64, i64* %tmp1.i27, align 8, !tbaa !26, !noalias !23
  ; store i64 %tmp5.i.i29, i64* %tmp4.i.i28, align 8, !tbaa !26, !alias.scope !23
exit:
  ret void
}

; alias domain
!0 = !{!0}

; Distinct scopes in the domain
!1 = !{!1, !0}
!2 = !{!2, !0}
!3 = !{!3, !0}
!4 = !{!4, !0}

!11 = !{!1}
!12 = !{!2}
!13 = !{!3}
!14 = !{!4}

!101 = !{!2, !3, !4}
!102 = !{!1, !3, !4}
!103 = !{!1, !2, !4}
!104 = !{!1, !2, !3}

; CHECK: define internal void @diffef(double** %outp, double** %"outp'", double** %inp, double** %"inp'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   br label %invertloop

; CHECK: invertentry:                                      ; preds = %invertloop
; CHECK-NEXT:   ret void

; CHECK: invertloop:                                       ; preds = %entry, %incinvertloop
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ 9, %entry ], [ %5, %incinvertloop ]
; CHECK-NEXT:   %"out'il_phi_unwrap" = load double*, double** %"outp'", align 8
; CHECK-NEXT:   %0 = load double, double* %"out'il_phi_unwrap", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"out'il_phi_unwrap", align 8
; CHECK-NEXT:   %in_unwrap = load double*, double** %inp, align 8
; CHECK-NEXT:   %v_unwrap = load double, double* %in_unwrap, align 8
; CHECK-NEXT:   %m0diffev = fmul fast double %0, %v_unwrap
; CHECK-NEXT:   %m1diffev = fmul fast double %0, %v_unwrap
; CHECK-NEXT:   %1 = fadd fast double %m0diffev, %m1diffev
; CHECK-NEXT:   %"in'il_phi_unwrap" = load double*, double** %"inp'", align 8
; CHECK-NEXT:   %2 = load double, double* %"in'il_phi_unwrap", align 8
; CHECK-NEXT:   %3 = fadd fast double %2, %1
; CHECK-NEXT:   store double %3, double* %"in'il_phi_unwrap", align 8
; CHECK-NEXT:   %4 = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %4, label %invertentry, label %incinvertloop

; CHECK: incinvertloop:                                    ; preds = %invertloop
; CHECK-NEXT:   %5 = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertloop
; CHECK-NEXT: }
