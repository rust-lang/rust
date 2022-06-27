; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -S | FileCheck %s

define double @relu(double %x, double %a) {
entry:
  %cmp = fcmp fast ogt double %x, 0.000000e+00
  br i1 %cmp, label %cond.true, label %cond.end

cond.true:                                        ; preds = %entry
  %ax = fmul double %x, %a
  ret double %ax

cond.end:                                         ; preds = %entry, %cond.true
  ret double %x
}

define [4 x double] @vecrelu(double %x, double %a1, double %a2, double %a3, double %a4) {
entry:
  %0 = tail call [4 x double] (...) @__enzyme_batch(double (double, double)* nonnull @relu, metadata !"enzyme_width", i64 4, metadata !"enzyme_scalar", double %x, metadata !"enzyme_vector", double %a1, double %a2, double %a3, double %a4)
  ret [4 x double] %0
}

declare [4 x double] @__enzyme_batch(...)


; CHECK: define internal [4 x double] @batch_relu(double %x, [4 x double] %a)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %unwrap.a0 = extractvalue [4 x double] %a, 0
; CHECK-NEXT:   %unwrap.a1 = extractvalue [4 x double] %a, 1
; CHECK-NEXT:   %unwrap.a2 = extractvalue [4 x double] %a, 2
; CHECK-NEXT:   %unwrap.a3 = extractvalue [4 x double] %a, 3
; CHECK-NEXT:   %cmp = fcmp fast ogt double %x, 0.000000e+00
; CHECK-NEXT:   br i1 %cmp, label %cond.true, label %cond.end

; CHECK: cond.true:                                        ; preds = %entry
; CHECK-NEXT:   %ax0 = fmul double %x, %unwrap.a0
; CHECK-NEXT:   %ax1 = fmul double %x, %unwrap.a1
; CHECK-NEXT:   %ax2 = fmul double %x, %unwrap.a2
; CHECK-NEXT:   %ax3 = fmul double %x, %unwrap.a3
; CHECK-NEXT:   %mrv = insertvalue [4 x double] undef, double %ax0, 0
; CHECK-NEXT:   %mrv1 = insertvalue [4 x double] %mrv, double %ax1, 1
; CHECK-NEXT:   %mrv2 = insertvalue [4 x double] %mrv1, double %ax2, 2
; CHECK-NEXT:   %mrv3 = insertvalue [4 x double] %mrv2, double %ax3, 3
; CHECK-NEXT:   ret [4 x double] %mrv3

; CHECK: cond.end:                                         ; preds = %entry
; CHECK-NEXT:   %mrv4 = insertvalue [4 x double] undef, double %x, 0
; CHECK-NEXT:   %mrv5 = insertvalue [4 x double] %mrv4, double %x, 1
; CHECK-NEXT:   %mrv6 = insertvalue [4 x double] %mrv5, double %x, 2
; CHECK-NEXT:   %mrv7 = insertvalue [4 x double] %mrv6, double %x, 3
; CHECK-NEXT:   ret [4 x double] %mrv7
; CHECK-NEXT: }