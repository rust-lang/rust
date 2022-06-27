; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -S | FileCheck %s

declare [4 x double] @__enzyme_batch(...)

define double @add3(double %x, double %a) {
entry:
  %add = fadd double %x, %a
  ret double %add
}

define double @square(double %x, double %a) {
entry:
  %call = call double @add3(double %a, double 3.0)
  %mul = fmul double %x, %call
  ret double %mul
}

define [4 x double] @dsquare(double %x1, double %x2, double %x3, double %x4, double %a) {
entry:
  %call = call [4 x double] (...) @__enzyme_batch(double (double, double)* @square, metadata !"enzyme_width", i64 4, metadata !"enzyme_vector", double %x1, double %x2, double %x3, double %x4, metadata !"enzyme_scalar", double %a)
  ret [4 x double] %call
}


; CHECK: define internal [4 x double] @batch_square([4 x double] %x, double %a)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %unwrap.x0 = extractvalue [4 x double] %x, 0
; CHECK-NEXT:   %unwrap.x1 = extractvalue [4 x double] %x, 1
; CHECK-NEXT:   %unwrap.x2 = extractvalue [4 x double] %x, 2
; CHECK-NEXT:   %unwrap.x3 = extractvalue [4 x double] %x, 3
; CHECK-NEXT:   %call = call [4 x double] @batch_add3(double %a, double 3.000000e+00)
; CHECK-NEXT:   %unwrap.call0 = extractvalue [4 x double] %call, 0
; CHECK-NEXT:   %unwrap.call1 = extractvalue [4 x double] %call, 1
; CHECK-NEXT:   %unwrap.call2 = extractvalue [4 x double] %call, 2
; CHECK-NEXT:   %unwrap.call3 = extractvalue [4 x double] %call, 3
; CHECK-NEXT:   %mul0 = fmul double %unwrap.x0, %unwrap.call0
; CHECK-NEXT:   %mul1 = fmul double %unwrap.x1, %unwrap.call1
; CHECK-NEXT:   %mul2 = fmul double %unwrap.x2, %unwrap.call2
; CHECK-NEXT:   %mul3 = fmul double %unwrap.x3, %unwrap.call3
; CHECK-NEXT:   %mrv = insertvalue [4 x double] undef, double %mul0, 0
; CHECK-NEXT:   %mrv1 = insertvalue [4 x double] %mrv, double %mul1, 1
; CHECK-NEXT:   %mrv2 = insertvalue [4 x double] %mrv1, double %mul2, 2
; CHECK-NEXT:   %mrv3 = insertvalue [4 x double] %mrv2, double %mul3, 3
; CHECK-NEXT:   ret [4 x double] %mrv3
; CHECK-NEXT: }

; CHECK: define internal [4 x double] @batch_add3(double %x, double %a)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %add = fadd double %x, %a
; CHECK-NEXT:   %mrv = insertvalue [4 x double] undef, double %add, 0
; CHECK-NEXT:   %mrv1 = insertvalue [4 x double] %mrv, double %add, 1
; CHECK-NEXT:   %mrv2 = insertvalue [4 x double] %mrv1, double %add, 2
; CHECK-NEXT:   %mrv3 = insertvalue [4 x double] %mrv2, double %add, 3
; CHECK-NEXT:   ret [4 x double] %mrv3
; CHECK-NEXT: }