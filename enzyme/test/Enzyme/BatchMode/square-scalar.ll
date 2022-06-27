; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -S | FileCheck %s

declare [4 x double] @__enzyme_batch(...)

define double @square(double %x) {
entry:
  %mul = fmul double %x, %x
  ret double %mul
}

define [4 x double] @dsquare(double %x1) {
entry:
  %call = call [4 x double] (...) @__enzyme_batch(double (double)* @square, metadata !"enzyme_width", i64 4, metadata !"enzyme_scalar", double %x1)
  ret [4 x double] %call
}


; CHECK: define internal [4 x double] @batch_square(double %x)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %mul = fmul double %x, %x
; CHECK-NEXT:   %mrv = insertvalue [4 x double] undef, double %mul, 0
; CHECK-NEXT:   %mrv1 = insertvalue [4 x double] %mrv, double %mul, 1
; CHECK-NEXT:   %mrv2 = insertvalue [4 x double] %mrv1, double %mul, 2
; CHECK-NEXT:   %mrv3 = insertvalue [4 x double] %mrv2, double %mul, 3
; CHECK-NEXT:   ret [4 x double] %mrv3
; CHECK-NEXT: }