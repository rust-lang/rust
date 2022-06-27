; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -S | FileCheck %s

declare [4 x double] @__enzyme_batch(...)

define double @square_add(double %x, double %y) {
entry:
  %mul = fmul double %x, %x
  %add = fadd double %mul, %y
  ret double %add
}

define [4 x double] @dsquare(double %x1, double %x2, double %x3, double %x4, double %y) {
entry:
  %call = call [4 x double] (...) @__enzyme_batch(double (double, double)* @square_add, metadata !"enzyme_width", i64 4, metadata !"enzyme_vector", double %x1, double %x2, double %x3, double %x4, metadata !"enzyme_scalar", double %y)
  ret [4 x double] %call
}


; CHECK: define internal [4 x double] @batch_square_add([4 x double] %x, double %y)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %unwrap.x0 = extractvalue [4 x double] %x, 0
; CHECK-NEXT:   %unwrap.x1 = extractvalue [4 x double] %x, 1
; CHECK-NEXT:   %unwrap.x2 = extractvalue [4 x double] %x, 2
; CHECK-NEXT:   %unwrap.x3 = extractvalue [4 x double] %x, 3
; CHECK-NEXT:   %mul0 = fmul double %unwrap.x0, %unwrap.x0
; CHECK-NEXT:   %mul1 = fmul double %unwrap.x1, %unwrap.x1
; CHECK-NEXT:   %mul2 = fmul double %unwrap.x2, %unwrap.x2
; CHECK-NEXT:   %mul3 = fmul double %unwrap.x3, %unwrap.x3
; CHECK-NEXT:   %add = fadd double %mul0, %y
; CHECK-NEXT:   %mrv = insertvalue [4 x double] undef, double %add, 0
; CHECK-NEXT:   %mrv1 = insertvalue [4 x double] %mrv, double %add, 1
; CHECK-NEXT:   %mrv2 = insertvalue [4 x double] %mrv1, double %add, 2
; CHECK-NEXT:   %mrv3 = insertvalue [4 x double] %mrv2, double %add, 3
; CHECK-NEXT:   ret [4 x double] %mrv3
; CHECK-NEXT: }