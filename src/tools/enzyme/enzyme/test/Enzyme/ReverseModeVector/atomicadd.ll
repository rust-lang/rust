; RUN: if [ %llvmver -ge 9 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -instsimplify -simplifycfg -S | FileCheck %s; fi

; ModuleID = '<source>'
source_filename = "<source>"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define dso_local void @foo1(double* %p, double %v) {
  %a10 = atomicrmw volatile fadd double* %p, double %v monotonic
  ret void
}
define dso_local void @foo6(double* %p, double %v) {
  %a10 = atomicrmw volatile fadd double* %p, double 1.000000e+00 seq_cst
  ret void
}

define void @caller(double* %a, double* %b, double %v) {
  %r1 = call [2 x double] (...) @_Z17__enzyme_autodiffPviRdS0_(i8* bitcast (void (double*, double)* @foo1 to i8*), metadata !"enzyme_width", i64 2, double* %a, double* %b, double* %b, double %v)
  %r6 = call [2 x double] (...) @_Z17__enzyme_autodiffPviRdS0_(i8* bitcast (void (double*, double)* @foo6 to i8*), metadata !"enzyme_width", i64 2, double* %a, double* %b, double* %b, double %v)
  ret void
}

declare [2 x double] @_Z17__enzyme_autodiffPviRdS0_(...)

; CHECK: define internal { [2 x double] } @diffe2foo1(double* %p, [2 x double*] %"p'", double %v)
; CHECK-NEXT: invert:
; CHECK-NEXT:   %a10 = atomicrmw volatile fadd double* %p, double %v monotonic
; CHECK-NEXT:   %0 = extractvalue [2 x double*] %"p'", 0
; CHECK-NEXT:   %1 = load atomic volatile double, double* %0 monotonic, align 8
; CHECK-NEXT:   %2 = extractvalue [2 x double*] %"p'", 1
; CHECK-NEXT:   %3 = load atomic volatile double, double* %2 monotonic, align 8
; CHECK-NEXT:   %.fca.0.insert5 = insertvalue [2 x double] {{(undef|poison)}}, double %1, 0
; CHECK-NEXT:   %.fca.1.insert8 = insertvalue [2 x double] %.fca.0.insert5, double %3, 1
; CHECK-NEXT:   %4 = insertvalue { [2 x double] } undef, [2 x double] %.fca.1.insert8, 0
; CHECK-NEXT:   ret { [2 x double] } %4
; CHECK-NEXT: }

; CHECK: define internal { [2 x double] } @diffe2foo6(double* %p, [2 x double*] %"p'", double %v)
; CHECK-NEXT: invert:
; CHECK-NEXT:   %a10 = atomicrmw volatile fadd double* %p, double 1.000000e+00 seq_cst
; CHECK-NEXT:   ret { [2 x double] } zeroinitializer
; CHECK-NEXT: }