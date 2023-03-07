; RUN: %opt < %s %loadEnzyme -enzyme -S | FileCheck %s
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare dso_local double @__enzyme_fwddiff(...)

declare double @cblas_ddot(i32, double*, i32, double*, i32)

define double @active(i32 %len, double* noalias %m, double* %dm, i32 %incm, double* noalias %n, double* %dn, i32 %incn) {
entry:
  %r = call double (...) @__enzyme_fwddiff(double (i32, double*, i32, double*, i32)* @f, i32 %len, double* noalias %m, double* %dm, i32 %incm, double* noalias %n, double* %dn, i32 %incn)
  ret double %r
}

define double @inactiveFirst(i32 %len, double* noalias %m, i32 %incm, double* noalias %n, double* %dn, i32 %incn) {
entry:
  %r = call double (...) @__enzyme_fwddiff(double (i32, double*, i32, double*, i32)* @f, i32 %len, metadata !"enzyme_const", double* noalias %m, i32 %incm, double* noalias %n, double* %dn, i32 %incn)
  ret double %r
}

define double @inactiveSecond(i32 %len, double* noalias %m, double* noalias %dm, i32 %incm, double* noalias %n, i32 %incn) {
entry:
  %r = call double (...) @__enzyme_fwddiff(double (i32, double*, i32, double*, i32)* @f, i32 %len, double* noalias %m, double* noalias %dm, i32 %incm, metadata !"enzyme_const", double* noalias %n, i32 %incn)
  ret double %r
}

define double @f(i32 %len, double* noalias %m, i32 %incm, double* noalias %n, i32 %incn) {
entry:
  %call = call double @cblas_ddot(i32 %len, double* %m, i32 %incm, double* %n, i32 %incn)
  ret double %call
}

; CHECK: define double @active
; CHECK-NEXT: entry
; CHECK-NEXT: call fast double @[[active:.+]](

; CHECK: define double @inactiveFirst
; CHECK-NEXT: entry
; CHECK-NEXT: call fast double @[[inactiveFirst:.+]](

; CHECK: define double @inactiveSecond
; CHECK-NEXT: entry
; CHECK-NEXT: call fast double @[[inactiveSecond:.+]](

; CHECK: define internal double @[[active]](i32 %len, double* noalias %m, double* %"m'", i32 %incm, double* noalias %n, double* %"n'", i32 %incn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call fast double @cblas_ddot(i32 %len, double* nocapture readonly %m, i32 %incm, double* nocapture readonly %"n'", i32 %incn)
; CHECK-NEXT:   %1 = call fast double @cblas_ddot(i32 %len, double* nocapture readonly %n, i32 %incn, double* nocapture readonly %"m'", i32 %incm)
; CHECK-NEXT:   %2 = fadd fast double %0, %1
; CHECK-NEXT:   ret double %2
; CHECK-NEXT: }

; CHECK: define internal double @[[inactiveFirst]](i32 %len, double* noalias %m, i32 %incm, double* noalias %n, double* %"n'", i32 %incn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call fast double @cblas_ddot(i32 %len, double* nocapture readonly %m, i32 %incm, double* nocapture readonly %"n'", i32 %incn)
; CHECK-NEXT:   ret double %0
; CHECK-NEXT: }

; CHECK: define internal double @[[inactiveSecond]](i32 %len, double* noalias %m, double* %"m'", i32 %incm, double* noalias %n, i32 %incn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call fast double @cblas_ddot(i32 %len, double* nocapture readonly %n, i32 %incn, double* nocapture readonly %"m'", i32 %incm)
; CHECK-NEXT:   ret double %0
; CHECK-NEXT: }
