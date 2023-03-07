;RUN: %opt < %s %loadEnzyme -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare dso_local [2 x double] @__enzyme_fwddiff(...)

declare double @cblas_ddot(i32, double*, i32, double*, i32)

define [2 x double] @active(i32 %len, double* noalias %m, double* %dm, i32 %incm, double* noalias %n, double* %dn, i32 %incn) {
entry:
  %r = call [2 x double] (...) @__enzyme_fwddiff(double (i32, double*, i32, double*, i32)* @f, metadata !"enzyme_width", i64 2, i32 %len, double* noalias %m, double* %dm, double* %dm, i32 %incm, double* noalias %n, double* %dn, double* %dn, i32 %incn)
  ret [2 x double] %r
}

define [2 x double] @inactiveFirst(i32 %len, double* noalias %m, i32 %incm, double* noalias %n, double* %dn, i32 %incn) {
entry:
  %r = call [2 x double] (...) @__enzyme_fwddiff(double (i32, double*, i32, double*, i32)* @f, metadata !"enzyme_width", i64 2, i32 %len, metadata !"enzyme_const", double* noalias %m, i32 %incm, double* noalias %n, double* %dn, double* %dn, i32 %incn)
  ret [2 x double] %r
}

define [2 x double] @inactiveSecond(i32 %len, double* noalias %m, double* noalias %dm, i32 %incm, double* noalias %n, i32 %incn) {
entry:
  %r = call [2 x double] (...) @__enzyme_fwddiff(double (i32, double*, i32, double*, i32)* @f, metadata !"enzyme_width", i64 2, i32 %len, double* noalias %m, double* noalias %dm, double* %dm, i32 %incm, metadata !"enzyme_const", double* noalias %n, i32 %incn)
  ret [2 x double] %r
}

define double @f(i32 %len, double* noalias %m, i32 %incm, double* noalias %n, i32 %incn) {
entry:
  %call = call double @cblas_ddot(i32 %len, double* %m, i32 %incm, double* %n, i32 %incn)
  ret double %call
}

; CHECK: define [2 x double] @active
; CHECK-NEXT: entry
; CHECK: call {{(fast )?}}[2 x double] @[[active:.+]](

; CHECK: define [2 x double] @inactiveFirst
; CHECK-NEXT: entry
; CHECK: call {{(fast )?}}[2 x double] @[[inactiveFirst:.+]](

; CHECK: define [2 x double] @inactiveSecond
; CHECK-NEXT: entry
; CHECK: call {{(fast )?}}[2 x double] @[[inactiveSecond:.+]](

; CHECK: define internal [2 x double] @[[active]](i32 %len, double* noalias %m, [2 x double*] %"m'", i32 %incm, double* noalias %n, [2 x double*] %"n'", i32 %incn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = extractvalue [2 x double*] %"m'", 0
; CHECK-NEXT:   %1 = extractvalue [2 x double*] %"n'", 0
; CHECK-NEXT:   %2 = call fast double @cblas_ddot(i32 %len, double* nocapture readonly %m, i32 %incm, double* nocapture readonly %1, i32 %incn)
; CHECK-NEXT:   %3 = call fast double @cblas_ddot(i32 %len, double* nocapture readonly %n, i32 %incn, double* nocapture readonly %0, i32 %incm)
; CHECK-NEXT:   %4 = fadd fast double %2, %3
; CHECK-NEXT:   %5 = insertvalue [2 x double] undef, double %4, 0
; CHECK-NEXT:   %6 = extractvalue [2 x double*] %"m'", 1
; CHECK-NEXT:   %7 = extractvalue [2 x double*] %"n'", 1
; CHECK-NEXT:   %8 = call fast double @cblas_ddot(i32 %len, double* nocapture readonly %m, i32 %incm, double* nocapture readonly %7, i32 %incn)
; CHECK-NEXT:   %9 = call fast double @cblas_ddot(i32 %len, double* nocapture readonly %n, i32 %incn, double* nocapture readonly %6, i32 %incm)
; CHECK-NEXT:   %10 = fadd fast double %8, %9
; CHECK-NEXT:   %11 = insertvalue [2 x double] %5, double %10, 1
; CHECK-NEXT:   ret [2 x double] %11
; CHECK-NEXT: }

; CHECK: define internal [2 x double] @[[inactiveFirst]](i32 %len, double* noalias %m, i32 %incm, double* noalias %n, [2 x double*] %"n'", i32 %incn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = extractvalue [2 x double*] %"n'", 0
; CHECK-NEXT:   %1 = call fast double @cblas_ddot(i32 %len, double* nocapture readonly %m, i32 %incm, double* nocapture readonly %0, i32 %incn)
; CHECK-NEXT:   %2 = insertvalue [2 x double] undef, double %1, 0
; CHECK-NEXT:   %3 = extractvalue [2 x double*] %"n'", 1
; CHECK-NEXT:   %4 = call fast double @cblas_ddot(i32 %len, double* nocapture readonly %m, i32 %incm, double* nocapture readonly %3, i32 %incn)
; CHECK-NEXT:   %5 = insertvalue [2 x double] %2, double %4, 1
; CHECK-NEXT:   ret [2 x double] %5
; CHECK-NEXT: }

; CHECK: define internal [2 x double] @[[inactiveSecond]](i32 %len, double* noalias %m, [2 x double*] %"m'", i32 %incm, double* noalias %n, i32 %incn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = extractvalue [2 x double*] %"m'", 0
; CHECK-NEXT:   %1 = call fast double @cblas_ddot(i32 %len, double* nocapture readonly %n, i32 %incn, double* nocapture readonly %0, i32 %incm)
; CHECK-NEXT:   %2 = insertvalue [2 x double] undef, double %1, 0
; CHECK-NEXT:   %3 = extractvalue [2 x double*] %"m'", 1
; CHECK-NEXT:   %4 = call fast double @cblas_ddot(i32 %len, double* nocapture readonly %n, i32 %incn, double* nocapture readonly %3, i32 %incm)
; CHECK-NEXT:   %5 = insertvalue [2 x double] %2, double %4, 1
; CHECK-NEXT:   ret [2 x double] %5
; CHECK-NEXT: }
