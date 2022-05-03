;RUN: %opt < %s %loadEnzyme -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare dso_local double @__enzyme_fwdsplit(...)

declare double @cblas_ddot(i32, double*, i32, double*, i32)

define double @active(i32 %len, double* noalias %m, double* %dm, i32 %incm, double* noalias %n, double* %dn, i32 %incn) {
entry:
  %r = call double (...) @__enzyme_fwdsplit(double (i32, double*, i32, double*, i32)* @f, i32 %len, double* noalias %m, double* %dm, i32 %incm, double* noalias %n, double* %dn, i32 %incn, i8* null)
  ret double %r
}

define double @inactiveFirst(i32 %len, double* noalias %m, i32 %incm, double* noalias %n, double* %dn, i32 %incn) {
entry:
  %r = call double (...) @__enzyme_fwdsplit(double (i32, double*, i32, double*, i32)* @f, i32 %len, metadata !"enzyme_const", double* noalias %m, i32 %incm, double* noalias %n, double* %dn, i32 %incn, i8* null)
  ret double %r
}

define double @inactiveSecond(i32 %len, double* noalias %m, double* noalias %dm, i32 %incm, double* noalias %n, i32 %incn) {
entry:
  %r = call double (...) @__enzyme_fwdsplit(double (i32, double*, i32, double*, i32)* @f, i32 %len, double* noalias %m, double* noalias %dm, i32 %incm, metadata !"enzyme_const", double* noalias %n, i32 %incn, i8* null)
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

; CHECK: define internal double @[[active]](i32 %len, double* noalias %m, double* %"m'", i32 %incm, double* noalias %n, double* %"n'", i32 %incn, i8* %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = bitcast i8* %tapeArg to { double*, double* }*
; CHECK-NEXT:   %1 = load { double*, double* }, { double*, double* }* %0
; CHECK-NEXT:   tail call void @free(i8* nonnull %tapeArg)
; CHECK-NEXT:   %2 = extractvalue { double*, double* } %1, 0
; CHECK-NEXT:   %3 = extractvalue { double*, double* } %1, 1
; CHECK-NEXT:   %4 = call fast double @cblas_ddot(i32 %len, double* nocapture readonly %2, i32 1, double* nocapture readonly %"n'", i32 %incn)
; CHECK-NEXT:   %5 = call fast double @cblas_ddot(i32 %len, double* nocapture readonly %3, i32 1, double* nocapture readonly %"m'", i32 %incm)
; CHECK-NEXT:   %6 = fadd fast double %4, %5
; CHECK-NEXT:   %7 = bitcast double* %2 to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %7)
; CHECK-NEXT:   %8 = bitcast double* %3 to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %8)
; CHECK-NEXT:   ret double %6
; CHECK-NEXT: }

; CHECK: define internal double @[[inactiveFirst]](i32 %len, double* noalias %m, i32 %incm, double* noalias %n, double* %"n'", i32 %incn, i8* %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = bitcast i8* %tapeArg to double**
; CHECK-NEXT:   %1 = load double*, double** %0
; CHECK-NEXT:   tail call void @free(i8* nonnull %tapeArg)
; CHECK-NEXT:   %2 = call fast double @cblas_ddot(i32 %len, double* nocapture readonly %1, i32 1, double* nocapture readonly %"n'", i32 %incn)
; CHECK-NEXT:   %3 = bitcast double* %1 to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %3)
; CHECK-NEXT:   ret double %2
; CHECK-NEXT: }

; CHECK: define internal double @[[inactiveSecond]](i32 %len, double* noalias %m, double* %"m'", i32 %incm, double* noalias %n, i32 %incn, i8* %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = bitcast i8* %tapeArg to double**
; CHECK-NEXT:   %1 = load double*, double** %0
; CHECK-NEXT:   tail call void @free(i8* nonnull %tapeArg)
; CHECK-NEXT:   %2 = call fast double @cblas_ddot(i32 %len, double* nocapture readonly %1, i32 1, double* nocapture readonly %"m'", i32 %incm)
; CHECK-NEXT:   %3 = bitcast double* %1 to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %3)
; CHECK-NEXT:   ret double %2
; CHECK-NEXT: }
