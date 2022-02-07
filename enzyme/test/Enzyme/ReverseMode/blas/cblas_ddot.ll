;RUN: %opt < %s %loadEnzyme -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare dso_local void @__enzyme_autodiff(...)

declare double @cblas_ddot(i32, double*, i32, double*, i32)

define void @active(i32 %len, double* noalias %m, double* %dm, i32 %incm, double* noalias %n, double* %dn, i32 %incn) {
entry:
  call void (...) @__enzyme_autodiff(double (i32, double*, i32, double*, i32)* @f, i32 %len, double* noalias %m, double* %dm, i32 %incm, double* noalias %n, double* %dn, i32 %incn)
  ret void
}

define void @inactiveFirst(i32 %len, double* noalias %m, i32 %incm, double* noalias %n, double* %dn, i32 %incn) {
entry:
  call void (...) @__enzyme_autodiff(double (i32, double*, i32, double*, i32)* @f, i32 %len, metadata !"enzyme_const", double* noalias %m, i32 %incm, double* noalias %n, double* %dn, i32 %incn)
  ret void
}

define void @inactiveSecond(i32 %len, double* noalias %m, double* noalias %dm, i32 %incm, double* noalias %n, i32 %incn) {
entry:
  call void (...) @__enzyme_autodiff(double (i32, double*, i32, double*, i32)* @f, i32 %len, double* noalias %m, double* noalias %dm, i32 %incm, metadata !"enzyme_const", double* noalias %n, i32 %incn)
  ret void
}

define void @activeMod(i32 %len, double* noalias %m, double* %dm, i32 %incm, double* noalias %n, double* %dn, i32 %incn) {
entry:
  call void (...) @__enzyme_autodiff(double (i32, double*, i32, double*, i32)* @modf, i32 %len, double* noalias %m, double* %dm, i32 %incm, double* noalias %n, double* %dn, i32 %incn)
  ret void
}

define void @inactiveModFirst(i32 %len, double* noalias %m, i32 %incm, double* noalias %n, double* %dn, i32 %incn) {
entry:
  call void (...) @__enzyme_autodiff(double (i32, double*, i32, double*, i32)* @modf, i32 %len, metadata !"enzyme_const", double* noalias %m, i32 %incm, double* noalias %n, double* %dn, i32 %incn)
  ret void
}

define void @inactiveModSecond(i32 %len, double* noalias %m, double* noalias %dm, i32 %incm, double* noalias %n, i32 %incn) {
entry:
  call void (...) @__enzyme_autodiff(double (i32, double*, i32, double*, i32)* @modf, i32 %len, double* noalias %m, double* noalias %dm, i32 %incm, metadata !"enzyme_const", double* noalias %n, i32 %incn)
  ret void
}

define double @f(i32 %len, double* noalias %m, i32 %incm, double* noalias %n, i32 %incn) {
entry:
  %call = call double @cblas_ddot(i32 %len, double* %m, i32 %incm, double* %n, i32 %incn)
  ret double %call
}

define double @modf(i32 %len, double* noalias %m, i32 %incm, double* noalias %n, i32 %incn) {
entry:
  %call = call double @f(i32 %len, double* %m, i32 %incm, double* %n, i32 %incn)
  store double 0.000000e+00, double* %m
  store double 0.000000e+00, double* %n
  ret double %call
}


; CHECK: define void @active
; CHECK-NEXT: entry
; CHECK-NEXT: call void @[[active:.+]](

; CHECK: define void @inactiveFirst
; CHECK-NEXT: entry
; CHECK-NEXT: call void @[[inactiveFirst:.+]](

; CHECK: define void @inactiveSecond
; CHECK-NEXT: entry
; CHECK-NEXT: call void @[[inactiveSecond:.+]](


; CHECK: define void @activeMod
; CHECK-NEXT: entry
; CHECK-NEXT: call void @[[activeMod:.+]](

; CHECK: define void @inactiveModFirst
; CHECK-NEXT: entry
; CHECK-NEXT: call void @[[inactiveModFirst:.+]](

; CHECK: define void @inactiveModSecond
; CHECK-NEXT: entry
; CHECK-NEXT: call void @[[inactiveModSecond:.+]](


; CHECK: define internal void @[[active]](i32 %len, double* noalias %m, double* %"m'", i32 %incm, double* noalias %n, double* %"n'", i32 %incn, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %call = call double @cblas_ddot(i32 %len, double* nocapture readonly %m, i32 %incm, double* nocapture readonly %n, i32 %incn)
; CHECK-NEXT:   call void @cblas_daxpy(i32 %len, double %differeturn, double* %m, i32 %incm, double* %"n'", i32 %incn)
; CHECK-NEXT:   call void @cblas_daxpy(i32 %len, double %differeturn, double* %n, i32 %incn, double* %"m'", i32 %incm)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @[[inactiveFirst]](i32 %len, double* noalias %m, i32 %incm, double* noalias %n, double* %"n'", i32 %incn, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %call = call double @cblas_ddot(i32 %len, double* nocapture readonly %m, i32 %incm, double* nocapture readonly %n, i32 %incn)
; CHECK-NEXT:   call void @cblas_daxpy(i32 %len, double %differeturn, double* %m, i32 %incm, double* %"n'", i32 %incn)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @[[inactiveSecond]](i32 %len, double* noalias %m, double* %"m'", i32 %incm, double* noalias %n, i32 %incn, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %call = call double @cblas_ddot(i32 %len, double* nocapture readonly %m, i32 %incm, double* nocapture readonly %n, i32 %incn)
; CHECK-NEXT:   call void @cblas_daxpy(i32 %len, double %differeturn, double* %n, i32 %incn, double* %"m'", i32 %incm)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @[[activeMod]](i32 %len, double* noalias %m, double* %"m'", i32 %incm, double* noalias %n, double* %"n'", i32 %incn, double %differeturn)
; CHECK-NEXT: entry:
; CHECK:        %call_augmented = call { double*, double* } @[[augMod:.+]](i32 %len, double* %m, double* %"m'", i32 %incm, double* %n, double* %"n'", i32 %incn)
; CHECK:        call void @[[revMod:.+]](i32 %len, double* %m, double* %"m'", i32 %incm, double* %n, double* %"n'", i32 %incn, double %differeturn, { double*, double* } %call_augmented)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal { double*, double* } @[[augMod]](i32 %len, double* noalias %m, double* %"m'", i32 %incm, double* noalias %n, double* %"n'", i32 %incn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = zext i32 %len to i64
; CHECK-NEXT:   %mallocsize = mul i64 %0, ptrtoint (double* getelementptr (double, double* null, i32 1) to i64)
; CHECK-NEXT:   %malloccall = tail call i8* @malloc(i64 %mallocsize)
; CHECK-NEXT:   %1 = bitcast i8* %malloccall to double*
; CHECK-NEXT:   call void @__enzyme_memcpy_doubleda0sa0stride(double* %1, double* %m, i32 %len, i32 %incm)
; CHECK-NEXT:   %2 = zext i32 %len to i64
; CHECK-NEXT:   %mallocsize1 = mul i64 %2, ptrtoint (double* getelementptr (double, double* null, i32 1) to i64)
; CHECK-NEXT:   %malloccall2 = tail call i8* @malloc(i64 %mallocsize1)
; CHECK-NEXT:   %3 = bitcast i8* %malloccall2 to double*
; CHECK-NEXT:   call void @__enzyme_memcpy_doubleda0sa0stride(double* %3, double* %n, i32 %len, i32 %incn)
; CHECK-NEXT:   %4 = insertvalue { double*, double* } undef, double* %1, 0
; CHECK-NEXT:   %5 = insertvalue { double*, double* } %4, double* %3, 1
; CHECK-NEXT:   %call = call double @cblas_ddot(i32 %len, double* nocapture readonly %m, i32 %incm, double* nocapture readonly %n, i32 %incn)
; CHECK-NEXT:   ret { double*, double* } %5
; CHECK-NEXT: }

; CHECK: define internal void @[[revMod]](i32 %len, double* noalias %m, double* %"m'", i32 %incm, double* noalias %n, double* %"n'", i32 %incn, double %differeturn, { double*, double* }
; CHECK-NEXT: entry:
; CHECK-NEXT:   %1 = extractvalue { double*, double* } %0, 0
; CHECK-NEXT:   %2 = extractvalue { double*, double* } %0, 1
; CHECK-NEXT:   call void @cblas_daxpy(i32 %len, double %differeturn, double* %1, i32 1, double* %"n'", i32 %incn)
; CHECK-NEXT:   %3 = bitcast double* %1 to i8*
; CHECK-NEXT:   tail call void @free(i8* %3)
; CHECK-NEXT:   call void @cblas_daxpy(i32 %len, double %differeturn, double* %2, i32 1, double* %"m'", i32 %incm)
; CHECK-NEXT:   %4 = bitcast double* %2 to i8*
; CHECK-NEXT:   tail call void @free(i8* %4)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @[[inactiveModFirst]](i32 %len, double* noalias %m, i32 %incm, double* noalias %n, double* %"n'", i32 %incn, double %differeturn)
; CHECK-NEXT: entry:
; CHECK:        %call_augmented = call double* @[[augModFirst:.+]](i32 %len, double* %m, i32 %incm, double* %n, double* %"n'", i32 %incn)
; CHECK:        call void @[[revModFirst:.+]](i32 %len, double* %m, i32 %incm, double* %n, double* %"n'", i32 %incn, double %differeturn, double* %call_augmented)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal double* @[[augModFirst]](i32 %len, double* noalias %m, i32 %incm, double* noalias %n, double* %"n'", i32 %incn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = zext i32 %len to i64
; CHECK-NEXT:   %mallocsize = mul i64 %0, ptrtoint (double* getelementptr (double, double* null, i32 1) to i64)
; CHECK-NEXT:   %malloccall = tail call i8* @malloc(i64 %mallocsize)
; CHECK-NEXT:   %1 = bitcast i8* %malloccall to double*
; CHECK-NEXT:   call void @__enzyme_memcpy_doubleda0sa0stride(double* %1, double* %m, i32 %len, i32 %incm)
; CHECK-NEXT:   %call = call double @cblas_ddot(i32 %len, double* nocapture readonly %m, i32 %incm, double* nocapture readonly %n, i32 %incn)
; CHECK-NEXT:   ret double* %1
; CHECK-NEXT: }

; CHECK: define internal void @[[revModFirst]](i32 %len, double* noalias %m, i32 %incm, double* noalias %n, double* %"n'", i32 %incn, double %differeturn, double*
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @cblas_daxpy(i32 %len, double %differeturn, double* %0, i32 1, double* %"n'", i32 %incn)
; CHECK-NEXT:   %1 = bitcast double* %0 to i8*
; CHECK-NEXT:   tail call void @free(i8* %1)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @[[inactiveModSecond]](i32 %len, double* noalias %m, double* %"m'", i32 %incm, double* noalias %n, i32 %incn, double %differeturn)
; CHECK-NEXT: entry:
; CHECK:        %call_augmented = call double* @[[augModSecond:.+]](i32 %len, double* %m, double* %"m'", i32 %incm, double* %n, i32 %incn)
; CHECK:        call void @[[revModSecond:.+]](i32 %len, double* %m, double* %"m'", i32 %incm, double* %n, i32 %incn, double %differeturn, double* %call_augmented)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal double* @[[augModSecond]](i32 %len, double* noalias %m, double* %"m'", i32 %incm, double* noalias %n, i32 %incn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = zext i32 %len to i64
; CHECK-NEXT:   %mallocsize = mul i64 %0, ptrtoint (double* getelementptr (double, double* null, i32 1) to i64)
; CHECK-NEXT:   %malloccall = tail call i8* @malloc(i64 %mallocsize)
; CHECK-NEXT:   %1 = bitcast i8* %malloccall to double*
; CHECK-NEXT:   call void @__enzyme_memcpy_doubleda0sa0stride(double* %1, double* %n, i32 %len, i32 %incn)
; CHECK-NEXT:   %call = call double @cblas_ddot(i32 %len, double* nocapture readonly %m, i32 %incm, double* nocapture readonly %n, i32 %incn)
; CHECK-NEXT:   ret double* %1
; CHECK-NEXT: }

; CHECK: define internal void @[[revModSecond]](i32 %len, double* noalias %m, double* %"m'", i32 %incm, double* noalias %n, i32 %incn, double %differeturn, double*
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @cblas_daxpy(i32 %len, double %differeturn, double* %0, i32 1, double* %"m'", i32 %incm)
; CHECK-NEXT:   %1 = bitcast double* %0 to i8*
; CHECK-NEXT:   tail call void @free(i8* %1)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

