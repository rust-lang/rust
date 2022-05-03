; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -instsimplify -simplifycfg -S | FileCheck %s

declare void @__enzyme_autodiff(i8*, ...)

define void @derivative(double* %mat, double* %dmat) {
entry:
  call void (i8*, ...) @__enzyme_autodiff(i8* bitcast (void (double*)* @called to i8*), metadata !"enzyme_dup", double* %mat, double* %dmat)
  ret void
}

define void @called(double* %mat1) {
entry:
  %all = alloca double*
  %call17 = call double* @callrt(double* %mat1)
  store double* %call17, double** %all
  %er = bitcast double** %all to i8*
  ret void
}

define double* @callrt(double* %c) {
  %call17 = call double* @callrt2(double* %c)
  ret double* %call17
}

define double* @callrt2(double* %c) {
  %call17 = call double* @sqret(double* %c)
  ret double* %call17
}

define double* @sqret(double* %x) {
entry:
    %z = load double, double* %x
    %z2 = fmul double %z, %z
    store double %z2, double* %x
    ret double* %x
}

; CHECK: define internal void @diffesqret(double* %x, double* %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %z = load double, double* %x
; CHECK-NEXT:   %z2 = fmul double %z, %z
; CHECK-NEXT:   store double %z2, double* %x
; CHECK-NEXT:   %0 = load double, double* %"x'"
; CHECK-NEXT:   store double 0.000000e+00, double* %"x'"
; CHECK-NEXT:   %m0diffez = fmul fast double %0, %z
; CHECK-NEXT:   %m1diffez = fmul fast double %0, %z
; CHECK-NEXT:   %1 = fadd fast double %m0diffez, %m1diffez
; CHECK-NEXT:   %2 = load double, double* %"x'"
; CHECK-NEXT:   %3 = fadd fast double %2, %1
; CHECK-NEXT:   store double %3, double* %"x'"
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
