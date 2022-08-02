; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -simplifycfg -S | FileCheck %s --check-prefixes SHARED,MEMSET
; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -simplifycfg -S | FileCheck %s --check-prefixes SHARED,STORE


declare void @__enzyme_autodiff(i8*, double*, double*, double*, double*)

declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1)

define void @f(double* %x, double* %y) {
  %yptr = bitcast double* %y to i8*
  call void @llvm.memset.p0i8.i64(i8* %yptr, i8 0, i64 8, i1 false)
  %x1 = load double, double* %x, align 8
  %y1 = load double, double* %y, align 8
  %mul = fmul double %x1, %y1
  store double %mul, double* %x, align 8
  ret void
}

define void @g(double* %x, double* %y) {
  %yptr = bitcast double* %y to i8*
  store double 0.0, double* %y, align 8
  %x1 = load double, double* %x, align 8
  %y1 = load double, double* %y, align 8
  %mul = fmul double %x1, %y1
  store double %mul, double* %x, align 8
  ret void
}

define void @df(double* %x, double* %xp, double* %y, double* %dy) {
  tail call void @__enzyme_autodiff(i8* bitcast (void (double*, double*)* @f to i8*), double* %x, double* %xp, double* %y, double* %dy)
  tail call void @__enzyme_autodiff(i8* bitcast (void (double*, double*)* @g to i8*), double* %x, double* %xp, double* %y, double* %dy)
  ret void
}


; MEMSET: define internal void @diffef(double* %x, double* %"x'", double* %y, double* %"y'")
; STORE:  define internal void @diffeg(double* %x, double* %"x'", double* %y, double* %"y'")
; SHARED-NEXT: invert:
; MEMSET-NEXT:   %"yptr'ipc" = bitcast double* %"y'" to i8*
; MEMSET-NEXT:   %yptr = bitcast double* %y to i8*
; MEMSET-NEXT:   call void @llvm.memset.p0i8.i64(i8* %yptr, i8 0, i64 8, i1 false)
; STORE-NEXT:    store double 0.000000e+00, double* %y, align 8
; SHARED-NEXT:   %x1 = load double, double* %x, align 8
; SHARED-NEXT:   %y1 = load double, double* %y, align 8
; SHARED-NEXT:   %mul = fmul double %x1, %y1
; SHARED-NEXT:   store double %mul, double* %x, align 8
; SHARED-NEXT:   %0 = load double, double* %"x'", align 8
; SHARED-NEXT:   store double 0.000000e+00, double* %"x'", align 8
; SHARED-NEXT:   %1 = fadd fast double 0.000000e+00, %0
; SHARED-NEXT:   %m0diffex1 = fmul fast double %1, %y1
; SHARED-NEXT:   %m1diffey1 = fmul fast double %1, %x1
; SHARED-NEXT:   %2 = fadd fast double 0.000000e+00, %m0diffex1
; SHARED-NEXT:   %3 = fadd fast double 0.000000e+00, %m1diffey1
; SHARED-NEXT:   %4 = load double, double* %"y'", align 8
; SHARED-NEXT:   %5 = fadd fast double %4, %3
; SHARED-NEXT:   store double %5, double* %"y'", align 8
; SHARED-NEXT:   %6 = load double, double* %"x'", align 8
; SHARED-NEXT:   %7 = fadd fast double %6, %2
; SHARED-NEXT:   store double %7, double* %"x'", align 8
; MEMSET-NEXT:   call void @llvm.memset.p0i8.i64(i8* %"yptr'ipc", i8 0, i64 8, i1 false)
; STORE-NEXT:    store double 0.000000e+00, double* %"y'", align 8
; SHARED-NEXT:   ret void
; SHARED-NEXT: }