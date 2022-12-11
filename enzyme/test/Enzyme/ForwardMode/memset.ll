; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

declare void @__enzyme_fwddiff(i8*, double*, double*, double*, double*)

declare void @llvm.memset.p0i8.i64(i8*, i8, i64, i1)

define void @f(double* %x, double* %y) {
entry:
  %x1 = load double, double* %x
  %yptr = bitcast double* %y to i8*  
  call void @llvm.memset.p0i8.i64(i8* %yptr, i8 0, i64 8, i1 false)
  %y1 = load double, double* %y
  %x2 = fmul double %x1, %y1
  store double %x2, double* %x
  store double %x2, double* %y
  call void @llvm.memset.p0i8.i64(i8* %yptr, i8 0, i64 8, i1 false)
  ret void
}

define void @df(double* %x, double* %xp, double* %y, double* %dy) {
entry:
  tail call void @__enzyme_fwddiff(i8* bitcast (void (double*, double*)* @f to i8*), double* %x, double* %xp, double* %y, double* %dy)
  ret void
}


; CHECK: define internal void @fwddiffef(double* %x, double* %"x'", double* %y, double* %"y'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[i0:.+]] = load double, double* %"x'"
; CHECK-NEXT:   %x1 = load double, double* %x
; CHECK-NEXT:   %"yptr'ipc" = bitcast double* %"y'" to i8*
; CHECK-NEXT:   %yptr = bitcast double* %y to i8*
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* %yptr, i8 0, i64 8, i1 false)
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* %"yptr'ipc", i8 0, i64 8, i1 false)
; CHECK-NEXT:   %[[i1:.+]] = load double, double* %"y'"
; CHECK-NEXT:   %y1 = load double, double* %y
; CHECK-NEXT:   %x2 = fmul double %x1, %y1
; CHECK-NEXT:   %[[i2:.+]] = fmul fast double %[[i0]], %y1
; CHECK-NEXT:   %[[i3:.+]] = fmul fast double %[[i1]], %x1
; CHECK-NEXT:   %[[i4:.+]] = fadd fast double %[[i2]], %[[i3]]
; CHECK-NEXT:   store double %x2, double* %x
; CHECK-NEXT:   store double %[[i4]], double* %"x'"
; CHECK-NEXT:   store double %x2, double* %y
; CHECK-NEXT:   store double %[[i4]], double* %"y'"
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* %yptr, i8 0, i64 8, i1 false)
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* %"yptr'ipc", i8 0, i64 8, i1 false)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
