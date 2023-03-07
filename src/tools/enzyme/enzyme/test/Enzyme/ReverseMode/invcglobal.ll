; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

@g = constant i8* null, align 8

define void @_Z3barv(void ()** %i2) {
  store void ()* bitcast (i8** @g to void ()*), void ()** %i2, align 8
  ret void
}

define double @_Z3fooRd(double* nocapture readonly %arg) {
  %ai2 = alloca void ()*, align 8
  call void @_Z3barv(void ()** %ai2)
  %a8 = load void ()*, void ()** %ai2, align 8
  ret double 0.000000e+00
}

define void @caller(double* %i, double* %i2) {
  %i6 = call double (...) @__enzyme_autodiff(i8* bitcast (double (double*)* @_Z3fooRd to i8*), double* %i, double* %i2)
  ret void
}

declare double @__enzyme_autodiff(...)

; CHECK: define internal void @diffe_Z3fooRd(double* nocapture readonly %arg, double* nocapture %"arg'", double %differeturn)
; CHECK-NEXT: invert:
; CHECK-NEXT:   %"ai2'ipa" = alloca void ()*, align 8
; CHECK-NEXT:   store void ()* null, void ()** %"ai2'ipa", align 8
; CHECK-NEXT:   %ai2 = alloca void ()*, align 8
; CHECK-NEXT:   call void @diffe_Z3barv(void ()** %ai2, void ()** %"ai2'ipa")
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @diffe_Z3barv(void ()** %i2, void ()** %"i2'")
; CHECK-NEXT: invert:
; CHECK-NEXT:   store void ()* bitcast (i8** @g to void ()*), void ()** %"i2'", align 8
; CHECK-NEXT:   store void ()* bitcast (i8** @g to void ()*), void ()** %i2, align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

