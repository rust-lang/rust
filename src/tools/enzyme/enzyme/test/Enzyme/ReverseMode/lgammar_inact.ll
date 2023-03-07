; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

declare double @lgamma_r(double, i32* writeonly nocapture) 

define double @g(double %x)  {
  %a = alloca i32, align 4
  %p = ptrtoint i32* %a to i64
  %r = call double @lgamma_r(double 1.0, i32* %a) 
  ret double %r
}

define double @f(double %x)  {
  %r = call double @g(double %x)
  %m = fmul double %r, %r
  ret double %m
}

declare double @__enzyme_autodiff(i8*, double)

define void @_Z18wrapper_1body_intsv()  {
  %a = call double @__enzyme_autodiff(i8* bitcast (double (double)* @f to i8*), double 2.0)
  ret void
}

; CHECK: define internal double @augmented_g(double %x)
; CHECK-NEXT:   %a = alloca i32, i64 1, align 4
; CHECK-NEXT:   %r = call double @lgamma_r(double 1.000000e+00, i32* %a)
; CHECK-NEXT:   ret double %r
; CHECK-NEXT: }

; CHECK: define internal { double } @diffeg(double %x, double %differeturn)
; CHECK-NEXT: invert:
; CHECK-NEXT:   ret { double } zeroinitializer
; CHECK-NEXT: }
