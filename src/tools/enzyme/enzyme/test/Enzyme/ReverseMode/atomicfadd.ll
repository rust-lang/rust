; RUN: if [ %llvmver -ge 9 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -instsimplify -simplifycfg -S | FileCheck %s; fi

; ModuleID = '<source>'
source_filename = "<source>"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define dso_local void @foo1(double* %p, double %v) {
  %a10 = atomicrmw volatile fadd double* %p, double %v monotonic
  ret void
}
define dso_local void @foo2(double* %p, double %v) {
  %a10 = atomicrmw volatile fadd double* %p, double %v acquire
  ret void
}
define dso_local void @foo3(double* %p, double %v) {
  %a10 = atomicrmw volatile fadd double* %p, double %v release
  ret void
}
define dso_local void @foo4(double* %p, double %v) {
  %a10 = atomicrmw volatile fadd double* %p, double %v acq_rel
  ret void
}
define dso_local void @foo5(double* %p, double %v) {
  %a10 = atomicrmw volatile fadd double* %p, double %v seq_cst
  ret void
}
define dso_local void @foo6(double* %p, double %v) {
  %a10 = atomicrmw volatile fadd double* %p, double 1.000000e+00 seq_cst
  ret void
}

define void @caller(double* %a, double* %b, double %v) {
  %r1 = call double @_Z17__enzyme_autodiffPviRdS0_(i8* bitcast (void (double*, double)* @foo1 to i8*), double* %a, double* %b, double %v)
  %r2 = call double @_Z17__enzyme_autodiffPviRdS0_(i8* bitcast (void (double*, double)* @foo2 to i8*), double* %a, double* %b, double %v)
  %r3 = call double @_Z17__enzyme_autodiffPviRdS0_(i8* bitcast (void (double*, double)* @foo3 to i8*), double* %a, double* %b, double %v)
  %r4 = call double @_Z17__enzyme_autodiffPviRdS0_(i8* bitcast (void (double*, double)* @foo4 to i8*), double* %a, double* %b, double %v)
  %r5 = call double @_Z17__enzyme_autodiffPviRdS0_(i8* bitcast (void (double*, double)* @foo5 to i8*), double* %a, double* %b, double %v)
  %r6 = call double @_Z17__enzyme_autodiffPviRdS0_(i8* bitcast (void (double*, double)* @foo6 to i8*), double* %a, double* %b, double %v)
  ret void
}

declare double @_Z17__enzyme_autodiffPviRdS0_(i8*, double*, double*, double)


; CHECK: define internal { double } @diffefoo1(double* %p, double* %"p'", double %v)
; CHECK-NEXT: invert:
; CHECK-NEXT:   %a10 = atomicrmw volatile fadd double* %p, double %v monotonic
; CHECK-NEXT:   %0 = load atomic volatile double, double* %"p'" monotonic, align 8
; CHECK-NEXT:   %1 = insertvalue { double } {{(undef|poison)}}, double %0, 0
; CHECK-NEXT:   ret { double } %1
; CHECK-NEXT: }

; CHECK: define internal { double } @diffefoo2(double* %p, double* %"p'", double %v)
; CHECK-NEXT: invert:
; CHECK-NEXT:   %a10 = atomicrmw volatile fadd double* %p, double %v acquire
; CHECK-NEXT:   %0 = load atomic volatile double, double* %"p'" acquire, align 8
; CHECK-NEXT:   %1 = insertvalue { double } {{(undef|poison)}}, double %0, 0
; CHECK-NEXT:   ret { double } %1
; CHECK-NEXT: }

; CHECK: define internal { double } @diffefoo3(double* %p, double* %"p'", double %v)
; CHECK-NEXT: invert:
; CHECK-NEXT:   %a10 = atomicrmw volatile fadd double* %p, double %v release
; CHECK-NEXT:   %0 = load atomic volatile double, double* %"p'" monotonic, align 8
; CHECK-NEXT:   %1 = insertvalue { double } {{(undef|poison)}}, double %0, 0
; CHECK-NEXT:   ret { double } %1
; CHECK-NEXT: }

; CHECK: define internal { double } @diffefoo4(double* %p, double* %"p'", double %v)
; CHECK-NEXT: invert:
; CHECK-NEXT:   %a10 = atomicrmw volatile fadd double* %p, double %v acq_rel
; CHECK-NEXT:   %0 = load atomic volatile double, double* %"p'" acquire, align 8
; CHECK-NEXT:   %1 = insertvalue { double } {{(undef|poison)}}, double %0, 0
; CHECK-NEXT:   ret { double } %1
; CHECK-NEXT: }

; CHECK: define internal { double } @diffefoo5(double* %p, double* %"p'", double %v)
; CHECK-NEXT: invert:
; CHECK-NEXT:   %a10 = atomicrmw volatile fadd double* %p, double %v seq_cst
; CHECK-NEXT:   %0 = load atomic volatile double, double* %"p'" seq_cst, align 8
; CHECK-NEXT:   %1 = insertvalue { double } {{(undef|poison)}}, double %0, 0
; CHECK-NEXT:   ret { double } %1
; CHECK-NEXT: }

; CHECK: define internal { double } @diffefoo6(double* %p, double* %"p'", double %v)
; CHECK-NEXT: invert:
; CHECK-NEXT:   %a10 = atomicrmw volatile fadd double* %p, double 1.000000e+00 seq_cst
; CHECK-NEXT:   ret { double } zeroinitializer
; CHECK-NEXT: }
