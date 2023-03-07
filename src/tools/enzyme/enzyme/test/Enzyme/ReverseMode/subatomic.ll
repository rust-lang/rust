; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -simplifycfg -S | FileCheck %s

; ModuleID = '<source>'
source_filename = "<source>"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define internal i32 @_ZN9__gnu_cxxL18__exchange_and_addEPVii(i32* %a0, i32 %a1) {
  %a10 = atomicrmw volatile add i32* %a0, i32 %a1 acq_rel
  ret i32 %a10
}

define dso_local double @_Z3fooRd() {
  %ap = alloca i32, align 8
  %a5 = call i32 @_ZN9__gnu_cxxL18__exchange_and_addEPVii(i32* %ap, i32 -1)
  %a7 = sitofp i32 %a5 to double
  ret double %a7
}

define void @caller(double* %a, double* %b) {
  %r = call double @_Z17__enzyme_autodiffPviRdS0_(i8* bitcast (double ()* @_Z3fooRd to i8*))
  ret void
}

declare double @_Z17__enzyme_autodiffPviRdS0_(i8*)

; CHECK: define internal void @diffe_Z3fooRd(double %differeturn)
; CHECK-NEXT: invert:
; CHECK-NEXT:   %"ap'ipa" = alloca i32, align 8
; CHECK-NEXT:   store i32 0, i32* %"ap'ipa", align 8
; CHECK-NEXT:   %ap = alloca i32, align 8
; CHECK-NEXT:   call void @diffe_ZN9__gnu_cxxL18__exchange_and_addEPVii(i32* %ap, i32* %"ap'ipa", i32 -1)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @diffe_ZN9__gnu_cxxL18__exchange_and_addEPVii(i32* %a0, i32* %"a0'", i32 %a1)
; CHECK-NEXT: invert:
; CHECK-NEXT:   %a10 = atomicrmw volatile add i32* %a0, i32 %a1 acq_rel
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
