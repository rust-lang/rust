; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -instsimplify -simplifycfg -S | FileCheck %s

; ModuleID = 'ld-temp.o'
source_filename = "ld-temp.o"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@enzyme_const = internal global i32 0, align 4

define internal double @callable(double* %x) {
  ret double 1.000000e+00
}

define internal void @_Z19testSensitivitiesADv(double* %primal, double* %grad, double* %a) {
  br label %bb1

bb1:
  %a1 = load i32, i32* @enzyme_const, align 4
  br label %bb3

bb2:
  %a2 = load i32, i32* @enzyme_const, align 4
  br label %bb3

bb3:
  %a3 = phi i32 [ %a1, %bb1 ], [ %a2, %bb2 ]
  ;%a3 = load i32, i32* @enzyme_const, align 4

  %c1 = call double (...) @__enzyme_autodiff(i8* bitcast (double (double*)* @callable to i8*), i32 %a3, double* %a)
  ret void
}
declare dso_local double @__enzyme_autodiff(...)

; CHECK: define internal void @diffecallable(double* %x, double %differeturn)
; CHECK-NEXT: invert:
; CHECK-NEXT:  ret void
; CHECK-NEXT: }
