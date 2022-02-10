; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

source_filename = "text"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

define private fastcc double @julia___2797(double %x0, i64 signext %x1) unnamed_addr #0 {
top:
  switch i64 %x1, label %L20 [
    i64 -1, label %L3
    i64 0, label %L7
    i64 1, label %L7.fold.split
    i64 2, label %L13
    i64 3, label %L17
  ]

L3:                                               ; preds = %top
  %x2 = fdiv double 1.000000e+00, %x0
  ret double %x2

L7.fold.split:                                    ; preds = %top
  br label %L7

L7:                                               ; preds = %top, %L7.fold.split
  %merge = phi double [ 1.000000e+00, %top ], [ %x0, %L7.fold.split ]
  ret double %merge

L13:                                              ; preds = %top
  %x3 = fmul double %x0, %x0
  ret double %x3

L17:                                              ; preds = %top
  %x4 = fmul double %x0, %x0
  %x5 = fmul double %x4, %x0
  ret double %x5

L20:                                              ; preds = %top
  %x6 = sitofp i64 %x1 to double
  %x7 = call double @llvm.pow.f64(double %x0, double %x6)
  ret double %x7
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare double @llvm.pow.f64(double, double) #1

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double, i64)*, ...)

; Function Attrs: alwaysinline nosync readnone
define double @julia_f_2794(double %y0, i64 signext %y1) {
entry:
  %y2 = call fastcc double @julia___2797(double %y0, i64 signext %y1) #5
  ret double %y2
}

define double @test_derivative(double %x, i64 %y) {
entry:
  %0 = tail call double (double (double, i64)*, ...) @__enzyme_autodiff(double (double, i64)* nonnull @julia_f_2794, double %x, i64 %y)
  ret double %0
}

; CHECK: define internal { double } @diffejulia_f_2794(double %y0, i64 signext %y1, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = sub i64 %y1, 1
; CHECK-NEXT:   %1 = call fast fastcc double @julia___2797(double %y0, i64 %0)
; CHECK-DAG:    %[[a2:.+]]  = fmul fast double %differeturn, %1
; CHECK-DAG:    %[[a3:.+]]  = sitofp i64 %y1 to double
; CHECK-NEXT:   %4 = fmul fast double %[[a2]], %[[a3]]
; CHECK-NEXT:   %5 = icmp eq i64 0, %y1
; CHECK-NEXT:   %6 = select {{(fast )?}}i1 %5, double 0.000000e+00, double %4
; CHECK-NEXT:   %7 = insertvalue { double } undef, double %6, 0
; CHECK-NEXT:   ret { double } %7
; CHECK-NEXT: }

; Function Attrs: inaccessiblemem_or_argmemonly
declare void @jl_gc_queue_root({} addrspace(10)*) #3

; Function Attrs: allocsize(1)
declare noalias nonnull {} addrspace(10)* @jl_gc_pool_alloc(i8*, i32, i32) #4

; Function Attrs: allocsize(1)
declare noalias nonnull {} addrspace(10)* @jl_gc_big_alloc(i8*, i64) #4

attributes #0 = { noinline readnone "enzyme_math"="powi" "enzyme_shouldrecompute"="powi"}
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { alwaysinline readnone "probe-stack"="inline-asm" }
attributes #3 = { inaccessiblemem_or_argmemonly }
attributes #4 = { allocsize(1) }
attributes #5 = { "probe-stack"="inline-asm" }

