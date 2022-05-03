; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -simplifycfg -adce -S | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:10:11:12:13"
target triple = "x86_64-linux-gnu"

; Function Attrs: readnone
declare i16 @julia.tid()

define void @foo(double* %a0, i64 signext %a1) {
entry:
  %.not = icmp eq i64 %a1, 0
  br i1 %.not, label %L6.i, label %L9.i

L6.i:                                             ; preds = %entry
  %a5 = call i16 @julia.tid()
  %a6 = icmp slt i16 %a5, 1
  br i1 %a6, label %L9.i, label %julia_foo_1762.inner.exit

L9.i:                                             ; preds = %L6.i, %entry
  store double 2.000000e+00, double* %a0, align 8
  %.not11 = icmp eq i64 %a1, 1
  br i1 %.not11, label %L13.i, label %julia_foo_1762.inner.exit

L13.i:                                            ; preds = %L9.i
  %g = getelementptr inbounds double, double* %a0, i32 1
  store double 3.000000e+00, double* %g, align 8
  br label %julia_foo_1762.inner.exit

julia_foo_1762.inner.exit:                        ; preds = %L9.i, %L13.i, %L6.i
  ret void
}

declare void @__enzyme_autodiff.f64(...) local_unnamed_addr

define void @caller(double* %a0, double* %b0, i64 signext %a1) {
entry:
  call void (...) @__enzyme_autodiff.f64(void (double*, i64)* nonnull @foo, double* nonnull %a0, double* nonnull %b0, i64 %a1)
  ret void
}

; CHECK: define internal void @diffefoo(double* %a0, double* %"a0'", i64 signext %a1)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %.not = icmp eq i64 %a1, 0
; CHECK-NEXT:   br i1 %.not, label %L6.i, label %L9.i

; CHECK: L6.i:                                             ; preds = %entry
; CHECK-NEXT:   %a5 = call i16 @julia.tid()
; CHECK-NEXT:   %a6 = icmp slt i16 %a5, 1
; CHECK-NEXT:   br i1 %a6, label %L9.i, label %invertjulia_foo_1762.inner.exit

; CHECK: L9.i:                                             ; preds = %L6.i, %entry
; CHECK-NEXT:   store double 2.000000e+00, double* %a0, align 8
; CHECK-NEXT:   %.not11 = icmp eq i64 %a1, 1
; CHECK-NEXT:   br i1 %.not11, label %L13.i, label %invertjulia_foo_1762.inner.exit

; CHECK: L13.i:                                            ; preds = %L9.i
; CHECK-NEXT:   %g = getelementptr inbounds double, double* %a0, i32 1
; CHECK-NEXT:   store double 3.000000e+00, double* %g, align 8
; CHECK-NEXT:   br label %invertjulia_foo_1762.inner.exit

; CHECK: invertentry:
; CHECK-NEXT:   ret void

; CHECK: invertL9.i:
; CHECK-NEXT:   store double 0.000000e+00, double* %"a0'", align 8
; CHECK-NEXT:   br label %invertentry

; CHECK: invertL13.i:
; CHECK-NEXT:   %"g'ipg_unwrap" = getelementptr inbounds double, double* %"a0'", i32 1
; CHECK-NEXT:   store double 0.000000e+00, double* %"g'ipg_unwrap", align 8
; CHECK-NEXT:   br label %invertL9.i

; NOTE this needs to be a cache rather than attempting to merge an uncomputable condition
; CHECK: invertjulia_foo_1762.inner.exit:
; CHECK-NEXT:   %_cache.0 = phi i8 [ 2, %L13.i ], [ 1, %L9.i ], [ 0, %L6.i ]
; CHECK-NEXT:   switch i8 %_cache.0, label %invertL13.i [
; CHECK-NEXT:     i8 0, label %invertentry
; CHECK-NEXT:     i8 1, label %invertL9.i
; CHECK-NEXT:   ]
; CHECK-NEXT: }
