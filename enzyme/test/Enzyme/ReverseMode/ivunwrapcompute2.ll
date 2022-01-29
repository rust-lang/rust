; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -correlated-propagation -adce -instsimplify -early-cse-memssa -simplifycfg -correlated-propagation -adce -jump-threading -instsimplify -early-cse -simplifycfg -S | FileCheck %s

; ModuleID = 'q2.ll'
source_filename = "text"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:10:11:12:13"
target triple = "x86_64-linux-gnu"

define double @test_derivative(double %x) {
entry:
  %tmp = tail call double (double (double)*, ...) @__enzyme_autodiff(double (double)* nonnull @f, double %x)
  ret double %tmp
}

declare double @__enzyme_autodiff(double (double)*, ...)

declare double @get()

declare i1 @bool()

define i64 @idx(i64 %x) {
entry:
  %y = add i64 %x, 3
  ret i64 %y
}

define double @f(double %arg) {
entry:
  %cmp0 = call i1 @bool()
  br label %loop1

loop1:                                       ; preds = %bb, %zsqrt_.exit
  %i = phi i64 [ %ni, %bb ], [ 0, %entry ]
  %ni = add nuw nsw i64 %i, 1
  %cmp1 = icmp eq i64 %ni, 16
  br i1 %cmp0, label %split, label %bb

split:                                 ; preds = %.preheader
  %fval = call i64 @idx(i64 %ni)
  br label %mid

bb:                                               ; preds = %.preheader
  br i1 %cmp1, label %mid, label %loop1

mid:                               ; preds = %bb, %.split.loop.exit
  %mpn = phi i64 [ %fval, %split ], [ 15, %bb ]
  %mcmp = icmp slt i64 %mpn, 0
  %lim = select i1 %mcmp, i64 1, i64 %mpn
  br label %loop2

loop2:                                           ; preds = %.lr.ph, %.split.loop.exit11
  %psum = phi double [ %res, %loop2 ], [ 0.000000e+00, %mid ]
  %j = phi i64 [ %nj, %loop2 ], [ 0, %mid ]
  %g = call double @get()
  %rere = fmul double %g, %arg
  %res = fadd double %psum, %rere
  %nj = add nuw nsw i64 %j, 1
  %cmp2 = icmp eq i64 %nj, %lim
  br i1 %cmp2, label %exit, label %loop2

exit:                                        ; preds = %.lr.ph
  ret double %res
}

; CHECK: define internal { double } @diffef(double %arg, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %cmp0 = call i1 @bool()
; CHECK-NEXT:   br label %loop1

; CHECK: loop1:                                            ; preds = %bb, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %bb ], [ 0, %entry ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %cmp1 = icmp eq i64 %iv.next, 16
; CHECK-NEXT:   br i1 %cmp0, label %mid, label %bb

; CHECK: bb:                                               ; preds = %loop1
; CHECK-NEXT:   br i1 %cmp1, label %mid.thread, label %loop1

; CHECK: mid:                                              ; preds = %loop1
; CHECK-NEXT:   %fval = call i64 @idx(i64 %iv.next)
; CHECK-NEXT:   %mcmp = icmp slt i64 %fval, 0
; CHECK-NEXT:   %spec.select = select i1 %mcmp, i64 1, i64 %fval
; CHECK-NEXT:   br label %mid.thread

; CHECK: mid.thread:                                       ; preds = %mid, %bb
; CHECK-NEXT:   %0 = phi i64 [ 15, %bb ], [ %spec.select, %mid ]
; CHECK-NEXT:   %1 = add {{(nsw )?}}i64 %0, -1
; CHECK-NEXT:   %mallocsize = mul nuw nsw i64 %0, 8
; CHECK-NEXT:   %malloccall = tail call noalias nonnull i8* @malloc(i64 %mallocsize)
; CHECK-NEXT:   %g_malloccache = bitcast i8* %malloccall to double*
; CHECK-NEXT:   br label %loop2

; CHECK: loop2:                                            ; preds = %loop2, %mid.thread
; CHECK-NEXT:   %iv1 = phi i64 [ %iv.next2, %loop2 ], [ 0, %mid.thread ]
; CHECK-NEXT:   %iv.next2 = add nuw nsw i64 %iv1, 1
; CHECK-NEXT:   %g = call double @get()
; CHECK-NEXT:   %2 = getelementptr inbounds double, double* %g_malloccache, i64 %iv1
; CHECK-NEXT:   store double %g, double* %2, align 8, !invariant.group !0
; CHECK-NEXT:   %cmp2 = icmp eq i64 %iv.next2, %0
; CHECK-NEXT:   br i1 %cmp2, label %invertloop2, label %loop2

; CHECK: invertentry:                                      ; preds = %invertloop1
; CHECK-NEXT:   %3 = insertvalue { double } undef, double %8, 0
; CHECK-NEXT:   ret { double } %3

; CHECK: invertloop1:                                      ; preds = %invertmid, %invertbb
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %"iv'ac.1", %invertbb ], [ 0, %invertmid ]
; CHECK-NEXT:   %4 = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %4, label %invertentry, label %incinvertloop1

; CHECK: incinvertloop1:                                   ; preds = %invertloop1
; CHECK-NEXT:   %5 = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertbb

; CHECK: invertbb:                                         ; preds = %invertmid, %incinvertloop1
; CHECK-NEXT:   %"iv'ac.1" = phi i64 [ %5, %incinvertloop1 ], [ 0, %invertmid ]
; CHECK-NEXT:   br label %invertloop1

; CHECK: invertmid:                                        ; preds = %invertloop2
; CHECK-NEXT:   tail call void @free(i8* nonnull %malloccall)
; CHECK-NEXT:   br i1 %cmp0, label %invertloop1, label %invertbb

; CHECK: invertloop2:                                      ; preds = %loop2, %incinvertloop2
; CHECK-NEXT:   %"arg'de.0" = phi double [ %8, %incinvertloop2 ], [ 0.000000e+00, %loop2 ]
; CHECK-NEXT:   %"iv1'ac.0" = phi i64 [ %10, %incinvertloop2 ], [ %1, %loop2 ]
; CHECK-NEXT:   %6 = getelementptr inbounds double, double* %g_malloccache, i64 %"iv1'ac.0"
; CHECK-NEXT:   %7 = load double, double* %6, align 8, !invariant.group !0
; CHECK-NEXT:   %m1diffearg = fmul fast double %differeturn, %7
; CHECK-NEXT:   %8 = fadd fast double %"arg'de.0", %m1diffearg
; CHECK-NEXT:   %9 = icmp eq i64 %"iv1'ac.0", 0
; CHECK-NEXT:   br i1 %9, label %invertmid, label %incinvertloop2

; CHECK: incinvertloop2:                                   ; preds = %invertloop2
; CHECK-NEXT:   %10 = add nsw i64 %"iv1'ac.0", -1
; CHECK-NEXT:   br label %invertloop2
; CHECK-NEXT: }
