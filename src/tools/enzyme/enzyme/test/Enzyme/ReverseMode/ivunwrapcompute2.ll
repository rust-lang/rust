; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -correlated-propagation -adce -instsimplify -early-cse-memssa -simplifycfg -correlated-propagation -adce -instsimplify -early-cse -simplifycfg -S | FileCheck %s

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
; CHECK-NEXT:   br i1 %cmp0, label %split, label %bb

; CHECK: split:
; CHECK-NEXT:  %fval = call i64 @idx(i64 %iv.next)
; CHECK-NEXT:  br label %mid

; CHECK: bb:                                               ; preds = %loop1
; CHECK-NEXT:   br i1 %cmp1, label %mid, label %loop1

; CHECK:  mid:                                              ; preds = %bb, %split
; CHECK-NEXT:    %mpn = phi i64 [ %fval, %split ], [ 15, %bb ]
; CHECK-NEXT:    %mcmp = icmp slt i64 %mpn, 0
; CHECK-NEXT:    %lim = select i1 %mcmp, i64 1, i64 %mpn
; CHECK-NEXT:    %[[a0:.+]] = add {{(nsw )?}}i64 %lim, -1
; CHECK-NEXT:    %mallocsize = mul nuw nsw i64 %lim, 8
; CHECK-NEXT:    %malloccall = tail call noalias nonnull i8* @malloc(i64 %mallocsize)
; CHECK-NEXT:    %g_malloccache = bitcast i8* %malloccall to double*
; CHECK-NEXT:    br label %loop2

; CHECK: loop2:  
; CHECK-NEXT:   %iv1 = phi i64 [ %iv.next2, %loop2 ], [ 0, %mid ]
; CHECK-NEXT:   %iv.next2 = add nuw nsw i64 %iv1, 1
; CHECK-NEXT:   %g = call double @get()
; CHECK-NEXT:   %[[a2:.+]] = getelementptr inbounds double, double* %g_malloccache, i64 %iv1
; CHECK-NEXT:   store double %g, double* %[[a2]], align 8, !invariant.group !
; CHECK-NEXT:   %cmp2 = icmp eq i64 %iv.next2, %lim
; CHECK-NEXT:   br i1 %cmp2, label %invertloop2, label %loop2

; CHECK: invertentry:                                      ; preds = %invertloop1
; CHECK-NEXT:   %[[a3:.+]] = insertvalue { double } undef, double %[[a8:.+]], 0
; CHECK-NEXT:   ret { double } %[[a3]]

; CHECK: invertloop1:                                      ; preds = %invertmid, %invertbb
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %"iv'ac.1", %invertbb ], [ 0, %invertmid ]
; CHECK-NEXT:   %[[a4:.+]] = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %[[a4]], label %invertentry, label %incinvertloop1

; CHECK: incinvertloop1:                                   ; preds = %invertloop1
; CHECK-NEXT:   %[[a5:.+]] = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertbb

; CHECK: invertbb:                                         ; preds = %invertmid, %incinvertloop1
; CHECK-NEXT:   %"iv'ac.1" = phi i64 [ %[[a5]], %incinvertloop1 ], [ 0, %invertmid ]
; CHECK-NEXT:   br label %invertloop1

; CHECK: invertmid:                                        ; preds = %invertloop2
; CHECK-NEXT:   tail call void @free(i8* nonnull %malloccall)
; CHECK-NEXT:   br i1 %cmp0, label %invertloop1, label %invertbb

; CHECK: invertloop2:                                      ; preds = %loop2, %incinvertloop2
; CHECK-NEXT:   %"arg'de.0" = phi double [ %[[a8]], %incinvertloop2 ], [ 0.000000e+00, %loop2 ]
; CHECK-NEXT:   %"iv1'ac.0" = phi i64 [ %[[a10:.+]], %incinvertloop2 ], [ %[[a0]], %loop2 ]
; CHECK-NEXT:   %[[a6:.+]] = getelementptr inbounds double, double* %g_malloccache, i64 %"iv1'ac.0"
; CHECK-NEXT:   %[[a7:.+]] = load double, double* %[[a6]], align 8, !invariant.group !0
; CHECK-NEXT:   %m1diffearg = fmul fast double %differeturn, %[[a7]]
; CHECK-NEXT:   %[[a8]] = fadd fast double %"arg'de.0", %m1diffearg
; CHECK-NEXT:   %[[a9:.+]] = icmp eq i64 %"iv1'ac.0", 0
; CHECK-NEXT:   br i1 %[[a9]], label %invertmid, label %incinvertloop2

; CHECK: incinvertloop2:                                   ; preds = %invertloop2
; CHECK-NEXT:   %[[a10]] = add nsw i64 %"iv1'ac.0", -1
; CHECK-NEXT:   br label %invertloop2
; CHECK-NEXT: }
