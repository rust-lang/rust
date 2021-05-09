; RUN: if [ %llvmver -ge 10 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -correlated-propagation -adce -instsimplify -early-cse-memssa -simplifycfg -correlated-propagation -adce -jump-threading -instsimplify -early-cse -simplifycfg -S | FileCheck %s; fi

; ModuleID = 'q2.ll'
source_filename = "text"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:10:11:12:13"
target triple = "x86_64-linux-gnu"

define double @test_derivative(double %x) {
entry:
  %tmp = tail call double (double (double)*, ...) @__enzyme_autodiff(double (double)* nonnull @julia_airyai_795, double %x)
  ret double %tmp
}

declare double @__enzyme_autodiff(double (double)*, ...)

; Function Attrs: alwaysinline nofree norecurse nounwind
define double @julia_airyai_795(double %arg) {
entry:
  br label %loop1

loop1:                                             ; preds = %bb, %zsqrt_.exit
  %i = phi i64 [ 0, %entry ], [ %nexti, %bb ]
  %nexti = add nuw nsw i64 %i, 1
  %tmp16 = icmp eq i64 %nexti, 16
  %tmp23 = fcmp olt double %arg, 1.234567e+00
  br i1 %tmp23, label %.split.loop.exit, label %bb

.split.loop.exit:                                 ; preds = %bb17
  %tmp27 = trunc i64 %i to i32
  br label %mid

bb:                                               ; preds = %bb17
  br i1 %tmp16, label %mid, label %loop1
  
mid:                               ; preds = %bb, %.split.loop.exit
  ; this is the issue
  %tmp28 = phi i32 [ %tmp27, %.split.loop.exit ], [ 15, %bb ]
  %tmp29 = sext i32 %tmp28 to i64
  %tmp30 = add nsw i64 %tmp29, 1
  %tmp31 = icmp slt i32 %tmp28, 0
  %tmp32 = select i1 %tmp31, i64 1, i64 %tmp30
  br label %loop2

loop2:                                           ; preds = %loop2, %mid
  %psum = phi double [ %res, %loop2 ], [ 0.000000e+00, %mid ]
  %j = phi i64 [ %nextj, %loop2 ], [ 0, %mid ]
  %tmp35 = phi double [ %tmp38, %loop2 ], [ 1.000000e+00, %mid ]
  %tmp36 = fmul double %arg, %tmp35
  %res = fadd double %psum, %tmp36
  %tmp38 = fneg double %tmp35
  %nextj = add nuw nsw i64 %j, 1
  %tmp40 = icmp eq i64 %nextj, %tmp32
  br i1 %tmp40, label %exit, label %loop2

exit:                                        ; preds = %loop2
  ret double %res
}

; CHECK: define internal { double } @diffejulia_airyai_795(double %arg, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   br label %loop1

; CHECK: loop1:                                            ; preds = %bb, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %bb ], [ 0, %entry ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %tmp16 = icmp eq i64 %iv.next, 16
; CHECK-NEXT:   %tmp23 = fcmp olt double %arg, 0x3FF3C0C9539B8887
; CHECK-NEXT:   br i1 %tmp23, label %mid, label %bb

; CHECK: bb:                                               ; preds = %loop1
; CHECK-NEXT:   br i1 %tmp16, label %mid.thread, label %loop1

; CHECK: mid:                                              ; preds = %loop1
; CHECK-NEXT:   %tmp27 = trunc i64 %iv to i32
; CHECK-NEXT:   %tmp29 = sext i32 %tmp27 to i64
; CHECK-NEXT:   %tmp30 = add nsw i64 %tmp29, 1
; CHECK-NEXT:   %tmp31 = icmp slt i32 %tmp27, 0
; CHECK-NEXT:   %spec.select = select i1 %tmp31, i64 1, i64 %tmp30
; CHECK-NEXT:   br label %mid.thread

; CHECK: mid.thread:                                       ; preds = %mid, %bb
; CHECK-NEXT:   %0 = phi i64 [ 16, %bb ], [ %spec.select, %mid ]
; CHECK-NEXT:   %1 = sub nsw i64 %0, 1
; CHECK-NEXT:   %mallocsize = mul nuw nsw i64 %0, 8
; CHECK-NEXT:   %malloccall = tail call noalias nonnull i8* @malloc(i64 %mallocsize)
; CHECK-NEXT:   %tmp35_malloccache = bitcast i8* %malloccall to double*
; CHECK-NEXT:   br label %loop2

; CHECK: loop2:                                            ; preds = %loop2, %mid.thread
; CHECK-NEXT:   %iv1 = phi i64 [ %iv.next2, %loop2 ], [ 0, %mid.thread ]
; CHECK-NEXT:   %tmp35 = phi double [ %tmp38, %loop2 ], [ 1.000000e+00, %mid.thread ]
; CHECK-NEXT:   %2 = getelementptr inbounds double, double* %tmp35_malloccache, i64 %iv1
; CHECK-NEXT:   store double %tmp35, double* %2, align 8
; CHECK-NEXT:   %iv.next2 = add nuw nsw i64 %iv1, 1
; CHECK-NEXT:   %tmp38 = fneg double %tmp35
; CHECK-NEXT:   %tmp40 = icmp eq i64 %iv.next2, %0
; CHECK-NEXT:   br i1 %tmp40, label %invertloop2, label %loop2

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
; CHECK-NEXT:   br i1 %tmp23, label %invertloop1, label %invertbb

; CHECK: invertloop2:                                      ; preds = %loop2, %incinvertloop2
; CHECK-NEXT:   %"arg'de.0" = phi double [ %8, %incinvertloop2 ], [ 0.000000e+00, %loop2 ]
; CHECK-NEXT:   %"iv1'ac.0" = phi i64 [ %10, %incinvertloop2 ], [ %1, %loop2 ]
; CHECK-NEXT:   %6 = getelementptr inbounds double, double* %tmp35_malloccache, i64 %"iv1'ac.0"
; CHECK-NEXT:   %7 = load double, double* %6, align 8
; CHECK-NEXT:   %m0diffearg = fmul fast double %differeturn, %7
; CHECK-NEXT:   %8 = fadd fast double %"arg'de.0", %m0diffearg
; CHECK-NEXT:   %9 = icmp eq i64 %"iv1'ac.0", 0
; CHECK-NEXT:   br i1 %9, label %invertmid, label %incinvertloop2

; CHECK: incinvertloop2:                                   ; preds = %invertloop2
; CHECK-NEXT:   %10 = add nsw i64 %"iv1'ac.0", -1
; CHECK-NEXT:   br label %invertloop2
; CHECK-NEXT: }