; RUN: %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=caller -enzyme-max-type-offset=1000 -o /dev/null | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare i64* @get()

define void @caller() {
entry:
  %p = call i64* @get()
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ -102, %entry ], [ %indvars.iv.next, %for.body ]
  %sub = sub i64 0, %indvars.iv
  %np = getelementptr i64, i64* %p, i64 %sub
  %ld = load i64, i64* %np, align 8, !tbaa !2
  %indvars.iv.next = add nuw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv, -99
  br i1 %exitcond, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void
}

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0, i64 8}
!3 = !{!4, i64 8, !"double"}
!4 = !{!5, i64 1, !"omnipotent char"}
!5 = !{!"Simple C++ TBAA"}

; CHECK: entry
; CHECK-NEXT:   %p = call i64* @get(): {[-1]:Pointer, [-1,792]:Float@double, [-1,800]:Float@double}
; CHECK-NEXT:   br label %for.body: {}
; CHECK-NEXT: for.body
; CHECK-NEXT:   %indvars.iv = phi i64 [ -102, %entry ], [ %indvars.iv.next, %for.body ]: {[-1]:Integer}
; CHECK-NEXT:   %sub = sub i64 0, %indvars.iv: {[-1]:Anything}
; CHECK-NEXT:   %np = getelementptr i64, i64* %p, i64 %sub: {[-1]:Pointer, [-1,0]:Float@double}
; CHECK-NEXT:   %ld = load i64, i64* %np, align 8, !tbaa !2: {[-1]:Float@double}
; CHECK-NEXT:   %indvars.iv.next = add nuw i64 %indvars.iv, 1: {[-1]:Integer}
; CHECK-NEXT:   %exitcond = icmp eq i64 %indvars.iv, -99: {[-1]:Integer}
; CHECK-NEXT:   br i1 %exitcond, label %for.cond.cleanup, label %for.body: {}
; CHECK-NEXT: for.cond.cleanup
; CHECK-NEXT:   ret void: {}
