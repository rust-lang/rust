; RUN: %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=inp -o /dev/null | FileCheck %s

declare void @f(i64 %x)

define void @inp(i64* %arrayidx) {
entry:
  br label %for.body.i.i

for.body.i.i:                                     ; preds = %for.body.i.i, %entry
  %indvars.iv.i.i = phi i64 [ %indvars.iv.next.i.i, %for.body.i.i ], [ 0, %entry ]
  %size.015.i.i = phi i64 [ %mul.i.i, %for.body.i.i ], [ 1, %entry ]
  %ld = load i64, i64* %arrayidx, !tbaa !2
  %mul.i.i = mul nsw i64 %ld, %size.015.i.i
  %indvars.iv.next.i.i = add nuw nsw i64 %indvars.iv.i.i, 1
  call void @f(i64 %mul.i.i)
  %exitcond.i.i = icmp eq i64 %indvars.iv.next.i.i, 4
  br i1 %exitcond.i.i, label %for.end.i.i, label %for.body.i.i

for.end.i.i:                                      ; preds = %for.body.i.i
  ret void
}


!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0, i64 8}
!3 = !{!4, i64 8, !"long"}
!4 = !{!5, i64 1, !"omnipotent char"}
!5 = !{!"Simple C++ TBAA"}

; CHECK: inp - {} |{[-1]:Pointer}:{} 
; CHECK-NEXT: i64* %arrayidx: {[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer, [-1,7]:Integer}
; CHECK-NEXT: entry
; CHECK-NEXT:   br label %for.body.i.i: {}
; CHECK-NEXT: for.body.i.i
; CHECK-NEXT:   %indvars.iv.i.i = phi i64 [ %indvars.iv.next.i.i, %for.body.i.i ], [ 0, %entry ]: {[-1]:Integer}
; CHECK-NEXT:   %size.015.i.i = phi i64 [ %mul.i.i, %for.body.i.i ], [ 1, %entry ]: {[-1]:Integer}
; CHECK-NEXT:   %ld = load i64, i64* %arrayidx{{(, align 4)?}}, !tbaa !2: {[-1]:Integer}
; CHECK-NEXT:   %mul.i.i = mul nsw i64 %ld, %size.015.i.i: {[-1]:Integer}
; CHECK-NEXT:   %indvars.iv.next.i.i = add nuw nsw i64 %indvars.iv.i.i, 1: {[-1]:Integer}
; CHECK-NEXT:   call void @f(i64 %mul.i.i)
; CHECK-NEXT:   %exitcond.i.i = icmp eq i64 %indvars.iv.next.i.i, 4: {[-1]:Integer}
; CHECK-NEXT:   br i1 %exitcond.i.i, label %for.end.i.i, label %for.body.i.i: {}
; CHECK-NEXT: for.end.i.i
; CHECK-NEXT:   ret void: {}
