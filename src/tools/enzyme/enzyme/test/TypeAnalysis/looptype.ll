; RUN: %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=callee -o /dev/null | FileCheck %s

define void @callee(i64* %call, i64* %call3, i64* %call6) {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  %add8 = add nsw i64 %sub, %add
  ret void

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 3, %entry ], [ %indvars.iv.next, %for.body ]
  %index.addr.023 = phi i64 [ 0, %entry ], [ %sub, %for.body ]
  %startInput.021 = phi i64 [ 0, %entry ], [ %add, %for.body ]
  %0 = load i64, i64* %call, align 8, !tbaa !8
  %div = sdiv i64 %index.addr.023, %0
  %1 = load i64, i64* %call3, align 8, !tbaa !8
  %mul = mul nsw i64 %1, %div
  %add = add nsw i64 %mul, %startInput.021
  %2 = load i64, i64* %call6, align 8, !tbaa !8
  %mul7 = mul nsw i64 %2, %div
  %sub = sub nsw i64 %index.addr.023, %mul7
  %indvars.iv.next = add nsw i64 %indvars.iv, -1
  %cmp = icmp ugt i64 %indvars.iv, 1
  br i1 %cmp, label %for.body, label %for.cond.cleanup
}

!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}
!7 = !{!"long", !5, i64 0}
!8 = !{!7, !7, i64 0}

; CHECK: callee - {} |{[-1]:Pointer}:{} {[-1]:Pointer}:{} {[-1]:Pointer}:{} 
; CHECK-NEXT: i64* %call: {[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer, [-1,7]:Integer}
; CHECK-NEXT: i64* %call3: {[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer, [-1,7]:Integer}
; CHECK-NEXT: i64* %call6: {[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer, [-1,7]:Integer}
; CHECK-NEXT: entry
; CHECK-NEXT:   br label %for.body: {}
; CHECK-NEXT: for.cond.cleanup
; CHECK-NEXT:   %add8 = add nsw i64 %sub, %add: {[-1]:Integer}
; CHECK-NEXT:   ret void: {}
; CHECK-NEXT: for.body
; CHECK-NEXT:   %indvars.iv = phi i64 [ 3, %entry ], [ %indvars.iv.next, %for.body ]: {[-1]:Integer}
; CHECK-NEXT:   %index.addr.023 = phi i64 [ 0, %entry ], [ %sub, %for.body ]: {[-1]:Integer}
; CHECK-NEXT:   %startInput.021 = phi i64 [ 0, %entry ], [ %add, %for.body ]: {[-1]:Integer}
; CHECK-NEXT:   %0 = load i64, i64* %call, align 8, !tbaa !0: {[-1]:Integer}
; CHECK-NEXT:   %div = sdiv i64 %index.addr.023, %0: {[-1]:Integer}
; CHECK-NEXT:   %1 = load i64, i64* %call3, align 8, !tbaa !0: {[-1]:Integer}
; CHECK-NEXT:   %mul = mul nsw i64 %1, %div: {[-1]:Integer}
; CHECK-NEXT:   %add = add nsw i64 %mul, %startInput.021: {[-1]:Integer}
; CHECK-NEXT:   %2 = load i64, i64* %call6, align 8, !tbaa !0: {[-1]:Integer}
; CHECK-NEXT:   %mul7 = mul nsw i64 %2, %div: {[-1]:Integer}
; CHECK-NEXT:   %sub = sub nsw i64 %index.addr.023, %mul7: {[-1]:Integer}
; CHECK-NEXT:   %indvars.iv.next = add nsw i64 %indvars.iv, -1: {[-1]:Integer}
; CHECK-NEXT:   %cmp = icmp ugt i64 %indvars.iv, 1: {[-1]:Integer}
; CHECK-NEXT:   br i1 %cmp, label %for.body, label %for.cond.cleanup: {}
