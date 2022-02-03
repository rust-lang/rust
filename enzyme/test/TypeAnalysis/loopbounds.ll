; RUN: %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=compute_loops -o /dev/null | FileCheck %s

define void @compute_loops(i8* nocapture %out) {
entry:
  br label %body

body:
  %i = phi i64 [ 13, %entry ], [ %next, %body ]
  %next = add i64 %i, 3
  %gep = getelementptr i8, i8* %out, i64 %i
  store i8 0, i8* %gep, align 1, !tbaa !8
  %cmp = icmp eq i64 %next, 40
  br i1 %cmp, label %exit, label %body

exit:
  ret void
}

!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}
!7 = !{!"int", !5, i64 0}
!8 = !{!7, !7, i64 0}

; CHECK: compute_loops - {} |{[-1]:Pointer}:{}
; CHECK-NEXT: i8* %out: {[-1]:Pointer, [-1,13]:Integer, [-1,16]:Integer, [-1,19]:Integer, [-1,22]:Integer, [-1,25]:Integer, [-1,28]:Integer, [-1,31]:Integer, [-1,34]:Integer, [-1,37]:Integer}
; CHECK-NEXT: entry
; CHECK-NEXT:   br label %body: {}
; CHECK-NEXT: body
; CHECK-NEXT:   %i = phi i64 [ 13, %entry ], [ %next, %body ]: {[-1]:Integer}
; CHECK-NEXT:   %next = add i64 %i, 3: {[-1]:Integer}
; CHECK-NEXT:   %gep = getelementptr i8, i8* %out, i64 %i: {[-1]:Pointer, [-1,0]:Integer}
; CHECK-NEXT:   store i8 0, i8* %gep, align 1, !tbaa !0: {}
; CHECK-NEXT:   %cmp = icmp eq i64 %next, 40: {[-1]:Integer}
; CHECK-NEXT:   br i1 %cmp, label %exit, label %body: {}
; CHECK-NEXT: exit
; CHECK-NEXT:   ret void: {}
