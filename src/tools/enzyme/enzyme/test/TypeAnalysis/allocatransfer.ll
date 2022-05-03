; RUN: %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=callee -o /dev/null | FileCheck %s

define void @callee(i64* %ptr) {
entry:
  %i = alloca i64, align 4
  store i64 2, i64* %i, align 4
  %l2 = load i64, i64* %i, align 4
  %ptr2 = getelementptr inbounds i64, i64* %ptr, i64 %l2
  %loadtype = load i64, i64* %ptr2, align 4, !tbaa !8

  %cptr2 = getelementptr inbounds i64, i64* %ptr, i64 2
  %loadnotype = load i64, i64* %cptr2, align 4
  ret void
}

!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}
!7 = !{!"double", !5, i64 0}
!8 = !{!7, !7, i64 0}


; CHECK: callee - {} |{[-1]:Pointer}:{} 
; CHECK-NEXT: i64* %ptr: {[-1]:Pointer, [-1,16]:Float@double}
; CHECK-NEXT: entry
; CHECK-NEXT:   %i = alloca i64, align 4: {[-1]:Pointer, [-1,-1]:Integer}
; CHECK-NEXT:   store i64 2, i64* %i, align 4: {}
; CHECK-NEXT:   %l2 = load i64, i64* %i, align 4: {[-1]:Integer}
; CHECK-NEXT:   %ptr2 = getelementptr inbounds i64, i64* %ptr, i64 %l2: {[-1]:Pointer, [-1,0]:Float@double}
; CHECK-NEXT:   %loadtype = load i64, i64* %ptr2, align 4, !tbaa !0: {[-1]:Float@double}
; CHECK-NEXT:   %cptr2 = getelementptr inbounds i64, i64* %ptr, i64 2: {[-1]:Pointer, [-1,0]:Float@double}
; CHECK-NEXT:   %loadnotype = load i64, i64* %cptr2, align 4: {[-1]:Float@double}
; CHECK-NEXT:   ret void: {}
