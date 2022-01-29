; RUN: %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=callee -o /dev/null | FileCheck %s

define void @callee(i64* %ptr) {
entry:
  %ptr2 = getelementptr inbounds i64, i64* %ptr, i64 2
  %loadnotype = load i64, i64* %ptr2
  %ptr3 = getelementptr inbounds i64, i64* %ptr, i64 3
  store i64 %loadnotype, i64* %ptr3

  %cast = bitcast i64* %ptr to <2 x float>*
  %cast2 = bitcast <2 x float>* %cast to i64*
  %cptr2 = getelementptr inbounds i64, i64* %cast2, i64 2
  %loadtype = load i64, i64* %cptr2
  %cptr4 = getelementptr inbounds i64, i64* %cast2, i64 4
  store i64 %loadtype, i64* %cptr4, !tbaa !8
  ret void
}

!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}
!7 = !{!"double", !5, i64 0}
!8 = !{!7, !7, i64 0}


; CHECK: callee - {} |{[-1]:Pointer}:{}
; CHECK-NEXT: i64* %ptr: {[-1]:Pointer, [-1,16]:Float@double, [-1,24]:Float@double, [-1,32]:Float@double}
; CHECK-NEXT: entry
; CHECK-NEXT:   %ptr2 = getelementptr inbounds i64, i64* %ptr, i64 2: {[-1]:Pointer, [-1,0]:Float@double, [-1,8]:Float@double, [-1,16]:Float@double}
; CHECK-NEXT:   %loadnotype = load i64, i64* %ptr2{{(, align 4)?}}: {[-1]:Float@double}
; CHECK-NEXT:   %ptr3 = getelementptr inbounds i64, i64* %ptr, i64 3: {[-1]:Pointer, [-1,0]:Float@double, [-1,8]:Float@double}
; CHECK-NEXT:   store i64 %loadnotype, i64* %ptr3{{(, align 4)?}}: {}
; CHECK-NEXT:   %cast = bitcast i64* %ptr to <2 x float>*: {[-1]:Pointer, [-1,16]:Float@double, [-1,24]:Float@double, [-1,32]:Float@double}
; CHECK-NEXT:   %cast2 = bitcast <2 x float>* %cast to i64*: {[-1]:Pointer, [-1,16]:Float@double, [-1,24]:Float@double, [-1,32]:Float@double}
; CHECK-NEXT:   %cptr2 = getelementptr inbounds i64, i64* %cast2, i64 2: {[-1]:Pointer, [-1,0]:Float@double, [-1,8]:Float@double, [-1,16]:Float@double}
; CHECK-NEXT:   %loadtype = load i64, i64* %cptr2{{(, align 4)?}}: {[-1]:Float@double}
; CHECK-NEXT:   %cptr4 = getelementptr inbounds i64, i64* %cast2, i64 4: {[-1]:Pointer, [-1,0]:Float@double}
; CHECK-NEXT:   store i64 %loadtype, i64* %cptr4{{(, align 4)?}}, !tbaa !0: {}
; CHECK-NEXT:   ret void: {}
