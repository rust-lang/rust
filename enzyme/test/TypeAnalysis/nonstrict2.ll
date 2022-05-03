; RUN: %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=matvec -enzyme-strict-aliasing=0 -o /dev/null | FileCheck %s

define internal void @matvec(i8* %ptr, i1 %cmp) {
entry:
  br i1 %cmp, label %doubleB, label %intB

doubleB:
  %dptr = bitcast i8* %ptr to double*
  store double 0.000000e+00, double* %dptr, align 8, !tbaa !8
  ret void

intB:
  %dint = bitcast i8* %ptr to i64*
  store i64 0, i64* %dint, align 8, !tbaa !10
  ret void
}

!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}
!7 = !{!"double", !5, i64 0}
!8 = !{!7, !7, i64 0}

!9 = !{!"int", !5, i64 0}
!10 = !{!9, !9, i64 0}

; CHECK: matvec - {} |{[-1]:Pointer}:{} {[-1]:Integer}:{}
; CHECK-NEXT: i8* %ptr: {[-1]:Pointer}
; CHECK-NEXT: i1 %cmp: {[-1]:Integer}
; CHECK-NEXT: entry
; CHECK-NEXT:   br i1 %cmp, label %doubleB, label %intB: {}
; CHECK-NEXT: doubleB
; CHECK-NEXT:   %dptr = bitcast i8* %ptr to double*: {[-1]:Pointer, [-1,0]:Float@double}
; CHECK-NEXT:   store double 0.000000e+00, double* %dptr, align 8, !tbaa !0: {}
; CHECK-NEXT:   ret void: {}
; CHECK-NEXT: intB
; CHECK-NEXT:   %dint = bitcast i8* %ptr to i64*: {[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer, [-1,7]:Integer}
; CHECK-NEXT:   store i64 0, i64* %dint, align 8, !tbaa !4: {}
; CHECK-NEXT:   ret void: {}
