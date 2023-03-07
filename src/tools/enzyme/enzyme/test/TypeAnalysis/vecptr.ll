; RUN: %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=callee -o /dev/null | FileCheck %s

define void @callee(<2 x i64>* %argp) {
entry:
  %arg = load <2 x i64>, <2 x i64>* %argp, align 16
  %bv = bitcast <2 x i64> %arg to i128
  %tr = trunc i128 %bv to i64
  %ptr = inttoptr i64 %tr to double*
  %ld = load double, double* %ptr, align 8, !tbaa !8
  ret void
}

!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}
!7 = !{!"double", !5, i64 0}
!8 = !{!7, !7, i64 0}

; CHECK: callee - {} |{[-1]:Pointer}:{} 
; CHECK-NEXT: <2 x i64>* %argp: {[-1]:Pointer, [-1,0]:Pointer, [-1,0,0]:Float@double}
; CHECK-NEXT: entry
; CHECK-NEXT:   %arg = load <2 x i64>, <2 x i64>* %argp, align 16: {[0]:Pointer, [0,0]:Float@double}
; CHECK-NEXT:   %bv = bitcast <2 x i64> %arg to i128: {[0]:Pointer, [0,0]:Float@double}
; CHECK-NEXT:   %tr = trunc i128 %bv to i64: {[-1]:Pointer, [-1,0]:Float@double}
; CHECK-NEXT:   %ptr = inttoptr i64 %tr to double*: {[-1]:Pointer, [-1,0]:Float@double}
; CHECK-NEXT:   %ld = load double, double* %ptr, align 8, !tbaa !0: {[-1]:Float@double}
; CHECK-NEXT:   ret void: {}
