; RUN: %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=callee -o /dev/null | FileCheck %s


@ptr = private unnamed_addr global [3 x i64] zeroinitializer, align 1

define void @callee() {
entry:
  %loadnotype = load i64, i64* getelementptr inbounds ([3 x i64], [3 x i64]* @ptr, i64 0, i32 1)
  store i64 %loadnotype, i64* getelementptr inbounds ([3 x i64], [3 x i64]* @ptr, i64 0, i32 0)
  %loadtype = load i64, i64* getelementptr inbounds ([3 x i64], [3 x i64]* @ptr, i64 0, i32 1)
  store i64 %loadtype, i64* getelementptr inbounds ([3 x i64], [3 x i64]* @ptr, i64 0, i32 2), !tbaa !8
  %self = getelementptr inbounds [3 x i64], [3 x i64]* @ptr, i64 0
  ret void
}

!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}
!7 = !{!"double", !5, i64 0}
!8 = !{!7, !7, i64 0}


; CHECK: callee - {} |
; CHECK-NEXT: entry
; CHECK-NEXT:   %loadnotype = load i64, i64* getelementptr inbounds ([3 x i64], [3 x i64]* @ptr, i64 0, i32 1): {[-1]:Float@double}
; CHECK-NEXT:   store i64 %loadnotype, i64* getelementptr inbounds ([3 x i64], [3 x i64]* @ptr, i64 0, i32 0): {}
; CHECK-NEXT:   %loadtype = load i64, i64* getelementptr inbounds ([3 x i64], [3 x i64]* @ptr, i64 0, i32 1): {[-1]:Float@double}
; CHECK-NEXT:   store i64 %loadtype, i64* getelementptr inbounds ([3 x i64], [3 x i64]* @ptr, i64 0, i32 2), !tbaa !0: {}
; CHECK-NEXT:   %self = getelementptr inbounds [3 x i64], [3 x i64]* @ptr, i64 0: {[-1]:Pointer, [-1,0]:Float@double, [-1,8]:Float@double, [-1,16]:Float@double}
; CHECK-NEXT:   ret void: {}
