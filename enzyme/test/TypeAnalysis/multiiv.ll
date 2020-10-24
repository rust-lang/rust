; RUN: %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=compute_loops -o /dev/null | FileCheck %s

declare i64* @ptr1()

declare double* @ptr2()

define void @compute_loops() #1 {
entry:
  %a = call i64* @ptr1()
  %b = call double* @ptr2()

  %pair1 = insertvalue { i64*, double* } undef, i64* %a, 0
  %pair2 = insertvalue { i64*, double* } %pair1, double* %b, 1

  %sub_a = extractvalue { i64*, double* } %pair2, 0 
  %sub_b = extractvalue { i64*, double* } %pair2, 1

  %iload = load i64, i64* %sub_a, align 8, !tbaa !7

  %fload = load double, double* %sub_b, align 8, !tbaa !3


  ret void
}

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0}
!3 = !{!"double", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
!6 = !{!7, !7, i64 0}
!7 = !{!"int", !4, i64 0}

; CHECK: compute_loops - {} |
; CHECK-NEXT: entry
; CHECK-NEXT:   %a = call i64* @ptr1(): {[-1]:Pointer, [0,0]:Integer, [0,1]:Integer, [0,2]:Integer, [0,3]:Integer, [0,4]:Integer, [0,5]:Integer, [0,6]:Integer, [0,7]:Integer}
; CHECK-NEXT:   %b = call double* @ptr2(): {[-1]:Pointer, [-1,0]:Float@double}
; CHECK-NEXT:   %pair1 = insertvalue { i64*, double* } undef, i64* %a, 0: {[0]:Pointer, [0,0]:Integer, [0,1]:Integer, [0,2]:Integer, [0,3]:Integer, [0,4]:Integer, [0,5]:Integer, [0,6]:Integer, [0,7]:Integer, [8]:Anything, [9]:Anything, [10]:Anything, [11]:Anything, [12]:Anything, [13]:Anything, [14]:Anything, [15]:Anything}
; CHECK-NEXT:   %pair2 = insertvalue { i64*, double* } %pair1, double* %b, 1: {[-1]:Pointer, [0,0]:Integer, [0,1]:Integer, [0,2]:Integer, [0,3]:Integer, [0,4]:Integer, [0,5]:Integer, [0,6]:Integer, [0,7]:Integer, [8,0]:Float@double}
; CHECK-NEXT:   %sub_a = extractvalue { i64*, double* } %pair2, 0: {[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer, [-1,7]:Integer}
; CHECK-NEXT:   %sub_b = extractvalue { i64*, double* } %pair2, 1: {[-1]:Pointer, [-1,0]:Float@double}
; CHECK-NEXT:   %iload = load i64, i64* %sub_a, align 8, !tbaa !2: {[-1]:Integer}
; CHECK-NEXT:   %fload = load double, double* %sub_b, align 8, !tbaa !6: {[-1]:Float@double}
; CHECK-NEXT:   ret void: {}