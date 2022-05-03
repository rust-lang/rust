; RUN: %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=caller -o /dev/null | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @caller(i64* %p) {
entry:
  %ld = load i64, i64* %p, align 8, !tbaa !2
  %iv1 = insertvalue { i32, i64 } undef, i64 %ld, 1
  %iv2 = insertvalue { i32, i64 } %iv1, i32 4, 0
  %ev1 = extractvalue { i32, i64 } %iv2, 0
  %ev2 = extractvalue { i32, i64 } %iv2, 1
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

; CHECK: caller - {} |{[-1]:Pointer}:{} 
; CHECK-NEXT: i64* %p: {[-1]:Pointer, [-1,0]:Float@double}
; CHECK-NEXT: entry
; CHECK-NEXT:   %ld = load i64, i64* %p, align 8, !tbaa !2: {[-1]:Float@double}
; CHECK-NEXT:   %iv1 = insertvalue { i32, i64 } undef, i64 %ld, 1: {[0]:Anything, [1]:Anything, [2]:Anything, [3]:Anything, [4]:Anything, [5]:Anything, [6]:Anything, [7]:Anything, [8]:Float@double}
; CHECK-NEXT:   %iv2 = insertvalue { i32, i64 } %iv1, i32 4, 0: {[0]:Integer, [1]:Integer, [2]:Integer, [3]:Integer, [4]:Anything, [5]:Anything, [6]:Anything, [7]:Anything, [8]:Float@double}
; CHECK-NEXT:   %ev1 = extractvalue { i32, i64 } %iv2, 0: {[-1]:Integer}
; CHECK-NEXT:   %ev2 = extractvalue { i32, i64 } %iv2, 1: {[-1]:Float@double}
; CHECK-NEXT:   ret void: {}
