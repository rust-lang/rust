; RUN: %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=caller -o /dev/null | FileCheck %s

declare void @f(i64 %x)

define void @caller(i64* %q) {
entry:
  %p = bitcast i64* %q to i64**
  %x = alloca i64, align 8
  store i64 132, i64* %x, align 8
  store i64* %x, i64** %p, align 8
  %ld = load i64*, i64** %p, align 8
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
; CHECK-NEXT: i64* %q: {[-1]:Pointer, [-1,0]:Pointer, [-1,0,0]:Integer, [-1,0,1]:Integer, [-1,0,2]:Integer, [-1,0,3]:Integer, [-1,0,4]:Integer, [-1,0,5]:Integer, [-1,0,6]:Integer, [-1,0,7]:Integer}
; CHECK-NEXT: entry
; CHECK-NEXT:   %p = bitcast i64* %q to i64**: {[-1]:Pointer, [-1,0]:Pointer, [-1,0,0]:Integer, [-1,0,1]:Integer, [-1,0,2]:Integer, [-1,0,3]:Integer, [-1,0,4]:Integer, [-1,0,5]:Integer, [-1,0,6]:Integer, [-1,0,7]:Integer}
; CHECK-NEXT:   %x = alloca i64, align 8: {[-1]:Pointer, [-1,-1]:Integer}
; CHECK-NEXT:   store i64 132, i64* %x, align 8: {}
; CHECK-NEXT:   store i64* %x, i64** %p, align 8: {}
; CHECK-NEXT:   %ld = load i64*, i64** %p, align 8: {[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer, [-1,7]:Integer}
; CHECK-NEXT:   ret void: {}
