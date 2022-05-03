; RUN: %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=caller -o /dev/null | FileCheck %s

declare void @f(i64 %x)

define void @caller(i64* %p) {
entry:
  %ld = load i64, i64* %p, align 4
  %int = ptrtoint i64* %p to i64
  %and = and i64 %int, -16
  %ptr = inttoptr i64 %and to i64*
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
; CHECK-NEXT: i64* %p: {[-1]:Pointer}
; CHECK-NEXT: entry
; CHECK-NEXT:   %ld = load i64, i64* %p, align 4: {}
; CHECK-NEXT:   %int = ptrtoint i64* %p to i64: {[-1]:Pointer}
; CHECK-NEXT:   %and = and i64 %int, -16: {[-1]:Pointer}
; CHECK-NEXT:   %ptr = inttoptr i64 %and to i64*: {[-1]:Pointer}
; CHECK-NEXT:   ret void: {}
