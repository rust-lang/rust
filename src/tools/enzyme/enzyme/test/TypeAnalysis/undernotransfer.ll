; RUN: %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=caller -o /dev/null | FileCheck %s

declare void @f(i64 %x)

define void @caller(i8* %p) {
entry:
  store i8 undef, i8* %p, align 1
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
; CHECK-NEXT: i8* %p: {[-1]:Pointer}
; CHECK-NEXT: entry
; CHECK-NEXT:   store i8 undef, i8* %p, align 1: {}
; CHECK-NEXT:   ret void: {}
