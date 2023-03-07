; RUN: %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=caller -o /dev/null | FileCheck %s

declare void @f(i64 %x)

define void @caller(<4 x i32> %pre) {
entry:
  %post = sitofp <4 x i32> %pre to <4 x double>
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


; CHECK: caller - {} |{[-1]:Integer}:{} 
; CHECK-NEXT: <4 x i32> %pre: {[-1]:Integer}
; CHECK-NEXT: entry
; CHECK-NEXT:   %post = sitofp <4 x i32> %pre to <4 x double>: {[-1]:Float@double}
; CHECK-NEXT:   ret void: {}