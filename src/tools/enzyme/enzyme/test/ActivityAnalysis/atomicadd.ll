; RUN: %opt < %s %loadEnzyme -print-activity-analysis -activity-analysis-func=matvec -o /dev/null | FileCheck %s

define void @matvec(i32* %ptr, i32 %v) {
  %l = load i32, i32* %ptr, align 4, !tbaa !2
  %aa = atomicrmw volatile add i32* %ptr, i32 -1 acq_rel
  ret void
}

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0, i64 4}
!3 = !{!4, i64 4, !"long"}
!4 = !{!5, i64 1, !"omnipotent char"}
!5 = !{!"Simple C++ TBAA"}

; CHECK:  %l = load i32, i32* %ptr, align 4, !tbaa !2: icv:1 ici:1
; CHECK-NEXT:  %aa = atomicrmw volatile add i32* %ptr, i32 -1 acq_rel{{(, align 4)?}}: icv:1 ici:1
; CHECK-NEXT:  ret void: icv:1 ici:1
