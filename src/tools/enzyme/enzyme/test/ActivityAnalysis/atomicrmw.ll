; RUN: %opt < %s %loadEnzyme -print-activity-analysis -activity-analysis-func=foo -o /dev/null | FileCheck %s

define void @foo() {
entry:
  %ap = alloca i32, align 8
  %a5 = atomicrmw volatile add i32* %ap, i32 -1 acq_rel
  %a7 = icmp eq i32 %a5, 1
  ret void
}

; CHECK: entry
; CHECK-NEXT:   %ap = alloca i32, align 8: icv:1 ici:1
; CHECK-NEXT:   %a5 = atomicrmw volatile add i32* %ap, i32 -1 acq_rel{{(, align 4)?}}: icv:1 ici:1
; CHECK-NEXT:   %a7 = icmp eq i32 %a5, 1: icv:1 ici:1
; CHECK-NEXT:   ret void: icv:1 ici:1
