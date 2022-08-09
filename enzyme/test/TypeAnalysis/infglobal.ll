; RUN: %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=mainloop -o /dev/null | FileCheck %s

@timeron = internal unnamed_addr global i1 false, align 4

define void @mainloop() {
entry:
  %a3 = load i1, i1* @timeron, align 4
  %c3 = load i1, i1* @timeron, align 4
  br i1 %a3, label %a4, label %a5

a4:
  br label %a5

a5:
  ret void
}

; CHECK: mainloop - {} |
; CHECK-NEXT: entry
; CHECK-NEXT:   %a3 = load i1, i1* @timeron, align 4: {[-1]:Integer}
; CHECK-NEXT:   %c3 = load i1, i1* @timeron, align 4: {[-1]:Integer}
; CHECK-NEXT:   br i1 %a3, label %a4, label %a5: {}
; CHECK-NEXT: a4
; CHECK-NEXT:   br label %a5: {}
; CHECK-NEXT: a5
; CHECK-NEXT:   ret void: {}
