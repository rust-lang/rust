; RUN: %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=mainloop -o /dev/null | FileCheck %s

@timeron = internal unnamed_addr global float 0.000000e+00, align 4

define void @mainloop() {
entry:
  %a3 = load float, float* @timeron, align 4
  %c3 = load float, float* @timeron, align 4
  %d = fadd float %a3, %c3
  %r = load float, float* @timeron, align 4
  ret void
}

; CHECK: mainloop - {} |
; CHECK-NEXT: entry
; CHECK-NEXT:   %a3 = load float, float* @timeron, align 4: {[-1]:Float@float}
; CHECK-NEXT:   %c3 = load float, float* @timeron, align 4: {[-1]:Float@float}
; CHECK-NEXT:   %d = fadd float %a3, %c3: {[-1]:Float@float}
; CHECK-NEXT:   %r = load float, float* @timeron, align 4: {[-1]:Float@float}
; CHECK-NEXT:   ret void: {}
