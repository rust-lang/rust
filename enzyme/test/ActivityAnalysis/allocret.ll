; RUN: %opt < %s %loadEnzyme -print-activity-analysis -activity-analysis-func=kernel_main -activity-analysis-duplicated-ret=1 -o /dev/null | FileCheck %s
; RUN: %opt < %s %loadEnzyme -print-activity-analysis -activity-analysis-func=kernel_main -activity-analysis-duplicated-ret=0 -o /dev/null | FileCheck %s --check-prefix=INCHECK

declare i8* @malloc(i64)

define float* @kernel_main(float %tmp1) {
entry:
  %tmp11 = call i8* @malloc(i64 4)
  %tmp12 = bitcast i8* %tmp11 to float*
  store float %tmp1, float* %tmp12, align 4
  ret float* %tmp12
}

; CHECK: float %tmp1: icv:0
; CHECK-NEXT: entry
; CHECK-NEXT:   %tmp11 = call i8* @malloc(i64 4): icv:0 ici:1
; CHECK-NEXT:   %tmp12 = bitcast i8* %tmp11 to float*: icv:0 ici:1
; CHECK-NEXT:   store float %tmp1, float* %tmp12, align 4: icv:1 ici:0
; CHECK-NEXT:   ret float* %tmp12: icv:1 ici:1

; INCHECK: float %tmp1: icv:0
; INCHECK-NEXT: entry
; INCHECK-NEXT:   %tmp11 = call i8* @malloc(i64 4): icv:1 ici:1
; INCHECK-NEXT:   %tmp12 = bitcast i8* %tmp11 to float*: icv:1 ici:1
; INCHECK-NEXT:   store float %tmp1, float* %tmp12, align 4: icv:1 ici:1
; INCHECK-NEXT:   ret float* %tmp12: icv:1 ici:1
