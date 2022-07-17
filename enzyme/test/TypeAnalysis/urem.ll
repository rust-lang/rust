; RUN: %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=kernel_main -o /dev/null | FileCheck %s

declare i8* @malloc(i64)

define void @kernel_main(float* %tmp1) {
entry:
  %tmp11 = call i8* @malloc(i64 140)
  %tmp12 = bitcast i8* %tmp11 to float*
  %tmp13 = ptrtoint float* %tmp12 to i64
  %tmp14 = add i64 %tmp13, 127
  %tmp15 = urem i64 %tmp14, 128
  %tmp16 = sub i64 %tmp14, %tmp15
  %tmp17 = inttoptr i64 %tmp16 to float*
  %tmp29 = load float, float* %tmp1, align 4
  store float %tmp29, float* %tmp17, align 4
  ret void
}

; CHECK: kernel_main - {} |{[-1]:Pointer, [-1,-1]:Float@float}:{}
; CHECK-NEXT: float* %tmp1: {[-1]:Pointer, [-1,-1]:Float@float}
; CHECK-NEXT: entry
; CHECK-NEXT:   %tmp11 = call i8* @malloc(i64 140): {[-1]:Pointer}
; CHECK-NEXT:   %tmp12 = bitcast i8* %tmp11 to float*: {[-1]:Pointer}
; CHECK-NEXT:   %tmp13 = ptrtoint float* %tmp12 to i64: {[-1]:Pointer}
; CHECK-NEXT:   %tmp14 = add i64 %tmp13, 127: {[-1]:Pointer}
; CHECK-NEXT:   %tmp15 = urem i64 %tmp14, 128: {[-1]:Integer}
; CHECK-NEXT:   %tmp16 = sub i64 %tmp14, %tmp15: {[-1]:Pointer, [-1,0]:Float@float}
; CHECK-NEXT:   %tmp17 = inttoptr i64 %tmp16 to float*: {[-1]:Pointer, [-1,0]:Float@float}
; CHECK-NEXT:   %tmp29 = load float, float* %tmp1, align 4: {[-1]:Float@float}
; CHECK-NEXT:   store float %tmp29, float* %tmp17, align 4: {}
; CHECK-NEXT:   ret void: {}

