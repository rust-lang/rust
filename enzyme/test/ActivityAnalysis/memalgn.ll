; RUN: %opt < %s %loadEnzyme -print-activity-analysis -activity-analysis-func=matvec -o /dev/null | FileCheck %s

define float @matvec(float %inp) {
entry:
  %x = alloca float*, align 8
  %i = bitcast float** %x to i8**
  %call = call i32 @posix_memalign(i8** %i, i64 8, i64 8)
  %i2 = load float*, float** %x, align 8
  store float %inp, float* %i2, align 4
  %i3 = load float*, float** %x, align 8
  %i4 = load float, float* %i3, align 4
  ret float %i4
}

declare i32 @posix_memalign(i8**, i64, i64)

; CHECK: float %inp: icv:0
; CHECK: entry
; CHECK-NEXT:   %x = alloca float*, align 8: icv:0 ici:1
; CHECK-NEXT:   %i = bitcast float** %x to i8**: icv:0 ici:1
; CHECK-NEXT:   %call = call i32 @posix_memalign(i8** %i, i64 8, i64 8): icv:1 ici:0
; CHECK-NEXT:   %i2 = load float*, float** %x, align 8: icv:0 ici:1
; CHECK-NEXT:   store float %inp, float* %i2, align 4: icv:1 ici:0
; CHECK-NEXT:   %i3 = load float*, float** %x, align 8: icv:0 ici:1
; CHECK-NEXT:   %i4 = load float, float* %i3, align 4: icv:0 ici:0
; CHECK-NEXT:   ret float %i4: icv:1 ici:1
