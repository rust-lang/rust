; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -S | FileCheck %s

define float @f(float %this) {
entry:
  %call = tail call float @sub(float %this)
  %res = fmul float %call, %call
  ret float %res
}

declare void @julia.write_barrier(float* readnone nocapture)

define float @sub(float %this)  {
entry:
  %alloc = alloca float, align 256
  store float %this, float* %alloc, align 8
  call void @julia.write_barrier(float* %alloc)
  ret float %this
}

define float @g(float %t) {
entry:
  %0 = tail call float (float (float)*, ...) @__enzyme_autodiff(float (float)* @f, float %t)
  ret float %0
}

declare float @__enzyme_autodiff(float (float)*, ...)

; ensure both alignment is maintained and that the alloca
; is not preserved for the reverse
; CHECK: define internal float @augmented_sub(float %this)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %alloc = alloca float, i64 1, align 256
; CHECK-NEXT:   store float %this, float* %alloc, align 8
; CHECK-NEXT:   call void @julia.write_barrier(float* %alloc)
; CHECK-NEXT:   ret float %this
; CHECK-NEXT: }
