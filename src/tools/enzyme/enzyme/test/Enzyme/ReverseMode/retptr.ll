; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -instsimplify -adce -correlated-propagation -simplifycfg -S | FileCheck %s

%mat = type { float* }

define float @f(float** %this) {
entry:
  %call = tail call float* @sub(float** %this)
  %res = load float, float* %call, align 4
  ret float %res
}

define float* @sub(float** %this)  {
entry:
  %0 = load float*, float** %this, align 8
  ret float* %0
}

define float @g(float** %this, float** %dthis) {
entry:
  %0 = tail call float (float (float**)*, ...) @__enzyme_autodiff(float (float**)* @f, float** %this, float** %dthis)
  ret float %0
}

declare float @__enzyme_autodiff(float (float**)*, ...)

; CHECK: define internal void @diffef(float** %this, float** %"this'", float %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[augsub:.+]] = call float* @augmented_sub(float** %this, float** %"this'")
; CHECK-NEXT:   %[[loadsub:.+]] = load float, float* %[[augsub]]
; CHECK-NEXT:   %[[fadd:.+]] = fadd fast float %[[loadsub]], %differeturn
; CHECK-NEXT:   store float %[[fadd]], float* %[[augsub]]
; CHECK-NEXT:   call void @diffesub(float** %this, float** %"this'")
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal float* @augmented_sub(float** %this, float** %"this'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"'ipl" = load float*, float** %"this'", align 8
; CHECK-NEXT:   ret float* %"'ipl"
; CHECK-NEXT: }

; CHECK: define internal void @diffesub(float** %this, float** %"this'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
