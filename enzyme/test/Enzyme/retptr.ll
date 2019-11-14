; RUN: %opt < %s %loadEnzyme -enzyme -enzyme_preopt=false -mem2reg -sroa -instsimplify -adce -correlated-propagation -simplifycfg -S | FileCheck %s

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

; CHECK: define internal {} @diffef(float** %this, float** %"this'", float %differeturn) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call { {}, float*, float* } @augmented_sub(float** %this, float** %"this'")
; CHECK-NEXT:   %1 = extractvalue { {}, float*, float* } %0, 2
; CHECK-NEXT:   %2 = load float, float* %1
; CHECK-NEXT:   %3 = fadd fast float %2, %differeturn
; CHECK-NEXT:   store float %3, float* %1
; CHECK-NEXT:   %4 = call {} @diffesub(float** %this, float** %"this'", {} undef)
; CHECK-NEXT:   ret {} undef
; CHECK-NEXT: }

; CHECK: define internal { {}, float*, float* } @augmented_sub(float** %this, float** %"this'") {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"'ipl" = load float*, float** %"this'", align 8
; CHECK-NEXT:   %[[real:.+]] = load float*, float** %this, align 8
; CHECK-NEXT:   %[[ins1:.+]] = insertvalue { {}, float*, float* } undef, float* %0, 1
; CHECK-NEXT:   %[[ins2:.+]] = insertvalue { {}, float*, float* } %[[ins1]], float* %"'ipl", 2
; CHECK-NEXT:   ret { {}, float*, float* } %[[ins2]]
; CHECK-NEXT: }

; CHECK: define internal {} @diffesub(float** %this, float** %"this'", {} %tapeArg) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret {} undef
; CHECK-NEXT: }
