; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -simplifycfg -instcombine -adce -S | FileCheck %s

define float* @sub(float* readonly returned %a, float* nocapture %ret) {
entry:
  %ld = load float, float* %a
  store float %ld, float* %ret
  ret float* %a
}

define float @caller(float* readonly %a) {
entry:
  %res = alloca float, align 4
  store float 0.000000e+00, float* %res
  %call = call float* @sub(float* %a, float* nonnull %res)
  %toret = load float, float* %res
  ret float %toret
}

define void @derivative(float* %a, float* %da) {
entry:
  %call = call double (i8*, ...) @__enzyme_autodiff(i8* bitcast (float (float*)* @caller to i8*), float* nonnull %a, float* nonnull %da)
  ret void
}

declare double @__enzyme_autodiff(i8*, ...)


; CHECK: define internal void @diffecaller(float* readonly %a, float* %"a'", float %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"res'ipa" = alloca float, align 4
; CHECK-NEXT:   store float 0.000000e+00, float* %"res'ipa", align 4
; CHECK-NEXT:   %res = alloca float, align 4
; CHECK-NEXT:   store float 0.000000e+00, float* %res, align 4
; CHECK-NEXT:   store float %differeturn, float* %"res'ipa", align 4
; CHECK-NEXT:   call void @diffesub(float* %a, float* %"a'", float* nonnull %res, float* nonnull %"res'ipa")
; CHECK-NEXT:   store float 0.000000e+00, float* %"res'ipa", align 4
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @diffesub(float* readonly %a, float* %"a'", float* nocapture %ret, float* nocapture %"ret'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %ld = load float, float* %a, align 4
; CHECK-NEXT:   store float %ld, float* %ret, align 4
; CHECK-NEXT:   %0 = load float, float* %"ret'", align 4
; CHECK-NEXT:   store float 0.000000e+00, float* %"ret'", align 4
; CHECK-NEXT:   %1 = load float, float* %"a'", align 4
; CHECK-NEXT:   %2 = fadd fast float %1, %0
; CHECK-NEXT:   store float %2, float* %"a'", align 4
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
