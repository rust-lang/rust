; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -simplifycfg -adce -S | FileCheck %s

; Function Attrs: noinline nounwind uwtable
define dso_local float @man_max(float* %a, float* %b) #0 {
entry:
  %0 = load float, float* %a, align 4
  %1 = load float, float* %b, align 4
  %cmp = fcmp ogt float %0, %1
  %preb1 = insertelement <2 x float*> undef, float* %a, i32 0
  %vec = insertelement <2 x float*> %preb1, float* %b, i32 1
  %ovec = shufflevector <2 x float*> %vec, <2 x float*> undef, <1 x i32> zeroinitializer
  %ptr = extractelement <1 x float*> %ovec, i32 0
  %retval.0 = load float, float* %ptr, align 4
  ret float %retval.0
}

define void @dman_max(float* %a, float* %da, float* %b, float* %db) {
entry:
  call void (...) @__enzyme_autodiff.f64(float (float*, float*)* @man_max, float* %a, float* %da, float* %b, float* %db)
  ret void
}

declare void @__enzyme_autodiff.f64(...)

attributes #0 = { noinline }

; CHECK: define internal void @diffeman_max(float* %a, float* %"a'", float* %b, float* %"b'", float %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"preb1'ipie" = insertelement <2 x float*> undef, float* %"a'", i32 0
; CHECK-NEXT:   %"vec'ipie" = insertelement <2 x float*> %"preb1'ipie", float* %"b'", i32 1
; CHECK-NEXT:   %"ovec'ipsv" = shufflevector <2 x float*> %"vec'ipie", <2 x float*> undef, <1 x i32> zeroinitializer
; CHECK-NEXT:   %"ptr'ipee" = extractelement <1 x float*> %"ovec'ipsv", i32 0
; CHECK-NEXT:   %0 = load float, float* %"ptr'ipee", align 4
; CHECK-NEXT:   %1 = fadd fast float %0, %differeturn
; CHECK-NEXT:   store float %1, float* %"ptr'ipee", align 4
; CHECK-NEXT:   ret void
; CHECK-NEXT: }