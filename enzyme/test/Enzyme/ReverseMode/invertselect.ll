; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -early-cse -simplifycfg -S | FileCheck %s

; Function Attrs: noinline nounwind uwtable
define dso_local float @man_max(float* %a, float* %b) #0 {
entry:
  %0 = load float, float* %a, align 4
  %1 = load float, float* %b, align 4
  %cmp = fcmp ogt float %0, %1
  %a.b = select i1 %cmp, float* %a, float* %b
  %retval.0 = load float, float* %a.b, align 4
  ret float %retval.0
}

define void @dman_max(float* %a, float* %da, float* %b, float* %db) {
entry:
  call void (...) @__enzyme_autodiff.f64(float (float*, float*)* @man_max, float* %a, float* %da, float* %b, float* %db)
  ret void
}

declare void @__enzyme_autodiff.f64(...)

attributes #0 = { noinline }

; CHECK: define internal {{(dso_local )?}}void @diffeman_max(float* %a, float* %"a'", float* %b, float* %"b'", float %[[differet:.+]])
; CHECK-NEXT: entry:
; CHECK-NEXT:  %[[la:.+]] = load float, float* %a, align 4
; CHECK-NEXT:  %[[lb:.+]] = load float, float* %b, align 4
; CHECK-NEXT:  %[[cmp:.+]] = fcmp ogt float %[[la]], %[[lb]]
; CHECK-NEXT:  %[[abp:.+]] = select i1 %[[cmp]], float* %"a'", float* %"b'"
; CHECK-NEXT:  %[[prep:.+]] = load float, float* %[[abp]]
; CHECK-NEXT:  %[[postp:.]] = fadd fast float %[[prep]], %[[differet]]
; CHECK-NEXT:  store float %[[postp]], float* %[[abp]]
; CHECK-NEXT:  ret void
; CHECK-NEXT: }
