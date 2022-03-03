; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -early-cse -simplifycfg -S | FileCheck %s

; Function Attrs: nounwind
declare void @__enzyme_fwddiff.f64(...)

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

define void @dman_max(float* %a, float* %da1, float* %da2, float* %da3, float* %b, float* %db1, float* %db2, float* %db3) {
entry:
  call void (...) @__enzyme_fwddiff.f64(float (float*, float*)* @man_max, metadata !"enzyme_width", i64 3, float* %a, float* %da1, float* %da2, float* %da3, float* %b, float* %db1, float* %db2, float* %db3)
  ret void
}

attributes #0 = { noinline }


; CHECK: define internal [3 x float] @fwddiffe3man_max(float* %a, [3 x float*] %"a'", float* %b, [3 x float*] %"b'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = load float, float* %a, align 4
; CHECK-NEXT:   %1 = load float, float* %b, align 4
; CHECK-NEXT:   %cmp = fcmp ogt float %0, %1
; CHECK-NEXT:   %"a.b'ipse" = select i1 %cmp, [3 x float*] %"a'", [3 x float*] %"b'"
; CHECK-NEXT:   %2 = extractvalue [3 x float*] %"a.b'ipse", 0
; CHECK-NEXT:   %3 = load float, float* %2, align 4
; CHECK-NEXT:   %4 = insertvalue [3 x float] undef, float %3, 0
; CHECK-NEXT:   %5 = extractvalue [3 x float*] %"a.b'ipse", 1
; CHECK-NEXT:   %6 = load float, float* %5, align 4
; CHECK-NEXT:   %7 = insertvalue [3 x float] %4, float %6, 1
; CHECK-NEXT:   %8 = extractvalue [3 x float*] %"a.b'ipse", 2
; CHECK-NEXT:   %9 = load float, float* %8, align 4
; CHECK-NEXT:   %10 = insertvalue [3 x float] %7, float %9, 2
; CHECK-NEXT:   ret [3 x float] %10
; CHECK-NEXT: }
