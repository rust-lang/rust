; RUN: if [ %llvmver -ge 9 ] && [ %llvmver -le 11 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 9 ] && [ %llvmver -le 11 ]; then %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s; fi

define float @tester(float %start_value, <4 x float> %input) {
entry:
  %ord = call float @llvm.experimental.vector.reduce.v2.fadd.f32.v4f32(float %start_value, <4 x float> %input)
  ret float %ord
}

define float @test_derivative(float %start_value, <4 x float> %input) {
entry:
  %0 = tail call float (float (float, <4 x float>)*, ...) @__enzyme_fwddiff(float (float, <4 x float>)* nonnull @tester, float %start_value, float 1.0, <4 x float> %input, <4 x float> <float 1.0, float 1.0, float 1.0, float 1.0>)
  ret float %0
}

declare float @llvm.experimental.vector.reduce.v2.fadd.f32.v4f32(float, <4 x float>)

; Function Attrs: nounwind
declare float @__enzyme_fwddiff(float (float, <4 x float>)*, ...)


; CHECK: define internal {{(dso_local )?}}float @fwddiffetester(float %start_value, float %"start_value'", <4 x float> %input, <4 x float> %"input'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call fast float @llvm.experimental.vector.reduce.v2.fadd.f32.v4f32(float %"start_value'", <4 x float> %"input'")
; CHECK-NEXT:   ret float %0
; CHECK-NEXT: }
