; RUN: if [ %llvmver -ge 9 ] &&  [ %llvmver -le 11 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi

define float @tester(float %start_value, <4 x float> %input) {
entry:
  %ord = call float @llvm.experimental.vector.reduce.v2.fadd.f32.v4f32(float %start_value, <4 x float> %input)
  ret float %ord
}

define float @test_derivative(float %start_value, <4 x float> %input) {
entry:
  %0 = tail call float (float (float, <4 x float>)*, ...) @__enzyme_autodiff(float (float, <4 x float>)* nonnull @tester, float %start_value, <4 x float> %input)
  ret float %0
}

declare float @llvm.experimental.vector.reduce.v2.fadd.f32.v4f32(float, <4 x float>)

; Function Attrs: nounwind
declare float @__enzyme_autodiff(float (float, <4 x float>)*, ...)


; CHECK: define internal {{(dso_local )?}}{ float, <4 x float> } @diffetester(float %start_value, <4 x float> %input, float %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = insertelement <4 x float> undef, float %differeturn, i64 0
; CHECK-NEXT:   %1 = shufflevector <4 x float> %0, <4 x float> undef, <4 x i32> zeroinitializer
; CHECK-NEXT:   %2 = insertvalue { float, <4 x float> } undef, float %differeturn, 0
; CHECK-NEXT:   %3 = insertvalue { float, <4 x float> } %2, <4 x float> %1, 1
; CHECK-NEXT:   ret { float, <4 x float> } %3
; CHECK-NEXT: }
