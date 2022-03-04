; RUN: if [ %llvmver -ge 12 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi

define float @tester(float %start_value, <4 x float> %input) {
entry:
  %ord = call float @llvm.vector.reduce.fadd.v4f32(float %start_value, <4 x float> %input)
  ret float %ord
}

define float @test_derivative(float %start_value, <4 x float> %input) {
entry:
  %0 = tail call float (float (float, <4 x float>)*, ...) @__enzyme_fwdsplit(float (float, <4 x float>)* nonnull @tester, float %start_value, float 1.0, <4 x float> %input, <4 x float> <float 1.0, float 1.0, float 1.0, float 1.0>, i8* null)
  ret float %0
}

declare float @llvm.vector.reduce.fadd.v4f32(float, <4 x float>)

; Function Attrs: nounwind
declare float @__enzyme_fwdsplit(float (float, <4 x float>)*, ...)


; CHECK: define internal {{(dso_local )?}}float @fwddiffetester(float %start_value, float %"start_value'", <4 x float> %input, <4 x float> %"input'", i8* %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   tail call void @free(i8* nonnull %tapeArg)
; CHECK-NEXT:   %0 = call fast float @llvm.vector.reduce.fadd.v4f32(float %"start_value'", <4 x float> %"input'")
; CHECK-NEXT:   ret float %0
; CHECK-NEXT: }
