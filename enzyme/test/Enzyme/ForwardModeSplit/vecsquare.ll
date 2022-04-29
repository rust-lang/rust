; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -simplifycfg -early-cse -instcombine -S | FileCheck %s

declare {float, float, float} @__enzyme_fwdsplit({float, float, float} (<4 x float>)*, <4 x float>, <4 x float>, i8*)

define {float, float, float} @square(<4 x float> %x) {
entry:
  %vec = insertelement <4 x float> %x, float 1.0, i32 3
  %sq = fmul <4 x float> %x, %x
  %cb = fmul <4 x float> %sq, %x          
  %id = shufflevector <4 x float> %sq, <4 x float> %cb, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %res1 = extractelement <4 x float> %id, i32 1
  %res2 = extractelement <4 x float> %id, i32 2
  %res3 = extractelement <4 x float> %id, i32 3
  %agg1 = insertvalue {float, float, float} undef, float %res1, 0
  %agg2 = insertvalue {float, float, float} %agg1, float %res2, 1
  %agg3 = insertvalue {float, float, float} %agg2, float %res3, 2
  ret {float, float, float} %agg3
}

define {float, float, float} @dsquare(<4 x float> %x) {
entry:
  %call = tail call {float, float, float} @__enzyme_fwdsplit({float, float, float} (<4 x float>)* @square, <4 x float> %x, <4 x float> <float 1.0, float 1.0, float 1.0, float 1.0>, i8* null)
  ret {float, float, float} %call
}


; CHECK: define internal { float, float, float } @fwddiffesquare(<4 x float> %x, <4 x float> %"x'", i8* %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:    %sq = fmul <4 x float> %x, %x
; CHECK-NEXT:    %[[i0:.+]] = fmul fast <4 x float> %"x'", %x
; CHECK-NEXT:    %[[i1:.+]] = fadd fast <4 x float> %[[i0]], %[[i0]]
; CHECK-NEXT:    %[[i2:.+]] = fmul fast <4 x float> %[[i1]], %x
; CHECK-NEXT:    %[[i3:.+]] = fmul fast <4 x float> %sq, %"x'"
; CHECK-NEXT:    %[[i4:.+]] = fadd fast <4 x float> %[[i2]], %[[i3]]
; CHECK-NEXT:    %[[i5:.+]] = extractelement <4 x float> %[[i1]], i32 1
; CHECK-NEXT:    %[[i6:.+]] = extractelement <4 x float> %[[i4]], i32 0
; CHECK-NEXT:    %[[i7:.+]] = extractelement <4 x float> %[[i4]], i32 1
; CHECK-NEXT:    %[[i8:.+]] = insertvalue { float, float, float } zeroinitializer, float %[[i5]], 0
; CHECK-NEXT:    %[[i9:.+]] = insertvalue { float, float, float } %[[i8]], float %[[i6]], 1
; CHECK-NEXT:    %[[i10:.+]] = insertvalue { float, float, float } %[[i9]], float %[[i7]], 2
; CHECK-NEXT:    ret { float, float, float } %[[i10]]
; CHECK-NEXT:  }
