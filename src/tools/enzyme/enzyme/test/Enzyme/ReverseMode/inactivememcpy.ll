; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -simplifycfg -dce -instcombine -S | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:10:11:12:13"
target triple = "x86_64-linux-gnu"

define i64 @metamove(i8* %dst, i8* %src, i64* %lenp) {
top:
  %len = load i64, i64* %lenp, align 8
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst, i8* %src, i64 %len, i1 false)
  ret i64 0
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8*, i8*, i64, i1)

declare i8* @malloc(i64)


define float @f(i8* %src, float %v) {
top:
  %i = alloca i64, align 8

  store i64 0, i64* %i, align 8

  %fdst = alloca float, align 4
  store float %v, float* %fdst, align 4

  %dst = bitcast float* %fdst to i8*

  %i12 = call i64 @metamove(i8* %dst, i8* %src, i64* %i)

  %l = load float, float* %fdst, align 4
  %l2 = fmul float %l, %l
  ret float %l2
}

declare float @__enzyme_autodiff(...)

define float @caller(i8* %arg1, float %v) {
entry:
  %f = call float (...) @__enzyme_autodiff(float (i8*, float)* @f, metadata !"enzyme_const", i8* %arg1, float %v)
  ret float %f
}

; CHECK: define internal i64 @augmented_metamove(i8* %dst, i8* %"dst'", i8* %src, i64* %lenp, i64* %"lenp'")
; CHECK-NEXT: top:
; CHECK-NEXT:   %len = load i64, i64* %lenp, align 8
; CHECK-NEXT:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %dst, i8* align 1 %src, i64 %len, i1 false)
; CHECK-NEXT:   ret i64 %len
; CHECK-NEXT: }

; CHECK: define internal void @diffemetamove(i8* %dst, i8* %"dst'", i8* %src, i64* %lenp, i64* %"lenp'", i64 %len)
; CHECK-NEXT: top:
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* align 1 %"dst'", i8 0, i64 %len, i1 false)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
