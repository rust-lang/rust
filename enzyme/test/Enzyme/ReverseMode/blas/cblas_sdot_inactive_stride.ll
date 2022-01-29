;RUN: %opt < %s %loadEnzyme -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

;#include <cblas.h>
;
;extern float __enzyme_autodiff(void *, float *, float *, float *,
;                                 float *, int);
;
;float g(float *restrict m, float *restrict n, int stride) {
;  float x = cblas_sdot(3, m, 2, n, stride);
;  float y = x * x;
;  return y;
;}
;
;int main() {
;  float m[6] = {1, 2, 3, 101, 102, 103};
;  float m1[6] = {0, 0, 0, 0, 0, 0};
;  float n[9] = {4, 5, 6, 104, 105, 106, 7, 8, 9};
;  float n1[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
;  __enzyme_autodiff((void*)g, m, m1, n, n1, 3);
;}

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@__const.main.m = private unnamed_addr constant [6 x float] [float 1.000000e+00, float 2.000000e+00, float 3.000000e+00, float 1.010000e+02, float 1.020000e+02, float 1.030000e+02], align 16
@__const.main.n = private unnamed_addr constant [9 x float] [float 4.000000e+00, float 5.000000e+00, float 6.000000e+00, float 1.040000e+02, float 1.050000e+02, float 1.060000e+02, float 7.000000e+00, float 8.000000e+00, float 9.000000e+00], align 16

define dso_local float @g(float* noalias %m, float* noalias %n, i32 %stride) {
entry:
  %m.addr = alloca float*, align 8
  %n.addr = alloca float*, align 8
  %stride.addr = alloca i32, align 4
  %x = alloca float, align 4
  %y = alloca float, align 4
  store float* %m, float** %m.addr, align 8
  store float* %n, float** %n.addr, align 8
  store i32 %stride, i32* %stride.addr, align 4
  %0 = load float*, float** %m.addr, align 8
  %1 = load float*, float** %n.addr, align 8
  %2 = load i32, i32* %stride.addr, align 4
  %call = call float @cblas_sdot(i32 3, float* %0, i32 2, float* %1, i32 %2)
  store float %call, float* %x, align 4
  %3 = load float, float* %x, align 4
  %4 = load float, float* %x, align 4
  %mul = fmul float %3, %4
  store float %mul, float* %y, align 4
  %5 = load float, float* %y, align 4
  ret float %5
}

declare dso_local float @cblas_sdot(i32, float*, i32, float*, i32)

define dso_local i32 @main() {
entry:
  %m = alloca [6 x float], align 16
  %m1 = alloca [6 x float], align 16
  %n = alloca [9 x float], align 16
  %n1 = alloca [9 x float], align 16
  %0 = bitcast [6 x float]* %m to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %0, i8* align 16 bitcast ([6 x float]* @__const.main.m to i8*), i64 24, i1 false)
  %1 = bitcast [6 x float]* %m1 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 16 %1, i8 0, i64 24, i1 false)
  %2 = bitcast [9 x float]* %n to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %2, i8* align 16 bitcast ([9 x float]* @__const.main.n to i8*), i64 36, i1 false)
  %3 = bitcast [9 x float]* %n1 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 16 %3, i8 0, i64 36, i1 false)
  %arraydecay = getelementptr inbounds [6 x float], [6 x float]* %m, i32 0, i32 0
  %arraydecay1 = getelementptr inbounds [6 x float], [6 x float]* %m1, i32 0, i32 0
  %arraydecay2 = getelementptr inbounds [9 x float], [9 x float]* %n, i32 0, i32 0
  %arraydecay3 = getelementptr inbounds [9 x float], [9 x float]* %n1, i32 0, i32 0
  %call = call float @__enzyme_autodiff(i8* bitcast (float (float*, float*, i32)* @g to i8*), float* %arraydecay, float* %arraydecay1, float* %arraydecay2, float* %arraydecay3, i32 3)
  ret i32 0
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1)

declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1)

declare dso_local float @__enzyme_autodiff(i8*, float*, float*, float*, float*, i32)

;CHECK:define internal void @diffeg(float* noalias %m, float* %"m'", float* noalias %n, float* %"n'", i32 %stride, float %differeturn)
;CHECK-NEXT:entry:
;CHECK-NEXT:  %call = call float @cblas_sdot(i32 3, float* nocapture readonly %m, i32 2, float* nocapture readonly %n, i32 %stride)
;CHECK-NEXT:  %m0diffecall = fmul fast float %differeturn, %call
;CHECK-NEXT:  %m1diffecall = fmul fast float %differeturn, %call
;CHECK-NEXT:  %0 = fadd fast float %m0diffecall, %m1diffecall
;CHECK-NEXT:  call void @cblas_saxpy(i32 3, float %0, float* %m, i32 2, float* %"n'", i32 %stride)
;CHECK-NEXT:  call void @cblas_saxpy(i32 3, float %0, float* %n, i32 %stride, float* %"m'", i32 2)
;CHECK-NEXT:  ret void
;CHECK-NEXT:}

;CHECK:declare void @cblas_saxpy(i32, float, float*, i32, float*, i32)
