;RUN: %opt < %s %loadEnzyme -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

;#include <cblas.h>
;
;extern float __enzyme_autodiff(float*, float*, float*, float*, float*);
;
;float g(float *restrict m, float *restrict n) {
;	 float x = cblas_sdot(3, m, 1, n, 1);
;    float y = x*x;
;	 return y;
;}
;
;int main() {
;	 float m[3] = {1, 2, 3};
;	 float n[3] = {4, 5, 6};
;	 float m1[3] = {0.};
;	 float n1[3] = {0.};
;	 float z = __enzyme_autodiff((float*)g, m, m1, n, n1);
;}

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@__const.main.m = private unnamed_addr constant [3 x float] [float 1.000000e+00, float 2.000000e+00, float 3.000000e+00], align 4
@__const.main.n = private unnamed_addr constant [3 x float] [float 4.000000e+00, float 5.000000e+00, float 6.000000e+00], align 4

define dso_local float @g(float* noalias %m, float* noalias %n) {
entry:
  %m.addr = alloca float*, align 8
  %n.addr = alloca float*, align 8
  %x = alloca float, align 4
  %y = alloca float, align 4
  store float* %m, float** %m.addr, align 8
  store float* %n, float** %n.addr, align 8
  %0 = load float*, float** %m.addr, align 8
  %1 = load float*, float** %n.addr, align 8
  %call = call float @cblas_sdot(i32 3, float* %0, i32 1, float* %1, i32 1)
  store float %call, float* %x, align 4
  %2 = load float, float* %x, align 4
  %3 = load float, float* %x, align 4
  %mul = fmul float %2, %3
  store float %mul, float* %y, align 4
  %4 = load float, float* %y, align 4
  ret float %4
}

declare dso_local float @cblas_sdot(i32, float*, i32, float*, i32)

define dso_local i32 @main() {
entry:
  %m = alloca [3 x float], align 4
  %n = alloca [3 x float], align 4
  %m1 = alloca [3 x float], align 4
  %n1 = alloca [3 x float], align 4
  %z = alloca float, align 4
  %0 = bitcast [3 x float]* %m to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %0, i8* align 4 bitcast ([3 x float]* @__const.main.m to i8*), i64 12, i1 false)
  %1 = bitcast [3 x float]* %n to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %1, i8* align 4 bitcast ([3 x float]* @__const.main.n to i8*), i64 12, i1 false)
  %2 = bitcast [3 x float]* %m1 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 4 %2, i8 0, i64 12, i1 false)
  %3 = bitcast [3 x float]* %n1 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 4 %3, i8 0, i64 12, i1 false)
  %arraydecay = getelementptr inbounds [3 x float], [3 x float]* %m, i32 0, i32 0
  %arraydecay1 = getelementptr inbounds [3 x float], [3 x float]* %m1, i32 0, i32 0
  %arraydecay2 = getelementptr inbounds [3 x float], [3 x float]* %n, i32 0, i32 0
  %arraydecay3 = getelementptr inbounds [3 x float], [3 x float]* %n1, i32 0, i32 0
  %call = call float @__enzyme_autodiff(float* bitcast (float (float*, float*)* @g to float*), float* %arraydecay, float* %arraydecay1, float* %arraydecay2, float* %arraydecay3)
  store float %call, float* %z, align 4
  ret i32 0
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1)

declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1)

declare dso_local float @__enzyme_autodiff(float*, float*, float*, float*, float*)

;CHECK:define internal void @diffeg(float* noalias %m, float* %"m'", float* noalias %n, float* %"n'", float %differeturn)
;CHECK-NEXT:entry:
;CHECK-NEXT:  %call = call float @cblas_sdot(i32 3, float* nocapture readonly %m, i32 1, float* nocapture readonly %n, i32 1)
;CHECK-NEXT:  %m0diffecall = fmul fast float %differeturn, %call
;CHECK-NEXT:  %m1diffecall = fmul fast float %differeturn, %call
;CHECK-NEXT:  %0 = fadd fast float %m0diffecall, %m1diffecall
;CHECK-NEXT:  call void @cblas_saxpy(i32 3, float %0, float* %m, i32 1, float* %"n'", i32 1)
;CHECK-NEXT:  call void @cblas_saxpy(i32 3, float %0, float* %n, i32 1, float* %"m'", i32 1)
;CHECK-NEXT:  ret void
;CHECK-NEXT:}

;CHECK:declare void @cblas_saxpy(i32, float, float*, i32, float*, i32)
