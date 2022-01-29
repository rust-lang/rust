;RUN: %opt < %s %loadEnzyme -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

;#include <cblas.h>
;
;extern double __enzyme_autodiff(double*, double*, double*);
;
;double g(double *restrict m) {
;    double n[3] = {4, 5, 6};
;	 double x = cblas_ddot(3, m, 1, n, 1);
;    double y = x*x;
;	 return y;
;}
;
;int main() {
;	 double m[3] = {1, 2, 3};
;	 double m1[3] = {0.};
;	 double z = __enzyme_autodiff((double*)g, m, m1);
;}

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@__const.g.n = private unnamed_addr constant [3 x double] [double 4.000000e+00, double 5.000000e+00, double 6.000000e+00], align 16
@__const.main.m = private unnamed_addr constant [3 x double] [double 1.000000e+00, double 2.000000e+00, double 3.000000e+00], align 16

define dso_local double @g(double* noalias %m) {
entry:
  %m.addr = alloca double*, align 8
  %n = alloca [3 x double], align 16
  %x = alloca double, align 8
  %y = alloca double, align 8
  store double* %m, double** %m.addr, align 8
  %0 = bitcast [3 x double]* %n to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %0, i8* align 16 bitcast ([3 x double]* @__const.g.n to i8*), i64 24, i1 false)
  %1 = load double*, double** %m.addr, align 8
  %arraydecay = getelementptr inbounds [3 x double], [3 x double]* %n, i32 0, i32 0
  %call = call double @cblas_ddot(i32 3, double* %1, i32 1, double* %arraydecay, i32 1)
  store double %call, double* %x, align 8
  %2 = load double, double* %x, align 8
  %3 = load double, double* %x, align 8
  %mul = fmul double %2, %3
  store double %mul, double* %y, align 8
  %4 = load double, double* %y, align 8
  ret double %4
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1)

declare dso_local double @cblas_ddot(i32, double*, i32, double*, i32)

define dso_local i32 @main() {
entry:
  %m = alloca [3 x double], align 16
  %m1 = alloca [3 x double], align 16
  %z = alloca double, align 8
  %0 = bitcast [3 x double]* %m to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %0, i8* align 16 bitcast ([3 x double]* @__const.main.m to i8*), i64 24, i1 false)
  %1 = bitcast [3 x double]* %m1 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 16 %1, i8 0, i64 24, i1 false)
  %arraydecay = getelementptr inbounds [3 x double], [3 x double]* %m, i32 0, i32 0
  %arraydecay1 = getelementptr inbounds [3 x double], [3 x double]* %m1, i32 0, i32 0
  %call = call double @__enzyme_autodiff(double* bitcast (double (double*)* @g to double*), double* %arraydecay, double* %arraydecay1)
  store double %call, double* %z, align 8
  ret i32 0
}

declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1)

declare dso_local double @__enzyme_autodiff(double*, double*, double*)

;CHECK:define internal void @diffeg(double* noalias %m, double* %"m'", double %differeturn)
;CHECK-NEXT:entry:
;CHECK-NEXT:  %n = alloca [3 x double], align 16
;CHECK-NEXT:  %0 = bitcast [3 x double]* %n to i8*
;CHECK-NEXT:  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %0, i8* align 16 bitcast ([3 x double]* @__const.g.n to i8*), i64 24, i1 false)
;CHECK-NEXT:  %arraydecay = getelementptr inbounds [3 x double], [3 x double]* %n, i32 0, i32 0
;CHECK-NEXT:  %call = call double @cblas_ddot(i32 3, double* nocapture readonly %m, i32 1, double* nocapture readonly %arraydecay, i32 1)
;CHECK-NEXT:  %m0diffecall = fmul fast double %differeturn, %call
;CHECK-NEXT:  %m1diffecall = fmul fast double %differeturn, %call
;CHECK-NEXT:  %1 = fadd fast double %m0diffecall, %m1diffecall
;CHECK-NEXT:  call void @cblas_daxpy(i32 3, double %1, double* %arraydecay, i32 1, double* %"m'", i32 1)
;CHECK-NEXT:  ret void
;CHECK-NEXT:}

;CHECK: declare void @cblas_daxpy(i32, double, double*, i32, double*, i32)
