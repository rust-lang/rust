;RUN: %opt < %s %loadEnzyme -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

;#include <cblas.h>
;
;extern double __enzyme_autodiff(void *, double *, double *, double *,
;                                 double *, int);
;
;double g(double *restrict m, double *restrict n, int stride) {
;  double x = cblas_ddot(3, m, 2, n, stride);
;  double y = x * x;
;  return y;
;}
;
;int main() {
;  double m[6] = {1, 2, 3, 101, 102, 103};
;  double m1[6] = {0, 0, 0, 0, 0, 0};
;  double n[9] = {4, 5, 6, 104, 105, 106, 7, 8, 9};
;  double n1[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
;  __enzyme_autodiff((void*)g, m, m1, n, n1, 3);
;}

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@__const.main.n = private unnamed_addr constant [9 x double] [double 4.000000e+00, double 5.000000e+00, double 6.000000e+00, double 1.040000e+02, double 1.050000e+02, double 1.060000e+02, double 7.000000e+00, double 8.000000e+00, double 9.000000e+00], align 16

define dso_local double @g(double* noalias %m, double* noalias %n, i32 %stride) {
entry:
  %m.addr = alloca double*, align 8
  %n.addr = alloca double*, align 8
  %stride.addr = alloca i32, align 4
  %x = alloca double, align 8
  %y = alloca double, align 8
  store double* %m, double** %m.addr, align 8
  store double* %n, double** %n.addr, align 8
  store i32 %stride, i32* %stride.addr, align 4
  %0 = load double*, double** %m.addr, align 8
  %1 = load double*, double** %n.addr, align 8
  %2 = load i32, i32* %stride.addr, align 4
  %call = call double @cblas_ddot(i32 3, double* %0, i32 2, double* %1, i32 %2)
  store double %call, double* %x, align 8
  %3 = load double, double* %x, align 8
  %4 = load double, double* %x, align 8
  %mul = fmul double %3, %4
  store double %mul, double* %y, align 8
  %5 = load double, double* %y, align 8
  ret double %5
}

declare dso_local double @cblas_ddot(i32, double*, i32, double*, i32)

define dso_local i32 @main() {
entry:
  %m = alloca [6 x double], align 16
  %m1 = alloca [6 x double], align 16
  %n = alloca [9 x double], align 16
  %n1 = alloca [9 x double], align 16
  %0 = bitcast [6 x double]* %m to i8*
  call void @llvm.memset.p0i8.i64(i8* align 16 %0, i8 0, i64 48, i1 false)
  %1 = bitcast i8* %0 to [6 x double]*
  %2 = getelementptr inbounds [6 x double], [6 x double]* %1, i32 0, i32 0
  store double 1.000000e+00, double* %2, align 16
  %3 = getelementptr inbounds [6 x double], [6 x double]* %1, i32 0, i32 1
  store double 2.000000e+00, double* %3, align 8
  %4 = getelementptr inbounds [6 x double], [6 x double]* %1, i32 0, i32 2
  store double 3.000000e+00, double* %4, align 16
  %5 = getelementptr inbounds [6 x double], [6 x double]* %1, i32 0, i32 3
  store double 1.010000e+02, double* %5, align 8
  %6 = getelementptr inbounds [6 x double], [6 x double]* %1, i32 0, i32 4
  store double 1.020000e+02, double* %6, align 16
  %7 = getelementptr inbounds [6 x double], [6 x double]* %1, i32 0, i32 5
  store double 1.030000e+02, double* %7, align 8
  %8 = bitcast [6 x double]* %m1 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 16 %8, i8 0, i64 48, i1 false)
  %9 = bitcast [9 x double]* %n to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %9, i8* align 16 bitcast ([9 x double]* @__const.main.n to i8*), i64 72, i1 false)
  %10 = bitcast [9 x double]* %n1 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 16 %10, i8 0, i64 72, i1 false)
  %arraydecay = getelementptr inbounds [6 x double], [6 x double]* %m, i32 0, i32 0
  %arraydecay1 = getelementptr inbounds [6 x double], [6 x double]* %m1, i32 0, i32 0
  %arraydecay2 = getelementptr inbounds [9 x double], [9 x double]* %n, i32 0, i32 0
  %arraydecay3 = getelementptr inbounds [9 x double], [9 x double]* %n1, i32 0, i32 0
  %call = call double @__enzyme_autodiff(i8* bitcast (double (double*, double*, i32)* @g to i8*), double* %arraydecay, double* %arraydecay1, double* %arraydecay2, double* %arraydecay3, i32 3)
  ret i32 0
}

declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1)

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1)

declare dso_local double @__enzyme_autodiff(i8*, double*, double*, double*, double*, i32)

;CHECK:define internal void @diffeg(double* noalias %m, double* %"m'", double* noalias %n, double* %"n'", i32 %stride, double %differeturn)
;CHECK-NEXT:entry:
;CHECK-NEXT:  %call = call double @cblas_ddot(i32 3, double* nocapture readonly %m, i32 2, double* nocapture readonly %n, i32 %stride)
;CHECK-NEXT:  %m0diffecall = fmul fast double %differeturn, %call
;CHECK-NEXT:  %m1diffecall = fmul fast double %differeturn, %call
;CHECK-NEXT:  %0 = fadd fast double %m0diffecall, %m1diffecall
;CHECK-NEXT:  call void @cblas_daxpy(i32 3, double %0, double* %m, i32 2, double* %"n'", i32 %stride)
;CHECK-NEXT:  call void @cblas_daxpy(i32 3, double %0, double* %n, i32 %stride, double* %"m'", i32 2)
;CHECK-NEXT:  ret void
;CHECK-NEXT:}

;CHECK:declare void @cblas_daxpy(i32, double, double*, i32, double*, i32)
