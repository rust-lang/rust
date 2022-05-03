;RUN: if [ %llvmver -ge 10 && %llvmver -le 12 ]; then %clang %s -Xclang -load -Xclang %loadBC -S -emit-llvm -o - | %FileCheck %s; fi
;RUN: if [ %llvmver -ge 12 ]; then %clang %s -fno-experimental-new-pass-manager -Xclang -load -Xclang %loadBC -S -emit-llvm -o - | %FileCheck %s; fi

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind optnone uwtable
define dso_local double @g(double* %m, double* %n) {
entry:
  %m.addr = alloca double*, align 8
  %n.addr = alloca double*, align 8
  %x = alloca double, align 8
  %y = alloca double, align 8
  store double* %m, double** %m.addr, align 8
  store double* %n, double** %n.addr, align 8
  %0 = load double*, double** %m.addr, align 8
  %1 = load double*, double** %n.addr, align 8
  %call = call double @cblas_ddot(i32 3, double* %0, i32 1, double* %1, i32 1)
  store double %call, double* %x, align 8
  %2 = load double*, double** %m.addr, align 8
  %arrayidx = getelementptr inbounds double, double* %2, i64 0
  store double 1.100000e+01, double* %arrayidx, align 8
  %3 = load double*, double** %m.addr, align 8
  %arrayidx1 = getelementptr inbounds double, double* %3, i64 1
  store double 1.200000e+01, double* %arrayidx1, align 8
  %4 = load double*, double** %m.addr, align 8
  %arrayidx2 = getelementptr inbounds double, double* %4, i64 2
  store double 1.300000e+01, double* %arrayidx2, align 8
  %5 = load double, double* %x, align 8
  %6 = load double, double* %x, align 8
  %mul = fmul double %5, %6
  store double %mul, double* %y, align 8
  %7 = load double, double* %y, align 8
  ret double %7
}

declare dso_local double @cblas_ddot(i32, double*, i32, double*, i32)

;CHECK: define internal double @cblas_ddot
