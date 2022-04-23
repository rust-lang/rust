;RUN: if [ %llvmver -ge 10 && %llvmver -le 12 ]; then %clang %s -Xclang -load -Xclang %loadBC -S -emit-llvm -o - | %FileCheck %s; fi
;RUN: if [ %llvmver -ge 12 ]; then %clang %s -fno-experimental-new-pass-manager -Xclang -load -Xclang %loadBC -S -emit-llvm -o - | %FileCheck %s; fi

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind optnone uwtable
define double @caller(i32* %N, double* %X, i32* %incX, double* %Y, i32* %incY) {
entry:
  %call = call double @ddot_(i32* %N, double* %X, i32* %incX, double* %Y, i32* %incY)
  ret double %call
}

declare dso_local double @ddot_(i32* %N, double* %X, i32* %incX, double* %Y, i32* %incY)

; CHECK: define internal double @ddot_
; CHECK: call double @cblas_ddot

; CHECK: define internal double @cblas_ddot
