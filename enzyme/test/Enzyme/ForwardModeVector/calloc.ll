; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

@enzyme_dupnoneed = dso_local global i32 0, align 4

%struct.Gradients = type { double, double, double }

define dso_local double @f(double %x, i64 %arg) {
entry:
  %call = call noalias i8* @calloc(i64 8, i64 %arg)
  %0 = bitcast i8* %call to double*
  store double %x, double* %0, align 8
  %1 = load double, double* %0, align 8
  ret double %1
}

declare dso_local noalias i8* @calloc(i64, i64)

define dso_local %struct.Gradients @df(double %x) {
entry:
  %x.addr = alloca double, align 8
  store double %x, double* %x.addr, align 8
  %0 = load i32, i32* @enzyme_dupnoneed, align 4
  %1 = load double, double* %x.addr, align 8
  %call = call %struct.Gradients (i8*, ...) @__enzyme_fwddiff(i8* bitcast (double (double,i64)* @f to i8*), metadata !"enzyme_width", i32 3, i32 %0, double %1, double 1.0, double 2.0, double 3.0, i64 1)
  ret %struct.Gradients %call
}

declare dso_local %struct.Gradients @__enzyme_fwddiff(i8*, ...)


; CHECK: define internal [3 x double] @fwddiffe3f(double %x, [3 x double] %"x'", i64 %arg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %call = call noalias i8* @calloc(i64 8, i64 %arg)
; CHECK-NEXT:   %0 = call noalias i8* @calloc(i64 8, i64 %arg)
; CHECK-NEXT:   %1 = call noalias i8* @calloc(i64 8, i64 %arg)
; CHECK-NEXT:   %2 = call noalias i8* @calloc(i64 8, i64 %arg)
; CHECK-NEXT:   %"'ipc" = bitcast i8* %0 to double*
; CHECK-NEXT:   %"'ipc1" = bitcast i8* %1 to double*
; CHECK-NEXT:   %"'ipc2" = bitcast i8* %2 to double*
; CHECK-NEXT:   %3 = bitcast i8* %call to double*
; CHECK-NEXT:   store double %x, double* %3, align 8
; CHECK-NEXT:   %4 = extractvalue [3 x double] %"x'", 0
; CHECK-NEXT:   store double %4, double* %"'ipc", align 8
; CHECK-NEXT:   %5 = extractvalue [3 x double] %"x'", 1
; CHECK-NEXT:   store double %5, double* %"'ipc1", align 8
; CHECK-NEXT:   %6 = extractvalue [3 x double] %"x'", 2
; CHECK-NEXT:   store double %6, double* %"'ipc2", align 8
; CHECK-NEXT:   %7 = load double, double* %"'ipc", align 8
; CHECK-NEXT:   %8 = insertvalue [3 x double] undef, double %7, 0
; CHECK-NEXT:   %9 = load double, double* %"'ipc1", align 8
; CHECK-NEXT:   %10 = insertvalue [3 x double] %8, double %9, 1
; CHECK-NEXT:   %11 = load double, double* %"'ipc2", align 8
; CHECK-NEXT:   %12 = insertvalue [3 x double] %10, double %11, 2
; CHECK-NEXT:   ret [3 x double] %12
; CHECK-NEXT: }