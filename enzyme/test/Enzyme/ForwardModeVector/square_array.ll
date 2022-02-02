; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -simplifycfg -early-cse -S | FileCheck %s

%struct.Gradients = type { { double, double }, { double, double }, { double, double } }

define { double, double } @squared(double %x) {
entry:
  %mul = fmul double %x, %x
  %mul2 = fmul double %mul, %x
  %.fca.0.insert = insertvalue { double, double } undef, double %mul, 0
  %.fca.1.insert = insertvalue { double, double } %.fca.0.insert, double %mul2, 1
  ret { double, double } %.fca.1.insert
}

define %struct.Gradients @dsquared(double %x) {
entry:
  %call = call %struct.Gradients (i8*, ...) @__enzyme_fwddiff(i8* bitcast ({ double, double } (double)* @squared to i8*), metadata !"enzyme_width", i64 3, double %x, double 1.0, double 2.0, double 3.0)
  ret %struct.Gradients %call
}

declare %struct.Gradients @__enzyme_fwddiff(i8*, ...)


; CHECK: define internal [3 x { double, double }] @fwddiffe3squared(double %x, [3 x double] %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %mul = fmul double %x, %x
; CHECK-NEXT:   %0 = extractvalue [3 x double] %"x'", 0
; CHECK-NEXT:   %1 = fmul fast double %0, %x
; CHECK-NEXT:   %2 = fadd fast double %1, %1
; CHECK-NEXT:   %3 = insertvalue [3 x double] undef, double %2, 0
; CHECK-NEXT:   %4 = extractvalue [3 x double] %"x'", 1
; CHECK-NEXT:   %5 = fmul fast double %4, %x
; CHECK-NEXT:   %6 = fadd fast double %5, %5
; CHECK-NEXT:   %7 = insertvalue [3 x double] %3, double %6, 1
; CHECK-NEXT:   %8 = extractvalue [3 x double] %"x'", 2
; CHECK-NEXT:   %9 = fmul fast double %8, %x
; CHECK-NEXT:   %10 = fadd fast double %9, %9
; CHECK-NEXT:   %11 = insertvalue [3 x double] %7, double %10, 2
; CHECK-NEXT:   %12 = fmul fast double %2, %x
; CHECK-NEXT:   %13 = fmul fast double %0, %mul
; CHECK-NEXT:   %14 = fadd fast double %12, %13
; CHECK-NEXT:   %15 = insertvalue [3 x double] undef, double %14, 0
; CHECK-NEXT:   %16 = fmul fast double %6, %x
; CHECK-NEXT:   %17 = fmul fast double %4, %mul
; CHECK-NEXT:   %18 = fadd fast double %16, %17
; CHECK-NEXT:   %19 = insertvalue [3 x double] %15, double %18, 1
; CHECK-NEXT:   %20 = fmul fast double %10, %x
; CHECK-NEXT:   %21 = fmul fast double %8, %mul
; CHECK-NEXT:   %22 = fadd fast double %20, %21
; CHECK-NEXT:   %23 = insertvalue [3 x double] %19, double %22, 2
; CHECK-NEXT:   %24 = insertvalue { double, double } zeroinitializer, double %2, 0
; CHECK-NEXT:   %25 = insertvalue [3 x { double, double }] undef, { double, double } %24, 0
; CHECK-NEXT:   %26 = insertvalue { double, double } zeroinitializer, double %6, 0
; CHECK-NEXT:   %27 = insertvalue [3 x { double, double }] %25, { double, double } %26, 1
; CHECK-NEXT:   %28 = insertvalue { double, double } zeroinitializer, double %10, 0
; CHECK-NEXT:   %29 = insertvalue [3 x { double, double }] %27, { double, double } %28, 2
; CHECK-NEXT:   %30 = insertvalue { double, double } %24, double %14, 1
; CHECK-NEXT:   %31 = insertvalue [3 x { double, double }] undef, { double, double } %30, 0
; CHECK-NEXT:   %32 = insertvalue { double, double } %26, double %18, 1
; CHECK-NEXT:   %33 = insertvalue [3 x { double, double }] %31, { double, double } %32, 1
; CHECK-NEXT:   %34 = insertvalue { double, double } %28, double %22, 1
; CHECK-NEXT:   %35 = insertvalue [3 x { double, double }] %33, { double, double } %34, 2
; CHECK-NEXT:   ret [3 x { double, double }] %35
; CHECK-NEXT: }