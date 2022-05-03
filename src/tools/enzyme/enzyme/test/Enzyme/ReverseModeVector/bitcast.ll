; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -simplifycfg -adce -instsimplify -S | FileCheck %s

%struct.Gradients = type { [2 x double] }

define double @tester(double %x) {
entry:
  %y = bitcast double %x to i64
  %z = bitcast i64 %y to double
  ret double %z
}

define %struct.Gradients @test_derivative(double %x) {
entry:
  %call = call %struct.Gradients (double (double)*, ...) @__enzyme_autodiff(double (double)* nonnull @tester, metadata !"enzyme_width", i64 2, double %x)
  ret %struct.Gradients %call
}

declare %struct.Gradients @__enzyme_autodiff(double (double)*, ...)


; CHECK: define internal { [2 x double] } @diffe2tester(double %x, [2 x double] %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"y'de" = alloca [2 x i64]
; CHECK-NEXT:   store [2 x i64] zeroinitializer, [2 x i64]* %"y'de"
; CHECK-NEXT:   %"x'de" = alloca [2 x double]
; CHECK-NEXT:   store [2 x double] zeroinitializer, [2 x double]* %"x'de"
; CHECK-NEXT:   %0 = extractvalue [2 x double] %differeturn, 0
; CHECK-NEXT:   %1 = extractvalue [2 x double] %differeturn, 1
; CHECK-NEXT:   %2 = getelementptr inbounds [2 x i64], [2 x i64]* %"y'de", i32 0, i32 0
; CHECK-NEXT:   %3 = load i64, i64* %2
; CHECK-NEXT:   %4 = bitcast i64 %3 to double
; CHECK-NEXT:   %5 = fadd fast double %4, %0
; CHECK-NEXT:   %6 = bitcast double %5 to i64
; CHECK-NEXT:   store i64 %6, i64* %2
; CHECK-NEXT:   %7 = getelementptr inbounds [2 x i64], [2 x i64]* %"y'de", i32 0, i32 1
; CHECK-NEXT:   %8 = load i64, i64* %7
; CHECK-NEXT:   %9 = bitcast i64 %8 to double
; CHECK-NEXT:   %10 = fadd fast double %9, %1
; CHECK-NEXT:   %11 = bitcast double %10 to i64
; CHECK-NEXT:   store i64 %11, i64* %7
; CHECK-NEXT:   %12 = load [2 x i64], [2 x i64]* %"y'de"
; CHECK-NEXT:   %13 = extractvalue [2 x i64] %12, 0
; CHECK-NEXT:   %14 = bitcast i64 %13 to double
; CHECK-NEXT:   %15 = extractvalue [2 x i64] %12, 1
; CHECK-NEXT:   %16 = bitcast i64 %15 to double
; CHECK-NEXT:   %17 = getelementptr inbounds [2 x double], [2 x double]* %"x'de", i32 0, i32 0
; CHECK-NEXT:   %18 = load double, double* %17
; CHECK-NEXT:   %19 = fadd fast double %18, %14
; CHECK-NEXT:   store double %19, double* %17
; CHECK-NEXT:   %20 = getelementptr inbounds [2 x double], [2 x double]* %"x'de", i32 0, i32 1
; CHECK-NEXT:   %21 = load double, double* %20
; CHECK-NEXT:   %22 = fadd fast double %21, %16
; CHECK-NEXT:   store double %22, double* %20
; CHECK-NEXT:   store [2 x i64] zeroinitializer, [2 x i64]* %"y'de"
; CHECK-NEXT:   %23 = load [2 x double], [2 x double]* %"x'de"
; CHECK-NEXT:   %24 = insertvalue { [2 x double] } undef, [2 x double] %23, 0
; CHECK-NEXT:   ret { [2 x double] } %24
; CHECK-NEXT: }