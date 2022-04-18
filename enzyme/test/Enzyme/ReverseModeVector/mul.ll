; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -early-cse -simplifycfg -S | FileCheck %s

%struct.Gradients = type { [2 x double], [2 x double] }

; Function Attrs: nounwind
declare %struct.Gradients @__enzyme_autodiff(double (double, double)*, ...)

; Function Attrs: noinline nounwind readnone uwtable
define double @tester(double %x, double %y) {
entry:
  %0 = fmul fast double %x, %y
  ret double %0
}

define %struct.Gradients @test_derivative(double %x, double %y) {
entry:
  %0 = tail call %struct.Gradients (double (double, double)*, ...) @__enzyme_autodiff(double (double, double)* nonnull @tester, metadata !"enzyme_width", i64 2, double %x, double %y)
  ret %struct.Gradients %0
}


; CHECK: define internal { [2 x double], [2 x double] } @diffe2tester(double %x, double %y, [2 x double] %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"x'de" = alloca [2 x double]
; CHECK-NEXT:   store [2 x double] zeroinitializer, [2 x double]* %"x'de"
; CHECK-NEXT:   %"y'de" = alloca [2 x double]
; CHECK-NEXT:   store [2 x double] zeroinitializer, [2 x double]* %"y'de"
; CHECK-NEXT:   %0 = extractvalue [2 x double] %differeturn, 0
; CHECK-NEXT:   %m0diffex = fmul fast double %0, %y
; CHECK-NEXT:   %1 = insertvalue [2 x double] undef, double %m0diffex, 0
; CHECK-NEXT:   %2 = extractvalue [2 x double] %differeturn, 1
; CHECK-NEXT:   %m0diffex1 = fmul fast double %2, %y
; CHECK-NEXT:   %3 = insertvalue [2 x double] %1, double %m0diffex1, 1
; CHECK-NEXT:   %m1diffey = fmul fast double %0, %x
; CHECK-NEXT:   %4 = insertvalue [2 x double] undef, double %m1diffey, 0
; CHECK-NEXT:   %m1diffey2 = fmul fast double %2, %x
; CHECK-NEXT:   %5 = insertvalue [2 x double] %4, double %m1diffey2, 1
; CHECK-NEXT:   %6 = getelementptr inbounds [2 x double], [2 x double]* %"x'de", i32 0, i32 0
; CHECK-NEXT:   %7 = load double, double* %6
; CHECK-NEXT:   %8 = fadd fast double %7, %m0diffex
; CHECK-NEXT:   store double %8, double* %6
; CHECK-NEXT:   %9 = getelementptr inbounds [2 x double], [2 x double]* %"x'de", i32 0, i32 1
; CHECK-NEXT:   %10 = load double, double* %9
; CHECK-NEXT:   %11 = fadd fast double %10, %m0diffex1
; CHECK-NEXT:   store double %11, double* %9
; CHECK-NEXT:   %12 = getelementptr inbounds [2 x double], [2 x double]* %"y'de", i32 0, i32 0
; CHECK-NEXT:   %13 = load double, double* %12
; CHECK-NEXT:   %14 = fadd fast double %13, %m1diffey
; CHECK-NEXT:   store double %14, double* %12
; CHECK-NEXT:   %15 = getelementptr inbounds [2 x double], [2 x double]* %"y'de", i32 0, i32 1
; CHECK-NEXT:   %16 = load double, double* %15
; CHECK-NEXT:   %17 = fadd fast double %16, %m1diffey2
; CHECK-NEXT:   store double %17, double* %15
; CHECK-NEXT:   %18 = load [2 x double], [2 x double]* %"x'de"
; CHECK-NEXT:   %19 = load [2 x double], [2 x double]* %"y'de"
; CHECK-NEXT:   %20 = insertvalue { [2 x double], [2 x double] } undef, [2 x double] %18, 0
; CHECK-NEXT:   %21 = insertvalue { [2 x double], [2 x double] } %20, [2 x double] %19, 1
; CHECK-NEXT:   ret { [2 x double], [2 x double] } %21
; CHECK-NEXT: }