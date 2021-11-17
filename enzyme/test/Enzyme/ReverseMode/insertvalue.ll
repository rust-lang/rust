; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

; Function Attrs: noinline nounwind readnone uwtable
define double @tester(double %x) {
entry:
  %agg1 = insertvalue [3 x double] undef, double %x, 0
  %mul = fmul double %x, %x
  %agg2 = insertvalue [3 x double] %agg1, double %mul, 1
  %add = fadd double %mul, 2.0
  %agg3 = insertvalue [3 x double] %agg2, double %add, 2
  %res = extractvalue [3 x double] %agg2, 1
  ret double %res
}

define double @test_derivative(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_autodiff(double (double)* nonnull @tester, double %x)
  ret double %0
}

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double)*, ...)

; CHECK: define internal { double } @diffetester(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"agg2'de" = alloca [3 x double], align 8
; CHECK-NEXT:   store [3 x double] zeroinitializer, [3 x double]* %"agg2'de"
; CHECK-NEXT:   %"agg1'de" = alloca [3 x double], align 8
; CHECK-NEXT:   store [3 x double] zeroinitializer, [3 x double]* %"agg1'de"
; CHECK-NEXT:   %0 = getelementptr inbounds [3 x double], [3 x double]* %"agg2'de", i32 0, i32 1
; CHECK-NEXT:   %1 = load double, double* %0
; CHECK-NEXT:   %2 = fadd fast double %1, %differeturn
; CHECK-NEXT:   store double %2, double* %0
; CHECK-NEXT:   %3 = load [3 x double], [3 x double]* %"agg2'de"
; CHECK-NEXT:   %4 = extractvalue [3 x double] %3, 1
; CHECK-NEXT:   %5 = load [3 x double], [3 x double]* %"agg2'de"
; CHECK-NEXT:   %6 = insertvalue [3 x double] %5, double 0.000000e+00, 1
; CHECK-NEXT:   %7 = extractvalue [3 x double] %6, 0
; CHECK-NEXT:   %8 = getelementptr inbounds [3 x double], [3 x double]* %"agg1'de", i32 0, i32 0
; CHECK-NEXT:   %9 = load double, double* %8
; CHECK-NEXT:   %10 = fadd fast double %9, %7
; CHECK-NEXT:   store double %10, double* %8
; CHECK-NEXT:   %11 = getelementptr inbounds [3 x double], [3 x double]* %"agg1'de", i32 0, i32 1
; CHECK-NEXT:   %12 = load double, double* %11
; CHECK-NEXT:   store double %12, double* %11
; CHECK-NEXT:   %13 = extractvalue [3 x double] %6, 2
; CHECK-NEXT:   %14 = getelementptr inbounds [3 x double], [3 x double]* %"agg1'de", i32 0, i32 2
; CHECK-NEXT:   %15 = load double, double* %14
; CHECK-NEXT:   %16 = fadd fast double %15, %13
; CHECK-NEXT:   store double %16, double* %14
; CHECK-NEXT:   store [3 x double] zeroinitializer, [3 x double]* %"agg2'de"
; CHECK-NEXT:   %m0diffex = fmul fast double %4, %x
; CHECK-NEXT:   %m1diffex = fmul fast double %4, %x
; CHECK-NEXT:   %17 = fadd fast double %m0diffex, %m1diffex
; CHECK-NEXT:   %18 = load [3 x double], [3 x double]* %"agg1'de"
; CHECK-NEXT:   %19 = extractvalue [3 x double] %18, 0
; CHECK-NEXT:   %20 = fadd fast double %17, %19
; CHECK-NEXT:   store [3 x double] zeroinitializer, [3 x double]* %"agg1'de"
; CHECK-NEXT:   %21 = insertvalue { double } undef, double %20, 0
; CHECK-NEXT:   ret { double } %21
; CHECK-NEXT: }