; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -simplifycfg -early-cse -S | FileCheck %s

%struct.Gradients = type { [2 x double] }

declare %struct.Gradients @__enzyme_autodiff(double (double)*, ...)

define double @square(double %x) {
entry:
  %mul = fmul fast double %x, %x
  ret double %mul
}

define %struct.Gradients @dsquare(double %x) {
entry:
  %0 = tail call %struct.Gradients (double (double)*, ...) @__enzyme_autodiff(double (double)* nonnull @square, metadata !"enzyme_width", i64 2, double %x)
  ret %struct.Gradients %0
}


; CHECK: define internal { [2 x double] } @diffe2square(double %x, [2 x double] %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"x'de" = alloca [2 x double]
; CHECK-NEXT:   store [2 x double] zeroinitializer, [2 x double]* %"x'de"
; CHECK-NEXT:   %0 = extractvalue [2 x double] %differeturn, 0
; CHECK-NEXT:   %m0diffex = fmul fast double %0, %x
; CHECK-NEXT:   %1 = insertvalue [2 x double] undef, double %m0diffex, 0
; CHECK-NEXT:   %2 = extractvalue [2 x double] %differeturn, 1
; CHECK-NEXT:   %m0diffex1 = fmul fast double %2, %x
; CHECK-NEXT:   %3 = insertvalue [2 x double] %1, double %m0diffex1, 1
; CHECK-NEXT:   %4 = getelementptr inbounds [2 x double], [2 x double]* %"x'de", i32 0, i32 0
; CHECK-NEXT:   %5 = load double, double* %4
; CHECK-NEXT:   %6 = fadd fast double %5, %m0diffex
; CHECK-NEXT:   store double %6, double* %4
; CHECK-NEXT:   %7 = getelementptr inbounds [2 x double], [2 x double]* %"x'de", i32 0, i32 1
; CHECK-NEXT:   %8 = load double, double* %7
; CHECK-NEXT:   %9 = fadd fast double %8, %m0diffex1
; CHECK-NEXT:   store double %9, double* %7
; CHECK-NEXT:   %10 = load double, double* %4
; CHECK-NEXT:   %11 = fadd fast double %10, %m0diffex
; CHECK-NEXT:   store double %11, double* %4
; CHECK-NEXT:   %12 = load double, double* %7
; CHECK-NEXT:   %13 = fadd fast double %12, %m0diffex1
; CHECK-NEXT:   store double %13, double* %7
; CHECK-NEXT:   %14 = load [2 x double], [2 x double]* %"x'de"
; CHECK-NEXT:   %15 = insertvalue { [2 x double] } undef, [2 x double] %14, 0
; CHECK-NEXT:   ret { [2 x double] } %15
; CHECK-NEXT: }