; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S -simplifycfg -mem2reg -instsimplify | FileCheck %s

declare double @frexp(double, i32*)
declare double @__enzyme_autodiff(i8*, ...)
declare float @__enzyme_autodifff(i8*, ...)

define double @test(double %x) {
entry:
  %exp = alloca i32, align 4
  %call = call double @frexp(double %x, i32* %exp)
  ret double %call
}

define double @dtest(double %x) {
entry:
  %call = call double (i8*, ...) @__enzyme_autodiff(i8* bitcast (double (double)* @test to i8*), double %x)
  ret double %call
}

declare float @frexpf(float, i32*)
define float @testf(float %x) {
entry:
  %exp = alloca i32, align 4
  %call = call float @frexpf(float %x, i32* %exp)
  ret float %call
}

define float @dtestf(float %x) {
entry:
  %call = call float (i8*, ...) @__enzyme_autodifff(i8* bitcast (float (float)* @testf to i8*), float %x)
  ret float %call
}

; CHECK: define internal { double } @diffetest(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %exp = alloca i32, align 4
; CHECK-NEXT:   %call = call double @frexp(double %x, i32* writeonly %exp)
; CHECK-NEXT:   %0 = bitcast double %x to i64
; CHECK-NEXT:   %1 = and i64 %0, 9218868437227405312
; CHECK-NEXT:   %2 = bitcast i64 %1 to double
; CHECK-NEXT:   %3 = fmul fast double %2, 2.000000e+00
; CHECK-NEXT:   %4 = fdiv fast double %differeturn, %3
; CHECK-NEXT:   %5 = insertvalue { double } {{(undef|poison)}}, double %4, 0
; CHECK-NEXT:   ret { double } %5
; CHECK-NEXT: }

; CHECK: define internal { float } @diffetestf(float %x, float %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %exp = alloca i32, align 4
; CHECK-NEXT:   %call = call float @frexpf(float %x, i32* writeonly %exp)
; CHECK-NEXT:   %0 = bitcast float %x to i32
; CHECK-NEXT:   %1 = and i32 %0, 2139095040
; CHECK-NEXT:   %2 = bitcast i32 %1 to float
; CHECK-NEXT:   %3 = fmul fast float %2, 2.000000e+00
; CHECK-NEXT:   %4 = fdiv fast float %differeturn, %3
; CHECK-NEXT:   %5 = insertvalue { float } {{(undef|poison)}}, float %4, 0
; CHECK-NEXT:   ret { float } %5
; CHECK-NEXT: }
