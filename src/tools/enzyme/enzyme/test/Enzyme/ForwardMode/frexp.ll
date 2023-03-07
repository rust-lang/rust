; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

declare double @frexp(double, i32*)
declare double @__enzyme_fwddiff(i8*, ...)
declare float @__enzyme_fwddifff(i8*, ...)

define double @test(double %x) {
entry:
  %exp = alloca i32, align 4
  %call = call double @frexp(double %x, i32* %exp)
  ret double %call
}

define double @dtest(double %x, double %dx) {
entry:
  %call = call double (i8*, ...) @__enzyme_fwddiff(i8* bitcast (double (double)* @test to i8*), double %x, double %dx)
  ret double %call
}

declare float @frexpf(float, i32*)
define float @testf(float %x) {
entry:
  %exp = alloca i32, align 4
  %call = call float @frexpf(float %x, i32* %exp)
  ret float %call
}

define float @dtestf(float %x, float %dx) {
entry:
  %call = call float (i8*, ...) @__enzyme_fwddifff(i8* bitcast (float (float)* @testf to i8*), float %x, float %dx)
  ret float %call
}

; CHECK: define internal double @fwddiffetest(double %x, double %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %exp = alloca i32, align 4
; CHECK-NEXT:   %call = call double @frexp(double %x, i32* writeonly %exp)
; CHECK-NEXT:   %0 = bitcast double %x to i64
; CHECK-NEXT:   %1 = and i64 %0, 9218868437227405312
; CHECK-NEXT:   %2 = bitcast i64 %1 to double
; CHECK-NEXT:   %3 = fmul fast double %2, 2.000000e+00
; CHECK-NEXT:   %4 = fdiv fast double %"x'", %3
; CHECK-NEXT:   ret double %4
; CHECK-NEXT: }

; CHECK: define internal float @fwddiffetestf(float %x, float %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %exp = alloca i32, align 4
; CHECK-NEXT:   %call = call float @frexpf(float %x, i32* writeonly %exp)
; CHECK-NEXT:   %0 = bitcast float %x to i32
; CHECK-NEXT:   %1 = and i32 %0, 2139095040
; CHECK-NEXT:   %2 = bitcast i32 %1 to float
; CHECK-NEXT:   %3 = fmul fast float %2, 2.000000e+00
; CHECK-NEXT:   %4 = fdiv fast float %"x'", %3
; CHECK-NEXT:   ret float %4
; CHECK-NEXT: }
