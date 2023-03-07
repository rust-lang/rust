; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

declare double @scalbn(double, i32)
declare double @__enzyme_fwddiff(i8*, ...)

define double @test(double %x, i32 %exp) {
entry:
  %call = call double @scalbn(double %x, i32 %exp)
  ret double %call
}

define double @dtest(double %x, double %dx, i32 %exp) {
entry:
  %call = call double (i8*, ...) @__enzyme_fwddiff(i8* bitcast (double (double, i32)* @test to i8*), double %x, double %dx, i32 %exp)
  ret double %call
}


; CHECK: define internal double @fwddiffetest(double %x, double %"x'", i32 %exp)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call fast double @scalbn(double %x, i32 %exp)
; CHECK-NEXT:   %1 = call fast double @scalbn(double %"x'", i32 %exp)
; CHECK-NEXT:   %2 = fmul fast double %0, 0x3FD3441350A96098
; CHECK-NEXT:   %3 = fadd fast double %2, %1
; CHECK-NEXT:   ret double %3
; CHECK-NEXT: }
