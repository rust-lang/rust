; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

declare double @scalbn(double, i32)
declare double @__enzyme_autodiff(i8*, ...)

define double @test(double %x, i32 %exp) {
entry:
  %call = call double @scalbn(double %x, i32 %exp)
  ret double %call
}

define double @dtest(double %x, i32 %exp) {
entry:
  %call = call double (i8*, ...) @__enzyme_autodiff(i8* bitcast (double (double, i32)* @test to i8*), double %x, i32 %exp)
  ret double %call
}


; CHECK: define internal { double } @diffetest(double %x, i32 %exp, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call fast double @scalbn(double %x, i32 %exp)
; CHECK-NEXT:   %1 = call fast double @scalbn(double %differeturn, i32 %exp)
; CHECK-NEXT:   %2 = fmul fast double %0, 0x3FD3441350A96098
; CHECK-NEXT:   %3 = fadd fast double %2, %1
; CHECK-NEXT:   %4 = insertvalue { double } undef, double %3, 0
; CHECK-NEXT:   ret { double } %4
; CHECK-NEXT: }