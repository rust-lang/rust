; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -instsimplify -simplifycfg -adce -S | FileCheck %s

declare double @__enzyme_autodiff(double (double)*, ...)

declare double @j0(double)

define double @testj0(double %x) {
entry:
  %call = call double @j0(double %x)
  ret double %call
}

; CHECK: define internal { double } @diffetestj0(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call fast double @j1(double %x)
; CHECK-NEXT:   %1 = {{(fsub fast double \-?0.000000e\+00,|fneg fast double)}} %0
; CHECK-NEXT:   %2 = fmul fast double %1, %differeturn
; CHECK-NEXT:   %3 = insertvalue { double } undef, double %2, 0
; CHECK-NEXT:   ret { double } %3
; CHECK-NEXT: }


define double @test_derivativej0(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_autodiff(double (double)* nonnull @testj0, double %x)
  ret double %0
}

declare double @y0(double)

define double @testy0(double %x) {
entry:
  %call = call double @y0(double %x)
  ret double %call
}

; CHECK: define internal { double } @diffetesty0(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call fast double @y1(double %x)
; CHECK-NEXT:   %1 = {{(fsub fast double \-?0.000000e\+00,|fneg fast double)}} %0
; CHECK-NEXT:   %2 = fmul fast double %1, %differeturn
; CHECK-NEXT:   %3 = insertvalue { double } undef, double %2, 0
; CHECK-NEXT:   ret { double } %3
; CHECK-NEXT: }

define double @test_derivativey0(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_autodiff(double (double)* nonnull @testy0, double %x)
  ret double %0
}



declare double @j1(double)

define double @testj1(double %x) {
entry:
  %call = call double @j1(double %x)
  ret double %call
}

; CHECK: define internal { double } @diffetestj1(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:    %0 = call fast double @j0(double %x)
; CHECK-NEXT:    %1 = call fast double @jn(i32 2, double %x)
; CHECK-NEXT:    %2 = fsub fast double %0, %1
; CHECK-NEXT:    %3 = fmul fast double %2, 5.000000e-01
; CHECK-NEXT:    %4 = fmul fast double %3, %differeturn
; CHECK-NEXT:    %5 = insertvalue { double } undef, double %4, 0
; CHECK-NEXT:    ret { double } %5
; CHECK-NEXT: }


define double @test_derivativej1(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_autodiff(double (double)* nonnull @testj1, double %x)
  ret double %0
}

declare double @y1(double)

define double @testy1(double %x) {
entry:
  %call = call double @y1(double %x)
  ret double %call
}

; CHECK: define internal { double } @diffetesty1(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:    %0 = call fast double @y0(double %x)
; CHECK-NEXT:    %1 = call fast double @yn(i32 2, double %x)
; CHECK-NEXT:    %2 = fsub fast double %0, %1
; CHECK-NEXT:    %3 = fmul fast double %2, 5.000000e-01
; CHECK-NEXT:    %4 = fmul fast double %3, %differeturn
; CHECK-NEXT:    %5 = insertvalue { double } undef, double %4, 0
; CHECK-NEXT:    ret { double } %5
; CHECK-NEXT: }

define double @test_derivativey1(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_autodiff(double (double)* nonnull @testy1, double %x)
  ret double %0
}



declare double @__enzyme_autodiff2(double (i32, double)*, ...)

declare double @jn(i32, double)

define double @testjn(i32 %n, double %x) {
entry:
  %call = call double @jn(i32 %n, double %x)
  ret double %call
}

; CHECK: define internal { double } @diffetestjn(i32 %n, double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:    %0 = sub i32 %n, 1
; CHECK-NEXT:    %1 = call fast double @jn(i32 %0, double %x)
; CHECK-NEXT:    %2 = add i32 %n, 1
; CHECK-NEXT:    %3 = call fast double @jn(i32 %2, double %x)
; CHECK-NEXT:    %4 = fsub fast double %1, %3
; CHECK-NEXT:    %5 = fmul fast double %4, 5.000000e-01
; CHECK-NEXT:    %6 = fmul fast double %5, %differeturn
; CHECK-NEXT:    %7 = insertvalue { double } undef, double %6, 0
; CHECK-NEXT:    ret { double } %7
; CHECK-NEXT: }


define double @test_derivativejn(i32 %n, double %x) {
entry:
  %0 = tail call double (double (i32, double)*, ...) @__enzyme_autodiff2(double (i32, double)* nonnull @testjn, i32 %n, double %x)
  ret double %0
}

declare double @yn(i32, double)

define double @testyn(i32 %n, double %x) {
entry:
  %call = call double @yn(i32 %n, double %x)
  ret double %call
}

; CHECK: define internal { double } @diffetestyn(i32 %n, double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = sub i32 %n, 1
; CHECK-NEXT:   %1 = call fast double @yn(i32 %0, double %x)
; CHECK-NEXT:   %2 = add i32 %n, 1
; CHECK-NEXT:   %3 = call fast double @yn(i32 %2, double %x)
; CHECK-NEXT:   %4 = fsub fast double %1, %3
; CHECK-NEXT:   %5 = fmul fast double %4, 5.000000e-01
; CHECK-NEXT:   %6 = fmul fast double %5, %differeturn
; CHECK-NEXT:   %7 = insertvalue { double } undef, double %6, 0
; CHECK-NEXT:   ret { double } %7
; CHECK-NEXT: }

define double @test_derivativeyn(i32 %n, double %x) {
entry:
  %0 = tail call double (double (i32, double)*, ...) @__enzyme_autodiff2(double (i32, double)* nonnull @testyn, i32 %n, double %x)
  ret double %0
}
