; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -simplifycfg -instcombine -adce -S | FileCheck %s

; THIS TEST ENSURES THAT STORES TO CONSTANT MEMORY SHOULD BE CONSIDERED CONSTANT INSTRUCTIONS

define double @caller(double* %W, double* %Wp) {
entry:
  %call = call double (i8*, ...) @__enzyme_autodiff(i8* bitcast (double (double*)* @matvec to i8*), double* %W, double* %Wp)
  ret double %call
}

define double @matvec(double* %W) {
entry:
  %result = call double @subfn(double* %W, double* null)
  ret double %result
}

define double @subfn(double* %a1, double* %res) {
entry:
  %a2 = load double, double* %a1
  store double %a2, double* %res
  ret double %a2
}

declare double @__enzyme_autodiff(i8*, ...)

; CHECK: define internal void @diffematvec(double* %W, double* %"W'", double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @diffesubfn(double* %W, double* %"W'", double* null, double %differeturn)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @diffesubfn(double* %a1, double* %"a1'", double* %res, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %a2 = load double, double* %a1, align 8
; CHECK-NEXT:   store double %a2, double* %res, align 8
; CHECK-NEXT:   %0 = load double, double* %"a1'", align 8
; CHECK-NEXT:   %1 = fadd fast double %0, %differeturn
; CHECK-NEXT:   store double %1, double* %"a1'", align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
