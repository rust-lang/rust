; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -early-cse -instsimplify -simplifycfg -S | FileCheck %s

; Function Attrs: noinline nounwind readnone uwtable
define double @tester(double* %xp) {
entry:
  fence syncscope("singlethread") seq_cst
  %x = load double, double* %xp, align 8
  fence syncscope("singlethread") seq_cst
  %x2 = fmul double %x, %x
  ret double %x2
}

define double @test_derivative(double* %x, double* %y) {
entry:
  %0 = tail call double (double (double*)*, ...) @__enzyme_autodiff(double (double*)* nonnull @tester, double* %x, double* %y)
  ret double %0
}

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double*)*, ...)

; CHECK: define internal void @diffetester(double* %xp, double* %"xp'", double %differeturn) 
; CHECK-NEXT: entry:
; CHECK-NEXT:   fence syncscope("singlethread") seq_cst
; CHECK-NEXT:   %x = load double, double* %xp, align 8
; CHECK-NEXT:   fence syncscope("singlethread") seq_cst
; CHECK-NEXT:   %m0diffex = fmul fast double %differeturn, %x
; CHECK-NEXT:   %[[i0:.+]] = fadd fast double %m0diffex, %m0diffex
; CHECK-NEXT:   fence syncscope("singlethread") seq_cst
; CHECK-NEXT:   %[[i1:.+]] = load double, double* %"xp'", align 8
; CHECK-NEXT:   %[[i2:.+]] = fadd fast double %[[i1]], %[[i0]]
; CHECK-NEXT:   store double %[[i2]], double* %"xp'", align 8
; CHECK-NEXT:   fence syncscope("singlethread") seq_cst
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
