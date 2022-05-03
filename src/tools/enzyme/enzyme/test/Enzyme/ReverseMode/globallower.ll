; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-lower-globals -mem2reg -sroa -simplifycfg -instsimplify -S | FileCheck %s

@global = external dso_local local_unnamed_addr global double, align 8

; Function Attrs: noinline norecurse nounwind readonly uwtable
define double @mulglobal(double %x) {
entry:
  %l1 = load double, double* @global, align 8
  %mul = fmul fast double %l1, %x
  store double %mul, double* @global, align 8
  %l2 = load double, double* @global, align 8
  %mul2 = fmul fast double %l2, %l2
  store double %mul2, double* @global, align 8
  %l3 = load double, double* @global, align 8
  ret double %l3
}

; Function Attrs: noinline nounwind uwtable
define double @derivative(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_autodiff(double (double)* nonnull @mulglobal, double %x)
  ret double %0
}

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double)*, ...)

; CHECK: define internal { double } @diffemulglobal(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[copyload:.+]] = load double, double* @global, align 8
; CHECK-NEXT:   %mul = fmul fast double %[[copyload]], %x
; CHECK-NEXT:   %mul2 = fmul fast double %mul, %mul
; CHECK-NEXT:   store double %mul2, double* @global, align 8
; CHECK-NEXT:   %m0diffemul = fmul fast double %differeturn, %mul
; CHECK-NEXT:   %m1diffemul = fmul fast double %differeturn, %mul
; CHECK-NEXT:   %[[add:.+]] = fadd fast double %m0diffemul, %m1diffemul
; CHECK-NEXT:   %m1diffex = fmul fast double %[[add]], %[[copyload]]
; CHECK-NEXT:   %[[res:.+]] = insertvalue { double } undef, double %m1diffex, 0
; CHECK-NEXT:   ret { double } %[[res]]
; CHECK-NEXT: }