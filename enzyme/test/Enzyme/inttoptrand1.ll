; RUN: %opt < %s %loadEnzyme -enzyme -enzyme_preopt=false -mem2reg -sroa -simplifycfg -instcombine -early-cse -adce -S | FileCheck %s

define double @caller(i64* %A, i64* %Ap, double* %res, double* %resp) {
entry:
  %call = call double (i8*, ...) @__enzyme_autodiff(i8* bitcast (double (i64*, double*)* @matvec to i8*), metadata !"diffe_dup", i64* %A, i64* %Ap, double* %res, double* %resp)
  ret double %call
}

declare double @__enzyme_autodiff(i8*, ...)

define internal double @matvec(i64* %lhs, double* %res) {
entry:
  %loaded = load i64, i64* %lhs, align 4
  %a2 = inttoptr i64 %loaded to double*
  %div = lshr i64 %loaded, 3
  %and = and i64 %div, 1
  %gep = getelementptr inbounds double, double* %a2, i64 %and
  %a4 = load double, double* %gep, align 8
  store double %a4, double* %res, align 8
  ret double %a4
}

; CHECK: define internal void @diffematvec(i64* %lhs, i64* %"lhs'", double* %res, double* %"res'", double %differeturn) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = bitcast i64* %"lhs'" to double**
; CHECK-NEXT:   %"loaded'ipl1" = load double*, double** %0, align 4
; CHECK-NEXT:   %loaded = load i64, i64* %lhs, align 4
; CHECK-NEXT:   %a2 = inttoptr i64 %loaded to double*
; CHECK-NEXT:   %div = lshr i64 %loaded, 3
; CHECK-NEXT:   %and = and i64 %div, 1
; CHECK-NEXT:   %[[gepipge:.+]] = getelementptr inbounds double, double* %"loaded'ipl1", i64 %and
; CHECK-NEXT:   %gep = getelementptr inbounds double, double* %a2, i64 %and
; CHECK-NEXT:   %a4 = load double, double* %gep, align 8
; CHECK-NEXT:   store double %a4, double* %res, align 8
; CHECK-NEXT:   %1 = load double, double* %"res'", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"res'", align 8
; CHECK-NEXT:   %2 = fadd fast double %1, %differeturn
; CHECK-NEXT:   %3 = load double, double* %[[gepipge]], align 8
; CHECK-NEXT:   %4 = fadd fast double %3, %2
; CHECK-NEXT:   store double %4, double* %[[gepipge]], align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
