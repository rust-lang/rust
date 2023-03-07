; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -simplifycfg -instsimplify -early-cse -adce -S | FileCheck %s

define double @caller(i64* %A, i64* %Ap, double* %res, double* %resp) {
entry:
  %call = call double (i8*, ...) @__enzyme_autodiff(i8* bitcast (double (i64*, double*)* @matvec to i8*), metadata !"enzyme_dup", i64* %A, i64* %Ap, double* %res, double* %resp)
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

; CHECK: define internal void @diffematvec(i64* %lhs, i64* %"lhs'", double* %res, double* %"res'", double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"loaded'ipl" = load i64, i64* %"lhs'", align 4
; CHECK-NEXT:   %loaded = load i64, i64* %lhs, align 4
; CHECK-NEXT:   %"a2'ipc" = inttoptr i64 %"loaded'ipl" to double*
; CHECK-NEXT:   %a2 = inttoptr i64 %loaded to double*
; CHECK-NEXT:   %div = lshr i64 %loaded, 3
; CHECK-NEXT:   %and = and i64 %div, 1
; CHECK-NEXT:   %[[gepipge:.+]] = getelementptr inbounds double, double* %"a2'ipc", i64 %and
; CHECK-NEXT:   %gep = getelementptr inbounds double, double* %a2, i64 %and
; CHECK-NEXT:   %a4 = load double, double* %gep, align 8
; CHECK-NEXT:   store double %a4, double* %res, align 8
; CHECK-NEXT:   %[[lres:.+]] = load double, double* %"res'", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"res'", align 8
; CHECK-NEXT:   %[[fad:.+]] = fadd fast double %differeturn, %[[lres]]
; CHECK-NEXT:   %[[lgep:.+]] = load double, double* %[[gepipge]], align 8
; CHECK-NEXT:   %[[fres:.+]] = fadd fast double %[[lgep]], %[[fad]]
; CHECK-NEXT:   store double %[[fres]], double* %[[gepipge]], align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
