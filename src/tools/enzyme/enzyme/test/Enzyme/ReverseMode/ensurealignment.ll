; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -simplifycfg -instcombine -adce -S | FileCheck %s

; this test should ensure that the alignment on the <2 x double> load is kept

define void @caller(double* %in_W, double* %in_Wp) {
entry:
  call void @__enzyme_autodiff(i8* bitcast (<2 x double> (double*)* @matvec to i8*), double* nonnull %in_W, double* nonnull %in_Wp) #8
  ret void
}

declare void @__enzyme_autodiff(i8*, double*, double*)

define internal <2 x double> @matvec(double* noalias %W) {
entry:
  %W3p = getelementptr inbounds double, double* %W, i64 3
  %W34p = bitcast double* %W3p to <2 x double>*
  %W34 = load <2 x double>, <2 x double>* %W34p, align 1
  ret <2 x double> %W34
}

; CHECK: define internal void @diffematvec(double* noalias %W, double* %"W'", <2 x double> %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[W3p:.+]] = getelementptr inbounds double, double* %"W'", i64 3
; CHECK-NEXT:   %[[vW34p:.+]] = bitcast double* %[[W3p]] to <2 x double>*
; CHECK-NEXT:   %[[ld:.+]] = load <2 x double>, <2 x double>* %[[vW34p]], align 1
; CHECK-NEXT:   %[[add:.+]] = fadd fast <2 x double> %[[ld]], %differeturn
; CHECK-NEXT:   store <2 x double> %[[add]], <2 x double>* %[[vW34p]], align 1
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
