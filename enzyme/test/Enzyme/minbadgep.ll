; RUN: %opt < %s %loadEnzyme -enzyme -enzyme_preopt=false -mem2reg -early-cse -instcombine -adce -simplifycfg -S | FileCheck %s

@diffe_const = external dso_local local_unnamed_addr global i32, align 4

; Function Attrs: nounwind uwtable
define dso_local double @mv(double* nocapture readonly %mat, double* nocapture readonly %vec) {
entry:
  %mload = load double, double* %mat
  %arrayidx = getelementptr inbounds double, double* %vec, i64 1
  %vload = load double, double* %arrayidx
  %mul = fmul fast double %vload, %mload
  %mul2 = fmul fast double %mul, %mul
  ret double %mul2
}

declare dso_local double @_Z17__enzyme_autodiffIdJPFdPdS0_ES0_S0_iS0_EET_DpT0_(double (double*, double*)*, double*, double*, i32, double*)
  
define dso_local double @_Z11matvec_realPdS_(double* nocapture readonly %mat, double* nocapture %dmat, double* nocapture readonly %vec) {
  %dc = load i32, i32* @diffe_const
  %call34.i = call fast double @_Z17__enzyme_autodiffIdJPFdPdS0_ES0_S0_iS0_EET_DpT0_(double (double*, double*)* nonnull @mv, double* nonnull %mat, double* nonnull %dmat, i32 %dc, double* nonnull %vec)
  ret double %call34.i
}

; CHECK: define internal {} @diffemv(double* nocapture readonly %mat, double* nocapture %"mat'", double* nocapture readonly %vec, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %arrayidx = getelementptr inbounds double, double* %vec, i64 1
; CHECK-NEXT:   %vload = load double, double* %arrayidx
; CHECK-NEXT:   %mload = load double, double* %mat
; CHECK-NEXT:   %mul_unwrap = fmul fast double %vload, %mload
; CHECK-NEXT:   %0 = fadd fast double %differeturn, %differeturn
; CHECK-NEXT:   %1 = fmul fast double %mul_unwrap, %0
; CHECK-NEXT:   %m1diffemload = fmul fast double %1, %vload
; CHECK-NEXT:   %2 = load double, double* %"mat'"
; CHECK-NEXT:   %3 = fadd fast double %2, %m1diffemload
; CHECK-NEXT:   store double %3, double* %"mat'"
; CHECK-NEXT:   ret {} undef
; CHECK-NEXT: }
