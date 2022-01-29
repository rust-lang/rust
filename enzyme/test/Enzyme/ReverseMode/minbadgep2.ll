; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -early-cse -instsimplify -adce -simplifycfg -S | FileCheck %s

;; TODO MAKE TEST WHERE gep of known constant pointer is returned or would otherwise be deduced active


@enzyme_const = external dso_local local_unnamed_addr global i32, align 4

; Function Attrs: nounwind uwtable
define dso_local double* @mv(double* nocapture %mat, double* nocapture readonly %vec) {
entry:
  %mload = load double, double* %mat
  %arrayidx = getelementptr inbounds double, double* %vec, i64 1
  %vload = load double, double* %arrayidx
  %mul = fmul fast double %vload, %mload
  store double %mul, double* %mat
  ret double* %arrayidx
}

declare dso_local double @_Z17__enzyme_autodiffIdJPFdPdS0_ES0_S0_iS0_EET_DpT0_(double* (double*, double*)*, double*, double*, i32, double*)

define dso_local double @_Z11matvec_realPdS_(double* nocapture %mat, double* nocapture %dmat, double* nocapture readonly %vec) {
  %dc = load i32, i32* @enzyme_const
  %call34.i = call fast double @_Z17__enzyme_autodiffIdJPFdPdS0_ES0_S0_iS0_EET_DpT0_(double* (double*, double*)* nonnull @mv, double* nonnull %mat, double* nonnull %dmat, i32 %dc, double* nonnull %vec)
  ret double %call34.i
}

; CHECK: define internal void @diffemv(double* nocapture %mat, double* nocapture %"mat'", double* nocapture readonly %vec)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %mload = load double, double* %mat
; CHECK-NEXT:   %arrayidx = getelementptr inbounds double, double* %vec, i64 1
; CHECK-NEXT:   %vload = load double, double* %arrayidx
; CHECK-NEXT:   %mul = fmul fast double %vload, %mload
; CHECK-NEXT:   store double %mul, double* %mat
; CHECK-NEXT:   %0 = load double, double* %"mat'"
; CHECK-NEXT:   %m1diffemload = fmul fast double %0, %vload
; CHECK-NEXT:   store double %m1diffemload, double* %"mat'"
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
