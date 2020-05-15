; RUN: %opt < %s %loadEnzyme -enzyme -enzyme_preopt=false -mem2reg -sroa -simplifycfg -instcombine -adce -S | FileCheck %s

define void @derivative(i64* %ptr, i64* %ptrp, i64* %ptr2, i64* %ptr2p) {
entry:
  %call = call double (i8*, ...) @__enzyme_autodiff(i8* bitcast (void (i64*, i64*)* @callee to i8*), metadata !"diffe_dup", i64* %ptr, i64* %ptrp, metadata !"diffe_dup", i64* %ptr2, i64* %ptr2p)
  ret void
}

define void @callee(i64* %ptr, i64* %ptr2) {
entry:
  %loadnotype = load i64, i64* %ptr
  %cst = bitcast i64 %loadnotype to double
  %add = fadd double %cst, %cst
  %cst2 = bitcast double %cst to i64
  store i64 %cst2, i64* %ptr2
  ret void
}

; Function Attrs: alwaysinline
declare double @__enzyme_autodiff(i8*, ...)

; CHECK: define internal void @diffecallee(i64* %ptr, i64* %"ptr'", i64* %ptr2, i64* %"ptr2'") {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %loadnotype12 = load i64, i64* %ptr, align 4
; CHECK-NEXT:   store i64 %loadnotype12, i64* %ptr2, align 4
; CHECK-NEXT:   %0 = bitcast i64* %"ptr2'" to double*
; CHECK-NEXT:   %1 = load double, double* %0, align 8
; CHECK-NEXT:   store i64 0, i64* %"ptr2'", align 4
; CHECK-NEXT:   %2 = bitcast i64* %"ptr'" to double*
; CHECK-NEXT:   %3 = load double, double* %2, align 8
; CHECK-NEXT:   %4 = fadd fast double %3, %1
; CHECK-NEXT:   %5 = bitcast i64* %"ptr'" to double*
; CHECK-NEXT:   store double %4, double* %5, align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
