; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -simplifycfg -instsimplify -adce -S | FileCheck %s

define void @derivative(i64* %ptr, i64* %ptrp, i64* %ptr2, i64* %ptr2p) {
entry:
  %call = call double (i8*, ...) @__enzyme_autodiff(i8* bitcast (void (i64*, i64*)* @callee to i8*), metadata !"enzyme_dup", i64* %ptr, i64* %ptrp, metadata !"enzyme_dup", i64* %ptr2, i64* %ptr2p)
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

; CHECK: define internal void @diffecallee(i64* %ptr, i64* %"ptr'", i64* %ptr2, i64* %"ptr2'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %loadnotype = load i64, i64* %ptr
; CHECK-NEXT:   store i64 %loadnotype, i64* %ptr2
; CHECK-NEXT:   %0 = load i64, i64* %"ptr2'"
; CHECK-NEXT:   store i64 0, i64* %"ptr2'"
; CHECK-NEXT:   %1 = bitcast i64* %"ptr'" to double*
; CHECK-DAG:    %[[add1:.+]] = bitcast i64 %0 to double
; CHECK-DAG:    %[[add2:.+]] = load double, double* %1
; CHECK-NEXT:   %4 = fadd fast double %[[add2]], %[[add1]]
; CHECK-NEXT:   store double %4, double* %1
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
