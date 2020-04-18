; RUN: %opt < %s %loadEnzyme -enzyme -enzyme_preopt=false -mem2reg -sroa -simplifycfg -instcombine -gvn -adce -S | FileCheck %s

define void @derivative(i64* %ptr, i64* %ptrp) {
entry:
  %call = call double (i8*, ...) @__enzyme_autodiff(i8* bitcast (void (i64*)* @callee to i8*), metadata !"diffe_dup", i64* %ptr, i64* %ptrp)
  ret void
}

define void @callee(i64* %ptr) {
entry:
  %ptr2 = getelementptr inbounds i64, i64* %ptr, i64 2
  %loadnotype = load i64, i64* %ptr2
  %ptr3 = getelementptr inbounds i64, i64* %ptr, i64 3
  store i64 %loadnotype, i64* %ptr3

  %cast = bitcast i64* %ptr to <2 x float>*
  %cast2 = bitcast <2 x float>* %cast to i64*
  %cptr2 = getelementptr inbounds i64, i64* %cast2, i64 2
  %loadtype = load i64, i64* %cptr2
  %cptr4 = getelementptr inbounds i64, i64* %cast2, i64 4
  store i64 %loadtype, i64* %cptr4, !tbaa !8
  ret void
}

; Function Attrs: alwaysinline
declare double @__enzyme_autodiff(i8*, ...)

!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}
!7 = !{!"double", !5, i64 0}
!8 = !{!7, !7, i64 0}

; CHECK: define internal {} @diffecallee(i64* %ptr, i64* %"ptr'") {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %ptr2 = getelementptr inbounds i64, i64* %ptr, i64 2
; CHECK-NEXT:   %loadnotype = load i64, i64* %ptr2, align 4
; CHECK-NEXT:   %ptr3 = getelementptr inbounds i64, i64* %ptr, i64 3
; CHECK-NEXT:   store i64 %loadnotype, i64* %ptr3, align 4
; CHECK-NEXT:   %cptr4 = getelementptr inbounds i64, i64* %ptr, i64 4
; CHECK-NEXT:   store i64 %loadnotype, i64* %cptr4, align 4, !tbaa !0
; CHECK-NEXT:   %[[cptr4ipge:.+]] = getelementptr inbounds i64, i64* %"ptr'", i64 4
; CHECK-NEXT:   %0 = bitcast i64* %[[cptr4ipge]] to double*
; CHECK-NEXT:   %1 = load double, double* %0, align 8
; CHECK-NEXT:   store i64 0, i64* %[[cptr4ipge]], align 4
; CHECK-NEXT:   %[[cptr2ipge:.+]] = getelementptr inbounds i64, i64* %"ptr'", i64 2
; CHECK-NEXT:   %2 = bitcast i64* %[[cptr2ipge]] to double*
; CHECK-NEXT:   %3 = load double, double* %2, align 8
; CHECK-NEXT:   %4 = fadd fast double %3, %1
; CHECK-NEXT:   store double %4, double* %2, align 8
; CHECK-NEXT:   %[[ptr3ipge:.+]] = getelementptr inbounds i64, i64* %"ptr'", i64 3
; CHECK-NEXT:   %[[dptr3:.+]] = bitcast i64* %[[ptr3ipge]] to double*
; CHECK-NEXT:   %[[dptr3load:.+]] = load double, double* %[[dptr3]], align 8
; CHECK-NEXT:   store i64 0, i64* %[[ptr3ipge]], align 4
; CHECK-NEXT:   %[[finalst:.+]] = fadd fast double %4, %[[dptr3load]]
; CHECK-NEXT:   store double %[[finalst]], double* %2, align 8
; CHECK-NEXT:   ret {} undef
; CHECK-NEXT: }
