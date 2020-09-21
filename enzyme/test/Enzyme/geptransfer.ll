; RUN: %opt < %s %loadEnzyme -enzyme -enzyme_preopt=false -mem2reg -sroa -simplifycfg -instsimplify -gvn -adce -S | FileCheck %s

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

; CHECK: define internal void @diffecallee(i64* %ptr, i64* %"ptr'") {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[cptr2ipge:.+]] = getelementptr inbounds i64, i64* %"ptr'", i64 2
; CHECK-NEXT:   %ptr2 = getelementptr inbounds i64, i64* %ptr, i64 2
; CHECK-NEXT:   %loadnotype = load i64, i64* %ptr2
; CHECK-NEXT:   %[[ptr3ipge:.+]] = getelementptr inbounds i64, i64* %"ptr'", i64 3
; CHECK-NEXT:   %ptr3 = getelementptr inbounds i64, i64* %ptr, i64 3
; CHECK-NEXT:   store i64 %loadnotype, i64* %ptr3
; CHECK-NEXT:   %[[cptr4ipge:.+]] = getelementptr inbounds i64, i64* %"ptr'", i64 4
; CHECK-NEXT:   %cptr4 = getelementptr inbounds i64, i64* %ptr, i64 4
; CHECK-NEXT:   store i64 %loadnotype, i64* %cptr4, !tbaa !0
; CHECK-NEXT:   %0 = load i64, i64* %"cptr4'ipg"
; CHECK-NEXT:   store i64 0, i64* %"cptr4'ipg"
; CHECK-NEXT:   %1 = load i64, i64* %"ptr2'ipg"
; CHECK-NEXT:   %2 = bitcast i64 %0 to double
; CHECK-NEXT:   %3 = bitcast i64 %1 to double
; CHECK-NEXT:   %4 = fadd fast double %3, %2
; CHECK-NEXT:   %5 = bitcast double %4 to i64
; CHECK-NEXT:   store i64 %5, i64* %"ptr2'ipg"
; CHECK-NEXT:   %6 = load i64, i64* %"ptr3'ipg"
; CHECK-NEXT:   store i64 0, i64* %"ptr3'ipg"
; CHECK-NEXT:   %7 = bitcast i64 %6 to double
; CHECK-NEXT:   %8 = fadd fast double %4, %7
; CHECK-NEXT:   %9 = bitcast double %8 to i64
; CHECK-NEXT:   store i64 %9, i64* %"ptr2'ipg"
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
