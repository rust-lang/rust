; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -simplifycfg -adce -S | FileCheck %s

define void @derivative(i64* %ptr, i64* %ptrp) {
entry:
  %call = call double (i8*, ...) @__enzyme_autodiff(i8* bitcast (void (i64*)* @callee to i8*), metadata !"enzyme_dup", i64* %ptr, i64* %ptrp)
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

; CHECK: define internal void @diffecallee(i64* %ptr, i64* %"ptr'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[cptr2ipge:.+]] = getelementptr inbounds i64, i64* %"ptr'", i64 2
; CHECK-NEXT:   %ptr2 = getelementptr inbounds i64, i64* %ptr, i64 2
; CHECK-NEXT:   %loadnotype = load i64, i64* %ptr2
; CHECK-NEXT:   %[[ptr3ipge:.+]] = getelementptr inbounds i64, i64* %"ptr'", i64 3
; CHECK-NEXT:   %ptr3 = getelementptr inbounds i64, i64* %ptr, i64 3
; CHECK-NEXT:   store i64 %loadnotype, i64* %ptr3
; CHECK-NEXT:   %"cast'ipc" = bitcast i64* %"ptr'" to <2 x float>*
; CHECK-NEXT:   %cast = bitcast i64* %ptr to <2 x float>*
; CHECK-NEXT:   %"cast2'ipc" = bitcast <2 x float>* %"cast'ipc" to i64*
; CHECK-NEXT:   %cast2 = bitcast <2 x float>* %cast to i64*
; CHECK-NEXT:   %"cptr2'ipg" = getelementptr inbounds i64, i64* %"cast2'ipc", i64 2
; CHECK-NEXT:   %cptr2 = getelementptr inbounds i64, i64* %cast2, i64 2
; CHECK-NEXT:   %loadtype = load i64, i64* %cptr2
; CHECK-NEXT:   %[[cptr4ipge:.+]] = getelementptr inbounds i64, i64* %"cast2'ipc", i64 4
; CHECK-NEXT:   %cptr4 = getelementptr inbounds i64, i64* %cast2, i64 4
; CHECK-NEXT:   store i64 %loadtype, i64* %cptr4{{(, align 4)?}}, !tbaa !0
; CHECK-NEXT:   %[[lcptr4:.+]] = load i64, i64* %"cptr4'ipg"
; CHECK-NEXT:   store i64 0, i64* %"cptr4'ipg"

; CHECK-NEXT:   %[[zerod:.+]] = bitcast i64 0 to double
; CHECK-NEXT:   %[[dder:.+]] = bitcast i64 %[[lcptr4]] to double
; CHECK-NEXT:   %[[same:.+]] = fadd fast double %[[zerod]], %[[dder]]
; CHECK-NEXT:   %[[backlcptr4:.+]] = bitcast double %[[same]] to i64

; CHECK-NEXT:   %[[lcptr2:.+]] = bitcast i64* %"cptr2'ipg" to double*
; CHECK-DAG:    %[[bcptr2:.+]] = load double, double* %[[lcptr2]]
; CHECK-DAG:    %[[bcptr4:.+]] = bitcast i64 %[[backlcptr4]] to double
; CHECK-NEXT:   %[[mmadd:.+]] = fadd fast double %[[bcptr2]], %[[bcptr4]]
; CHECK-NEXT:   store double %[[mmadd]], double* %[[lcptr2]]
; CHECK-NEXT:   %[[lptr3:.+]] = load i64, i64* %"ptr3'ipg"
; CHECK-NEXT:   store i64 0, i64* %"ptr3'ipg"

; CHECK-NEXT:  %[[zerod2:.+]] = bitcast i64 0 to double
; CHECK-NEXT:  %[[bczd:.+]] = bitcast i64 %[[lptr3]] to double
; CHECK-NEXT:  %[[fasd:.+]] = fadd fast double %[[zerod2]], %[[bczd]]
; CHECK-NEXT:  %[[nlptr3:.+]] = bitcast double %[[fasd]] to i64

; CHECK-NEXT:   %[[lptr2:.+]] = bitcast i64* %"ptr2'ipg" to double*
; CHECK-DAG:    %[[dptr2:.+]] = load double, double* %[[lptr2]]
; CHECK-DAG:    %[[dptr3:.+]] = bitcast i64 %[[nlptr3]] to double
; CHECK-NEXT:   %[[ladd:.+]] = fadd fast double %[[dptr2]], %[[dptr3]]
; CHECK-NEXT:   store double %[[ladd]], double* %[[lptr2]]
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
