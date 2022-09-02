; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -simplifycfg -instsimplify -adce -S | FileCheck %s

define void @derivative(i64* %ptr, i64* %ptrp) {
entry:
  %call = call double (i8*, ...) @__enzyme_autodiff(i8* bitcast (void (i64*)* @callee to i8*), metadata !"enzyme_dup", i64* %ptr, i64* %ptrp)
  ret void
}

define i64* @gep(i64* %ptr, i64 %idx) {
entry:
  %next = getelementptr inbounds i64, i64* %ptr, i64 %idx
  ret i64* %next
}

define void @callee(i64* %ptr) {
entry:
  %ptr2 = call i64* @gep(i64* %ptr, i64 2)
  %loadnotype = load i64, i64* %ptr2
  %ptr3 = call i64* @gep(i64* %ptr, i64 3)
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
; CHECK-NEXT:   %ptr2_augmented = call { i64*, i64* } @augmented_gep.1(i64* %ptr, i64* %"ptr'", i64 2)
; CHECK-NEXT:   %ptr2 = extractvalue { i64*, i64* } %ptr2_augmented, 0
; CHECK-NEXT:   %"ptr2'ac" = extractvalue { i64*, i64* } %ptr2_augmented, 1
; CHECK-NEXT:   %loadnotype = load i64, i64* %ptr2
; CHECK-NEXT:   %ptr3_augmented = call { i64*, i64* } @augmented_gep(i64* %ptr, i64* %"ptr'", i64 3)
; CHECK-NEXT:   %ptr3 = extractvalue { i64*, i64* } %ptr3_augmented, 0
; CHECK-NEXT:   %"ptr3'ac" = extractvalue { i64*, i64* } %ptr3_augmented, 1
; CHECK-NEXT:   store i64 %loadnotype, i64* %ptr3
; CHECK-NEXT:   %[[cptr2ipge:.+]] = getelementptr inbounds i64, i64* %"ptr'", i64 2
; CHECK-NEXT:   %cptr2 = getelementptr inbounds i64, i64* %ptr, i64 2
; CHECK-NEXT:   %loadtype = load i64, i64* %cptr2
; CHECK-NEXT:   %[[cptr4ipge:.+]] = getelementptr inbounds i64, i64* %"ptr'", i64 4
; CHECK-NEXT:   %cptr4 = getelementptr inbounds i64, i64* %ptr, i64 4
; CHECK-NEXT:   store i64 %loadtype, i64* %cptr4{{(, align 4)?}}, !tbaa !0
; CHECK-NEXT:   %0 = load i64, i64* %"cptr4'ipg"
; CHECK-NEXT:   store i64 0, i64* %"cptr4'ipg"
; CHECK-NEXT:   %1 = bitcast i64* %"cptr2'ipg" to double*
; CHECK-DAG:    %[[add1:.+]] = bitcast i64 %0 to double
; CHECK-DAG:    %[[add2:.+]] = load double, double* %1
; CHECK-NEXT:   %4 = fadd fast double %[[add2]], %[[add1]]
; CHECK-NEXT:   store double %4, double* %1
; CHECK-NEXT:   %[[a6:.+]] = load i64, i64* %"ptr3'ac"
; CHECK-NEXT:   store i64 0, i64* %"ptr3'ac"
; CHECK-NEXT:   call void @diffegep(i64* %ptr, i64* %"ptr'", i64 3)
; CHECK-NEXT:   %[[a7:.+]] = bitcast i64* %"ptr2'ac" to double*
; CHECK-DAG:    %[[sadd1:.+]] = bitcast i64 %[[a6]] to double
; CHECK-DAG:    %[[sadd2:.+]] = load double, double* %[[a7]]
; CHECK-NEXT:   %[[a10:.+]] = fadd fast double %[[sadd2]], %[[sadd1]]
; CHECK-NEXT:   store double %[[a10]], double* %[[a7]]
; CHECK-NEXT:   call void @diffegep.2(i64* %ptr, i64* %"ptr'", i64 2)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
