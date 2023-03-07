; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -simplifycfg -instsimplify -adce -S | FileCheck %s


%sub = type { [5 x i64] }
%sub2 = type { %sub }

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
  store i64 %loadnotype, i64* %ptr3, !tbaa !8

  %cast = bitcast i64* %ptr to %sub*
  %cptr2 = getelementptr inbounds %sub, %sub* %cast, i64 0, i32 0, i32 2
  %loadtype = load i64, i64* %cptr2
  %cptr4 = getelementptr inbounds %sub, %sub* %cast, i64 0, i32 0, i32 4
  store i64 %loadtype, i64* %cptr4
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
; CHECK-NEXT:   %"ptr2'ipg" = getelementptr inbounds i64, i64* %"ptr'", i64 2
; CHECK-NEXT:   %ptr2 = getelementptr inbounds i64, i64* %ptr, i64 2
; CHECK-NEXT:   %loadnotype = load i64, i64* %ptr2
; CHECK-NEXT:   %"ptr3'ipg" = getelementptr inbounds i64, i64* %"ptr'", i64 3
; CHECK-NEXT:   %ptr3 = getelementptr inbounds i64, i64* %ptr, i64 3
; CHECK-NEXT:   store i64 %loadnotype, i64* %ptr3{{(, align 4)?}}, !tbaa !0
; CHECK-NEXT:   %"cast'ipc" = bitcast i64* %"ptr'" to %sub*
; CHECK-NEXT:   %cast = bitcast i64* %ptr to %sub*
; CHECK-NEXT:   %"cptr2'ipg" = getelementptr inbounds %sub, %sub* %"cast'ipc", i64 0, i32 0, i32 2
; CHECK-NEXT:   %cptr2 = getelementptr inbounds %sub, %sub* %cast, i64 0, i32 0, i32 2
; CHECK-NEXT:   %loadtype = load i64, i64* %cptr2
; CHECK-NEXT:   %"cptr4'ipg" = getelementptr inbounds %sub, %sub* %"cast'ipc", i64 0, i32 0, i32 4
; CHECK-NEXT:   %cptr4 = getelementptr inbounds %sub, %sub* %cast, i64 0, i32 0, i32 4
; CHECK-NEXT:   store i64 %loadtype, i64* %cptr4
; CHECK-NEXT:   %0 = load i64, i64* %"cptr4'ipg"
; CHECK-NEXT:   store i64 0, i64* %"cptr4'ipg"
; CHECK-NEXT:   %1 = bitcast i64* %"cptr2'ipg" to double*
; CHECK-DAG:    %[[add1:.+]] = bitcast i64 %0 to double
; CHECK-DAG:    %[[add2:.+]] = load double, double* %1
; CHECK-NEXT:   %4 = fadd fast double %[[add2]], %[[add1]]
; CHECK-NEXT:   store double %4, double* %1
; CHECK-NEXT:   %[[a6:.+]] = load i64, i64* %"ptr3'ipg"
; CHECK-NEXT:   store i64 0, i64* %"ptr3'ipg"
; CHECK-NEXT:   %[[a7:.+]] = bitcast i64* %"ptr2'ipg" to double*
; CHECK-DAG:    %[[sadd1:.+]] = bitcast i64 %[[a6]] to double
; CHECK-DAG:    %[[sadd2:.+]] = load double, double* %[[a7]]
; CHECK-NEXT:   %[[a10:.+]] = fadd fast double %[[sadd2]], %[[sadd1]]
; CHECK-NEXT:   store double %[[a10]], double* %[[a7]]
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
