; RUN: %opt < %s %loadEnzyme -enzyme -enzyme_preopt=false -mem2reg -sroa -simplifycfg -instcombine -adce -S | FileCheck %s


%sub = type { [5 x i64] }
%sub2 = type { %sub }

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

; CHECK: define internal void @diffecallee(i64* %ptr, i64* %"ptr'") {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %ptr2 = getelementptr inbounds i64, i64* %ptr, i64 2
; CHECK-NEXT:   %loadnotype = load i64, i64* %ptr2, align 4
; CHECK-NEXT:   %ptr3 = getelementptr inbounds i64, i64* %ptr, i64 3
; CHECK-NEXT:   store i64 %loadnotype, i64* %ptr3, align 4, !tbaa !0
; CHECK-NEXT:   %[[ptr4:.+]] = getelementptr inbounds i64, i64* %ptr, i64 4
; CHECK-NEXT:   store i64 %loadnotype, i64* %[[ptr4]], align 4
; CHECK-NEXT:   %[[dptr4:.+]] = getelementptr inbounds i64, i64* %"ptr'", i64 4
; CHECK-NEXT:   %[[double_ptr4:.+]] = bitcast i64* %[[dptr4]] to double*
; CHECK-NEXT:   %[[ldouble_ptr4:.+]] = load double, double* %[[double_ptr4]], align 8
; CHECK-NEXT:   store i64 0, i64* %[[dptr4]], align 4
; CHECK-NEXT:   %[[dptr2:.+]] = getelementptr inbounds i64, i64* %"ptr'", i64 2
; CHECK-NEXT:   %[[double_dptr2:.+]] = bitcast i64* %[[dptr2]] to double*
; CHECK-NEXT:   %[[ldouble_dptr2:.+]] = load double, double* %[[double_dptr2]], align 8
; CHECK-NEXT:   %7 = fadd fast double %[[ldouble_dptr2]], %[[ldouble_ptr4]]
; CHECK-NEXT:   %8 = bitcast i64* %[[dptr2]] to double*
; CHECK-NEXT:   store double %7, double* %8, align 8
; CHECK-NEXT:   %[[ptr3ipge:.+]] = getelementptr inbounds i64, i64* %"ptr'", i64 3
; CHECK-NEXT:   %9 = bitcast i64* %[[ptr3ipge]] to double*
; CHECK-NEXT:   %10 = load double, double* %9, align 8
; CHECK-NEXT:   store i64 0, i64* %[[ptr3ipge]], align 4
; CHECK-NEXT:   %[[ptr2ipge:.+]] = getelementptr inbounds i64, i64* %"ptr'", i64 2
; CHECK-NEXT:   %[[double_dptr2:.+]] = bitcast i64* %[[ptr2ipge]] to double*
; CHECK-NEXT:   %[[nv:.+]] = load double, double* %[[double_dptr2]]
; CHECK-NEXT:   %[[final:.+]] = fadd fast double %[[nv]], %10
; CHECK-NEXT:   %[[double2_dptr2:.+]] = bitcast i64* %[[ptr2ipge]] to double*
; CHECK-NEXT:   store double %[[final]], double* %[[double2_dptr2]], align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
