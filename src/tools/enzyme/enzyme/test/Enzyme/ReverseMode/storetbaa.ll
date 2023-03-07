; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -simplifycfg -instcombine -adce -S | FileCheck %s

define void @derivative(i64* %from, i64* %fromp, i64* %to, i64* %top) {
entry:
  %call = call double (i8*, ...) @__enzyme_autodiff(i8* bitcast (void (i64*, i64*)* @callee to i8*), metadata !"enzyme_dup", i64* %from, i64* %fromp, metadata !"enzyme_dup", i64* %to, i64* %top)
  ret void
}

define void @callee(i64* %from, i64* %to) {
entry:
  %loadk = load i64, i64* %from, align 8
  store i64 %loadk, i64* %to, align 8, !tbaa !8
  ret void
}

; Function Attrs: alwaysinline
declare double @__enzyme_autodiff(i8*, ...)

!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}
!7 = !{!"long", !5, i64 0}
!8 = !{!7, !7, i64 0}

; CHECK: define internal void @diffecallee(i64* %from, i64* %"from'", i64* %to, i64* %"to'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %loadk = load i64, i64* %from, align 8
; CHECK-NEXT:   store i64 %loadk, i64* %"to'", align 8
; CHECK-NEXT:   store i64 %loadk, i64* %to, align 8, !tbaa !0
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
