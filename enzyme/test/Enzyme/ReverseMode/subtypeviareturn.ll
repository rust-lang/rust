; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -simplifycfg -instsimplify -adce -S | FileCheck %s

define i64 @subload(i64* %inp) {
entry:
  %res = load i64, i64* %inp
  ret i64 %res
}

define void @foo(i64* %inp, i64* %out) {
entry:
  %call = tail call i64 @subload(i64* %inp)
  store i64 %call, i64* %out, !tbaa !6
  ret void
}

define void @call(i64* %inp, i64* %inpp, i64* %out, i64* %outp) {
entry:
  call void (i8*, ...) @__enzyme_autodiff(i8* bitcast (void (i64*, i64*)* @foo to i8*), metadata !"enzyme_dup", i64* %inp, i64* %inpp, metadata !"enzyme_dup", i64* %out, i64* %outp)
  ret void
}

declare dso_local void @__enzyme_autodiff(i8*, ...)

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0}
!3 = !{!"long long", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
!6 = !{!7, !7, i64 0}
!7 = !{!"double", !4, i64 0}

; CHECK: define internal void @diffefoo(i64* %inp, i64* %"inp'", i64* %out, i64* %"out'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %call = call i64 @augmented_subload(i64* %inp, i64* %"inp'")
; CHECK-NEXT:   store i64 %call, i64* %out{{(, align 4)?}}, !tbaa !
; CHECK-NEXT:   %0 = load i64, i64* %"out'"
; CHECK-NEXT:   store i64 0, i64* %"out'"
; CHECK-NEXT:   call void @diffesubload(i64* %inp, i64* %"inp'", i64 %0)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal i64 @augmented_subload(i64* %inp, i64* %"inp'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %res = load i64, i64* %inp
; CHECK-NEXT:   ret i64 %res
; CHECK-NEXT: }

; CHECK: define internal void @diffesubload(i64* %inp, i64* %"inp'", i64 %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = bitcast i64* %"inp'" to double*
; CHECK-DAG:    %[[add1:.+]] = bitcast i64 %differeturn to double
; CHECK-DAG:    %[[add2:.+]] = load double, double* %0
; CHECK-NEXT:   %3 = fadd fast double %[[add2]], %[[add1]]
; CHECK-NEXT:   store double %3, double* %0
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
