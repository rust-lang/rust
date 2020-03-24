; RUN: %opt < %s %loadEnzyme -enzyme -enzyme_preopt=false -mem2reg -sroa -simplifycfg -instcombine -adce -S | FileCheck %s

; TODO: the calling convention isn't set up to deal with returning a float as an integer and must be updated for this to pass
; XFAIL: *

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
  call void (i8*, ...) @__enzyme_autodiff(i8* bitcast (void (i64*, i64*)* @foo to i8*), metadata !"diffe_dup", i64* %inp, i64* %inpp, metadata !"diffe_dup", i64* %out, i64* %outp)
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

; CHECK: define internal {} @diffefoo(i64* %inp, i64* %"inp'", i64* %out, i64* %"out'") {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %call_augmented = call { {}, i64, i64 } @augmented_subload(i64* %inp, i64* %"inp'")
; CHECK-NEXT:   %call = extractvalue { {}, i64, i64 } %call_augmented, 1
; CHECK-NEXT:   store i64 %call, i64* %out, align 4
; CHECK-NEXT:   ; TODO put extract 2 in out'
; CHECK-NEXT:   store i64 %2, i64* %"out'", align 4
; CHECK-NEXT:   %[[unused:.+]] = call {} @diffesubload(i64* %inp, i64* %"inp'", {} undef)
; CHECK-NEXT:   ret {} undef
; CHECK-NEXT: }

; CHECK: define internal { {}, i64, i64 } @augmented_subload(i64* %inp, i64* %"inp'") {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"res'ipl" = load i64, i64* %"inp'", align 4
; CHECK-NEXT:   %res = load i64, i64* %inp, align 4
; CHECK-NEXT:   %.fca.1.insert = insertvalue { {}, i64, i64 } undef, i64 %res, 1
; CHECK-NEXT:   %.fca.2.insert = insertvalue { {}, i64, i64 } %.fca.1.insert, i64 %"res'ipl", 2
; CHECK-NEXT:   ret { {}, i64, i64 } %.fca.2.insert
; CHECK-NEXT: }

; CHECK: define internal {} @diffesubload(i64* %inp, i64* %"inp'", {} %tapeArg) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret {} undef
; CHECK-NEXT: }
