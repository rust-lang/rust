; RUN: %opt < %s %loadEnzyme -enzyme -enzyme_preopt=false -mem2reg -sroa -simplifycfg -instcombine -adce -S | FileCheck %s

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

; CHECK: define internal { double } @diffefoo(double %inp, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %conv = fptoui double %inp to i64
; CHECK-NEXT:   %call_augmented = call { { i8* }, i64*, i64* } @augmented_substore(i64 %conv, i64 3)
; CHECK-NEXT:   %0 = extractvalue { { i8* }, i64*, i64* } %call_augmented, 0
; CHECK-NEXT:   %1 = extractvalue { { i8* }, i64*, i64* } %call_augmented, 2
; CHECK-NEXT:   %"'ipc" = bitcast i64* %1 to double*
; CHECK-NEXT:   %2 = load double, double* %"'ipc", align 8
; CHECK-NEXT:   %3 = fadd fast double %2, %differeturn
; CHECK-NEXT:   store double %3, double* %"'ipc", align 8
; CHECK-NEXT:   %conv_unwrap = fptoui double %inp to i64
; CHECK-NEXT:   %4 = call {} @diffesubstore(i64 %conv_unwrap, i64 3, { i8* } %0)
; CHECK-NEXT:   ret { double } zeroinitializer
; CHECK-NEXT: }

; CHECK: define dso_local noalias i64* @preprocess_substore(i64 %flt, i64 %integral)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %call = tail call noalias i8* @malloc(i64 16) #6
; CHECK-NEXT:   %0 = bitcast i8* %call to i64*
; CHECK-NEXT:   store i64 %flt, i64* %0, align 8
; CHECK-NEXT:   %arrayidx1 = getelementptr inbounds i8, i8* %call, i64 8
; CHECK-NEXT:   %1 = bitcast i8* %arrayidx1 to i64*
; CHECK-NEXT:   store i64 %integral, i64* %1, align 8
; CHECK-NEXT:   ret i64* %0
; CHECK-NEXT: }

; CHECK: define internal { { i8* }, i64*, i64* } @augmented_substore(i64 %flt, i64 %integral)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %call = tail call noalias i8* @malloc(i64 16) #6
; CHECK-NEXT:   %"call'mi" = tail call noalias nonnull i8* @malloc(i64 16) #6
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull align 1 %"call'mi", i8 0, i64 16, i1 false)
; CHECK-NEXT:   %0 = bitcast i8* %call to i64*
; CHECK-NEXT:   %"'ipc2" = bitcast i8* %"call'mi" to i64*
; CHECK-NEXT:   store i64 0, i64* %"'ipc2", align 8
; CHECK-NEXT:   store i64 %flt, i64* %0, align 8
; CHECK-NEXT:   %"arrayidx1'ipge" = getelementptr inbounds i8, i8* %"call'mi", i64 8
; CHECK-NEXT:   %arrayidx1 = getelementptr inbounds i8, i8* %call, i64 8
; CHECK-NEXT:   %1 = bitcast i8* %arrayidx1 to i64*
; CHECK-NEXT:   %"'ipc1" = bitcast i8* %"arrayidx1'ipge" to i64*
; CHECK-NEXT:   store i64 0, i64* %"'ipc1", align 8
; CHECK-NEXT:   store i64 %integral, i64* %1, align 8
; CHECK-NEXT:   %"'ipc" = bitcast i8* %"call'mi" to i64*
; CHECK-NEXT:   %.fca.0.0.insert = insertvalue { { i8* }, i64*, i64* } undef, i8* %"call'mi", 0, 0
; CHECK-NEXT:   %.fca.1.insert = insertvalue { { i8* }, i64*, i64* } %.fca.0.0.insert, i64* %0, 1
; CHECK-NEXT:   %.fca.2.insert = insertvalue { { i8* }, i64*, i64* } %.fca.1.insert, i64* %"'ipc", 2
; CHECK-NEXT:   ret { { i8* }, i64*, i64* } %.fca.2.insert
; CHECK-NEXT: }

; CHECK: define internal {} @diffesubstore(i64 %flt, i64 %integral, { i8* } %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"call'mi" = extractvalue { i8* } %tapeArg, 0
; CHECK-NEXT:   todo push back from arg int
; CHECK-NEXT:   tail call void @free(i8* nonnull %"call'mi")
; CHECK-NEXT:   ret {} undef
; CHECK-NEXT: }
