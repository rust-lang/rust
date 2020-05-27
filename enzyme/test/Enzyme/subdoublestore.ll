; RUN: %opt < %s %loadEnzyme -enzyme -enzyme_preopt=false -mem2reg -sroa -simplifycfg -instcombine -adce -S | FileCheck %s

source_filename = "subdoublestore.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind uwtable
define dso_local noalias i64* @substore(i64 %flt, i64 %integral) local_unnamed_addr #1 {
entry:
  %call = tail call noalias i8* @malloc(i64 16) #6
  %0 = bitcast i8* %call to i64*
  store i64 %flt, i64* %0
  %arrayidx1 = getelementptr inbounds i8, i8* %call, i64 8
  %1 = bitcast i8* %arrayidx1 to i64*
  store i64 %integral, i64* %1
  ret i64* %0
}

; Function Attrs: nounwind
declare dso_local noalias i8* @malloc(i64) local_unnamed_addr #2

; Function Attrs: nounwind uwtable
define dso_local double @foo(double %inp) #3 {
entry:
  %conv = bitcast double %inp to i64
  %call = tail call i64* @substore(i64 %conv, i64 3)
  %0 = bitcast i64* %call to double*
  %1 = load double, double* %0
  ret double %1
}

; Function Attrs: nounwind uwtable
define dso_local double @call(double %inp) local_unnamed_addr #3 {
entry:
  %call = tail call double (i8*, ...) @__enzyme_autodiff(i8* bitcast (double (double)* @foo to i8*), double %inp) #6
  ret double %call
}

declare dso_local double @__enzyme_autodiff(i8*, ...) local_unnamed_addr #4

; Function Attrs: nounwind readnone speculatable
declare float @llvm.fabs.f32(float) #5

attributes #0 = { nounwind readnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { nounwind readnone speculatable }
attributes #6 = { nounwind }

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
; CHECK-NEXT:   %conv = bitcast double %inp to i64
; CHECK-NEXT:   %call_augmented = call { i8*, i64* } @augmented_substore(i64 %conv, i64 3)
; CHECK-NEXT:   %[[tape:.+]] = extractvalue { i8*, i64* } %call_augmented, 0
; CHECK-NEXT:   %"call'ac" = extractvalue { i8*, i64* } %call_augmented, 1
; CHECK-NEXT:   %[[ipc:.+]] = bitcast i64* %"call'ac" to double*
; CHECK-NEXT:   %[[ldi1:.+]] = load double, double* %[[ipc]], align 8
; CHECK-NEXT:   %[[addf1:.+]] = fadd fast double %[[ldi1]], %differeturn
; CHECK-NEXT:   store double %[[addf1]], double* %[[ipc]], align 8
; CHECK-NEXT:   %[[substore:.+]] = call { i64 } @diffesubstore(i64 %conv, i64 3, i8* %[[tape]])
; CHECK-NEXT:   %[[ev0:.+]] = extractvalue { i64 } %[[substore]], 0
; CHECK-NEXT:   %[[bc0:.+]] = bitcast i64 %[[ev0]] to double
; CHECK-NEXT:   %[[ret:.+]] = insertvalue { double } undef, double %[[bc0]], 0
; CHECK-NEXT:   ret { double } %[[ret]]
; CHECK-NEXT: }

; CHECK: define internal { i8*, i64* } @augmented_substore(i64 %flt, i64 %integral)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"call'mi" = tail call noalias nonnull i8* @malloc(i64 16) #6
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull align 1 %"call'mi", i8 0, i64 16, i1 false)
; CHECK-NEXT:   %"'ipc" = bitcast i8* %"call'mi" to i64*
; CHECK-NEXT:   %"arrayidx1'ipg" = getelementptr inbounds i8, i8* %"call'mi", i64 8
; CHECK-NEXT:   %"'ipc1" = bitcast i8* %"arrayidx1'ipg" to i64*
; CHECK-NEXT:   store i64 %integral, i64* %"'ipc1", align 8
; CHECK-NEXT:   %.fca.0.insert = insertvalue { i8*, i64* } undef, i8* %"call'mi", 0
; CHECK-NEXT:   %.fca.1.insert = insertvalue { i8*, i64* } %.fca.0.insert, i64* %"'ipc", 1
; CHECK-NEXT:   ret { i8*, i64* } %.fca.1.insert
; CHECK-NEXT: }

; CHECK: define internal { i64 } @diffesubstore(i64 %flt, i64 %integral, i8* %[[tapeArg:.+]])
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[ipc:.+]] = bitcast i8* %[[tapeArg]] to i64*
; CHECK-NEXT:   %0 = bitcast i8* %[[tapeArg]] to i64*
; CHECK-NEXT:   %1 = load i64, i64* %0, align 8
; CHECK-NEXT:   store i64 0, i64* %[[ipc]], align 8
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[tapeArg]])
; CHECK-NEXT:   %2 = insertvalue { i64 } undef, i64 %1, 0
; CHECK-NEXT:   ret { i64 } %2
; CHECK-NEXT: }
