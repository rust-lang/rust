; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -simplifycfg -instsimplify -adce -S | FileCheck %s

declare void @__enzyme_autodiff(...)

; Function Attrs: alwaysinline norecurse nounwind uwtable
define dso_local double @caller() local_unnamed_addr #0 {
entry:
  %kernel = alloca i64, align 8
  %kernelp = alloca i64, align 8
  %akernel = alloca i64, align 8
  %akernelp = alloca i64, align 8
  call void (...) @__enzyme_autodiff(void (i64*, i64*)* nonnull @mv, metadata !"enzyme_dup", i64* nonnull %kernel, i64* nonnull %kernelp, metadata !"enzyme_dup", i64* nonnull %akernel, i64* nonnull %akernelp) #4
  ret double 0.000000e+00
}

; Function Attrs: alwaysinline nounwind uwtable
define internal void @mv(i64* %m_dims, i64* %out) #1 {
entry:
  %call4 = call i64 @sub(i64* nonnull %m_dims)
  store i64 %call4, i64* %out
  ret void
}

define i64 @mul(i64 %a) {
entry:
  ret i64 %a
}

; Function Attrs: inlinehint norecurse nounwind readnone uwtable
define i64 @pop(i64 %arr.coerce0)  {
entry:
  %arr = alloca i64
  store i64 %arr.coerce0, i64* %arr
  %call.i = call i64* @cast(i64* nonnull %arr)
  %a2 = load i64, i64* %call.i, !tbaa !2
  %call2 = call i64 @mul(i64 %a2)
  ret i64 %call2
}

define i64* @cast(i64* %a) {
entry:
  ret i64* %a
}

; Function Attrs: inlinehint norecurse nounwind uwtable
define i64 @sub(i64* %this) {
entry:
  %agg = load i64, i64* %this
  %call = tail call i64 @pop(i64 %agg)
  ret i64 %call
}

attributes #0 = { alwaysinline norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { alwaysinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { inlinehint norecurse nounwind readnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { inlinehint norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0, i64 8}
!3 = !{!4, i64 8, !"double"}
!4 = !{!5, i64 1, !"omnipotent char"}
!5 = !{!"Simple C++ TBAA"}

; CHECK: define internal void @diffemv(i64* %m_dims, i64* %"m_dims'", i64* %out, i64* %"out'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %call4_augmented = call { {{.*}}, i64 } @augmented_sub(i64* %m_dims, i64* %"m_dims'")
; CHECK-NEXT:   %[[tape:.+]] = extractvalue { {{.*}}, i64 } %call4_augmented, 0
; CHECK-NEXT:   %call4 = extractvalue { {{.*}}, i64 } %call4_augmented, 1
; CHECK-NEXT:   store i64 %call4, i64* %out
; CHECK-NEXT:   %[[oold:.+]] = load i64, i64* %"out'"
; CHECK-NEXT:   store i64 0, i64* %"out'"
; CHECK-NEXT:   call void @diffesub(i64* %m_dims, i64* %"m_dims'", i64 %[[oold]], { {{.*}}, i64 } %[[tape]])
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal i64 @augmented_mul(i64 %a)
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret i64 %a
; CHECK-NEXT: }

; CHECK: define internal { i64*, i64* } @augmented_cast(i64* %a, i64* %"a'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %.fca.0.insert = insertvalue { i64*, i64* } {{(undef|poison)}}, i64* %a, 0
; CHECK-NEXT:   %.fca.1.insert = insertvalue { i64*, i64* } %.fca.0.insert, i64* %"a'", 1
; CHECK-NEXT:   ret { i64*, i64* } %.fca.1.insert
; CHECK-NEXT: }

; CHECK: define internal { {{.*}}, i64 } @augmented_pop(i64 %arr.coerce0)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %malloccall = tail call noalias nonnull dereferenceable(8) dereferenceable_or_null(8) i8* @malloc(i64 8)
; CHECK-NEXT:   %"malloccall'mi" = tail call noalias nonnull dereferenceable(8) dereferenceable_or_null(8) i8* @malloc(i64 8)
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull dereferenceable(8) dereferenceable_or_null(8) %"malloccall'mi", i8 0, i64 8, i1 false)
; CHECK-NEXT:   %"arr'ipc" = bitcast i8* %"malloccall'mi" to i64*
; CHECK-NEXT:   %arr = bitcast i8* %malloccall to i64*
; CHECK-NEXT:   store i64 %arr.coerce0, i64* %arr
; CHECK-NEXT:   %call.i_augmented = call { i64*, i64* } @augmented_cast(i64*{{( nonnull)?}} %arr, i64*{{( nonnull)?}} %"arr'ipc")
; CHECK-NEXT:   %call.i = extractvalue { i64*, i64* } %call.i_augmented, 0
; CHECK-NEXT:   %[[dcall:.+]] = extractvalue { i64*, i64* } %call.i_augmented, 1
; CHECK-NEXT:   %a2 = load i64, i64* %call.i{{(, align 4)?}}, !tbaa !2
; CHECK-NEXT:   %call2 = call i64 @augmented_mul(i64 %a2)
; CHECK-NEXT:   %.fca.0.0.insert = insertvalue { { i64*, i8*, i8*, i64 }, i64 } {{(undef|poison)}}, i64* %[[dcall]], 0, 0
; CHECK-NEXT:   %.fca.0.1.insert = insertvalue { { i64*, i8*, i8*, i64 }, i64 } %.fca.0.0.insert, i8* %"malloccall'mi", 0, 1
; CHECK-NEXT:   %.fca.0.2.insert = insertvalue { { i64*, i8*, i8*, i64 }, i64 } %.fca.0.1.insert, i8* %malloccall, 0, 2
; CHECK-NEXT:   %.fca.0.3.insert = insertvalue { { i64*, i8*, i8*, i64 }, i64 } %.fca.0.2.insert, i64 %a2, 0, 3
; CHECK-NEXT:   %.fca.1.insert = insertvalue { { i64*, i8*, i8*, i64 }, i64 } %.fca.0.3.insert, i64 %call2, 1
; CHECK-NEXT:   ret { { i64*, i8*, i8*, i64 }, i64 } %.fca.1.insert
; CHECK-NEXT: }

; CHECK: define internal { {{.*}}, i64 } @augmented_sub(i64* %this, i64* %"this'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %agg = load i64, i64* %this
; CHECK-NEXT:   %call_augmented = call { {{.*}}, i64 } @augmented_pop(i64 %agg)
; CHECK-NEXT:   %subcache = extractvalue { {{.*}}, i64 } %call_augmented, 0
; CHECK:    %call = extractvalue { {{.*}}, i64 } %call_augmented, 1
; CHECK:    insertvalue {{.*}} i64 %agg
; CHECK:    insertvalue {{.*}} i64 %call

; CHECK: define internal void @diffesub(i64* %this, i64* %"this'", i64 %differeturn, { {{.*}}, i64 } %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[agg:.+]] = extractvalue { {{.*}}, i64 } %tapeArg, 1
; CHECK-NEXT:   %[[pret:.+]] = extractvalue { {{.*}}, i64 } %tapeArg, 0
; CHECK-NEXT:   %[[dpop:.+]] = call { i64 } @diffepop(i64 %[[agg]], i64 %differeturn, {{.*}} %[[pret]])
; CHECK-NEXT:   %[[ev:.+]] = extractvalue { i64 } %[[dpop]], 0
; CHECK-NEXT:   %2 = bitcast i64* %"this'" to double*
; CHECK-DAG:    %[[add1:.+]] = bitcast i64 %[[ev]] to double
; CHECK-DAG:    %[[add2:.+]] = load double, double* %2
; CHECK-NEXT:   %5 = fadd fast double %[[add2]], %[[add1]]
; CHECK-NEXT:   store double %5, double* %2
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal { i64 } @diffepop(i64 %arr.coerce0, i64 %differeturn, {{.*}} %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[malloccall:.+]] = extractvalue { i64*, i8*, i8*, i64 } %tapeArg, 2
; CHECK-NEXT:   %[[dmalloccall:.+]] = extractvalue { i64*, i8*, i8*, i64 } %tapeArg, 1
; CHECK-NEXT:   %[[darr:.+]] = bitcast i8* %[[dmalloccall]] to i64*
; CHECK-NEXT:   %[[arr:.+]] = bitcast i8* %[[malloccall]] to i64*
; CHECK-NEXT:   %[[dcall:.+]] = extractvalue { i64*, i8*, i8*, i64 } %tapeArg, 0
; CHECK-NEXT:   %[[a2:.+]] = extractvalue { i64*, i8*, i8*, i64 } %tapeArg, 3
; CHECK-NEXT:   %0 = call { i64 } @diffemul(i64 %[[a2]], i64 %differeturn)
; CHECK-NEXT:   %1 = extractvalue { i64 } %0, 0
; CHECK-NEXT:   %2 = bitcast i64* %"call.i'ip_phi" to double*
; CHECK-DAG:    %[[sadd1:.+]] = bitcast i64 %1 to double
; CHECK-DAG:    %[[sadd2:.+]] = load double, double* %2
; CHECK-NEXT:   %5 = fadd fast double %[[sadd2]], %[[sadd1]]
; CHECK-NEXT:   store double %5, double* %2
; CHECK-NEXT:   call void @diffecast(i64* %arr, i64* %"arr'ipc")
; CHECK-NEXT:   %[[a7:.+]] = load i64, i64* %"arr'ipc"
; CHECK-NEXT:   store i64 0, i64* %"arr'ipc"
; CHECK-NEXT:   tail call void @free(i8* nonnull %"malloccall'mi")
; CHECK-NEXT:   tail call void @free(i8* %malloccall)
; CHECK-NEXT:   %[[a8:.+]] = insertvalue { i64 } undef, i64 %[[a7]], 0
; CHECK-NEXT:   ret { i64 } %[[a8]]
; CHECK-NEXT: }

; CHECK: define internal { i64 } @diffemul(i64 %a, i64 %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = insertvalue { i64 } undef, i64 %differeturn, 0
; CHECK-NEXT:   ret { i64 } %0
; CHECK-NEXT: }

; CHECK: define internal void @diffecast(i64* %a, i64* %"a'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
