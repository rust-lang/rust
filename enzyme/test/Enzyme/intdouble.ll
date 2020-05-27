; RUN: %opt < %s %loadEnzyme -enzyme -enzyme_preopt=false -mem2reg -instsimplify -early-cse -simplifycfg -S | FileCheck %s

@.str = private unnamed_addr constant [12 x i8] c"x=%f xp=%f\0A\00", align 1

; Function Attrs: noinline norecurse nounwind readonly uwtable
define dso_local double @intcast(i64* nocapture readonly %x) #0 {
entry:
  %0 = bitcast i64* %x to double*
  %1 = load double, double* %0, align 8, !tbaa !2
  %mul = fmul fast double %1, %1
  ret double %mul
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

; Function Attrs: noinline nounwind uwtable
define dso_local void @derivative(i64* %x, i64* %xp) local_unnamed_addr #2 {
entry:
  %0 = tail call double (double (i64*)*, ...) @__enzyme_autodiff(double (i64*)* nonnull @intcast, metadata !"diffe_dup", i64* %x, i64* %xp)
  ret void
}

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (i64*)*, ...) #3

; Function Attrs: nounwind uwtable
define dso_local void @main(i32 %argc, i8** nocapture readonly %argv) local_unnamed_addr #4 {
entry:
  %x = alloca double, align 8
  %xp = alloca double, align 8
  %0 = bitcast double* %x to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %0) #3
  %arrayidx = getelementptr inbounds i8*, i8** %argv, i64 1
  %1 = load i8*, i8** %arrayidx, align 8, !tbaa !6
  %call.i = tail call fast double @strtod(i8* nocapture nonnull %1, i8** null) #3
  store double %call.i, double* %x, align 8, !tbaa !2
  %2 = bitcast double* %xp to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2) #3
  store double 0.000000e+00, double* %xp, align 8, !tbaa !2
  %3 = bitcast double* %x to i64*
  %4 = bitcast double* %xp to i64*
  call void @derivative(i64* nonnull %3, i64* nonnull %4)
  %5 = load double, double* %x, align 8, !tbaa !2
  %6 = load double, double* %xp, align 8, !tbaa !2
  %call1 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([12 x i8], [12 x i8]* @.str, i64 0, i64 0), double %5, double %6)
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2) #3
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %0) #3
  ret void
}

; Function Attrs: nounwind
declare dso_local i32 @printf(i8* nocapture readonly, ...) local_unnamed_addr #5

; Function Attrs: nounwind
declare dso_local double @strtod(i8* readonly, i8** nocapture) local_unnamed_addr #5

attributes #0 = { noinline norecurse nounwind readonly uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #3 = { nounwind }
attributes #4 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #5 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0}
!3 = !{!"double", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
!6 = !{!7, !7, i64 0}
!7 = !{!"any pointer", !4, i64 0}


; CHECK: define internal {{(dso_local )?}}void @diffeintcast(i64* nocapture readonly %x, i64* nocapture %"x'", double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[ipc:.+]] = bitcast i64* %"x'" to double*
; CHECK-NEXT:   %[[uw:.+]] = bitcast i64* %x to double*
; CHECK-NEXT:   %[[uw1:.+]] = load double, double* %[[uw]]
; CHECK-NEXT:   %m0diffe = fmul fast double %differeturn, %[[uw1]]
; CHECK-NEXT:   %[[added:.+]] = fadd fast double %m0diffe, %m0diffe
; CHECK-NEXT:   %[[prev:.+]] = load double, double* %[[ipc]]
; CHECK-NEXT:   %[[tostore:.+]] = fadd fast double %[[prev]], %[[added]]
; CHECK-NEXT:   store double %[[tostore]], double* %[[ipc]]
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
