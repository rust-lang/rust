; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -inline -O3 -dse -S | FileCheck %s

; Function Attrs: noinline norecurse nounwind readonly uwtable
define dso_local double @f(double* nocapture readonly %x) local_unnamed_addr #0 {
entry:
  %0 = load double, double* %x, align 8, !tbaa !2
  ret double %0
}

; Function Attrs: nounwind uwtable
define dso_local double @malloced(double %x, i64 %n) #1 {
entry:
  %mul = shl i64 %n, 3
  %call = tail call i8* @malloc(i64 %mul)
  %0 = bitcast i8* %call to double*
  store double %x, double* %0, align 8, !tbaa !2
  %call1 = tail call fast double @f(double* %0)
  %call2 = tail call i32 (double*, ...) bitcast (i32 (...)* @free to i32 (double*, ...)*)(double* %0) #4
  %mul3 = fmul fast double %call1, %call1
  ret double %mul3
}

; Function Attrs: nounwind
declare dso_local noalias i8* @malloc(i64) local_unnamed_addr #2

declare dso_local i32 @free(...) local_unnamed_addr #3

; Function Attrs: nounwind uwtable
define dso_local void @derivative(double %x, i64 %n) local_unnamed_addr #1 {
entry:
  %0 = tail call [3 x double] (double (double, i64)*, ...) @__enzyme_autodiff(double (double, i64)* nonnull @malloced, metadata !"enzyme_width", i64 3, double %x, i64 %n)
  ret void
}

; Function Attrs: nounwind
declare [3 x double] @__enzyme_autodiff(double (double, i64)*, ...) #4

attributes #0 = { noinline norecurse nounwind readonly uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #2 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #3 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #4 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0}
!3 = !{!"double", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}


; CHECK: define dso_local void @derivative(double %x, i64 %n)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %mul.i = shl i64 %n, 3
; CHECK-NEXT:   %call.i = tail call i8* @malloc(i64 %mul.i)
; CHECK-NEXT:   %"call'mi.i" = tail call noalias nonnull i8* @malloc(i64 %mul.i)
; CHECK-NEXT:   %"call'mi7.i" = tail call noalias nonnull i8* @malloc(i64 %mul.i)
; CHECK-NEXT:   %"call'mi8.i" = tail call noalias nonnull i8* @malloc(i64 %mul.i)
; CHECK-NEXT:   tail call void @llvm.memset.p0i8.i64(i8* nonnull align 1 %"call'mi.i", i8 0, i64 %mul.i, i1 false)
; CHECK-NEXT:   tail call void @llvm.memset.p0i8.i64(i8* nonnull align 1 %"call'mi7.i", i8 0, i64 %mul.i, i1 false)
; CHECK-NEXT:   tail call void @llvm.memset.p0i8.i64(i8* nonnull align 1 %"call'mi8.i", i8 0, i64 %mul.i, i1 false)
; CHECK-NEXT:   %"'ipc.i" = bitcast i8* %"call'mi.i" to double*
; CHECK-NEXT:   %0 = insertvalue [3 x double*] undef, double* %"'ipc.i", 0
; CHECK-NEXT:   %"'ipc5.i" = bitcast i8* %"call'mi7.i" to double*
; CHECK-NEXT:   %1 = insertvalue [3 x double*] %0, double* %"'ipc5.i", 1
; CHECK-NEXT:   %"'ipc6.i" = bitcast i8* %"call'mi8.i" to double*
; CHECK-NEXT:   %2 = insertvalue [3 x double*] %1, double* %"'ipc6.i", 2
; CHECK-NEXT:   %3 = bitcast i8* %call.i to double*
; CHECK-NEXT:   store double %x, double* %3
; CHECK-NEXT:   %call1.i = tail call fastcc double @augmented_f(double %x)
; CHECK-NEXT:   %factor = fmul fast double %call1.i, 2.000000e+00
; CHECK-NEXT:   %4 = insertvalue [3 x double] undef, double %factor, 0
; CHECK-NEXT:   %5 = insertvalue [3 x double] %4, double %factor, 1
; CHECK-NEXT:   %6 = insertvalue [3 x double] %5, double %factor, 2
; CHECK-NEXT:   tail call fastcc void @diffe3f([3 x double*] %2, [3 x double] %6)
; CHECK-NEXT:   store double 0.000000e+00, double* %"'ipc.i"
; CHECK-NEXT:   store double 0.000000e+00, double* %"'ipc5.i"
; CHECK-NEXT:   store double 0.000000e+00, double* %"'ipc6.i"
; CHECK-NEXT:   tail call void bitcast (i32 (...)* @free to void (i8*)*)(i8* nonnull %"call'mi.i")
; CHECK-NEXT:   tail call void bitcast (i32 (...)* @free to void (i8*)*)(i8* nonnull %"call'mi7.i")
; CHECK-NEXT:   tail call void bitcast (i32 (...)* @free to void (i8*)*)(i8* nonnull %"call'mi8.i")
; CHECK-NEXT:   tail call void bitcast (i32 (...)* @free to void (i8*)*)(i8* %call.i)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal fastcc void @diffe3f([3 x double*] %"x'", [3 x double] %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = extractvalue [3 x double*] %"x'", 0
; CHECK-NEXT:   %1 = load double, double* %0
; CHECK-NEXT:   %2 = extractvalue [3 x double*] %"x'", 1
; CHECK-NEXT:   %3 = load double, double* %2
; CHECK-NEXT:   %4 = extractvalue [3 x double*] %"x'", 2
; CHECK-NEXT:   %5 = load double, double* %4
; CHECK-NEXT:   %6 = extractvalue [3 x double] %differeturn, 0
; CHECK-NEXT:   %7 = fadd fast double %1, %6
; CHECK-NEXT:   %8 = extractvalue [3 x double] %differeturn, 1
; CHECK-NEXT:   %9 = fadd fast double %3, %8
; CHECK-NEXT:   %10 = extractvalue [3 x double] %differeturn, 2
; CHECK-NEXT:   %11 = fadd fast double %5, %10
; CHECK-NEXT:   store double %7, double* %0
; CHECK-NEXT:   store double %9, double* %2
; CHECK-NEXT:   store double %11, double* %4
; CHECK-NEXT:   ret void
; CHECK-NEXT: }