; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

@global = private unnamed_addr constant [1 x void (double*)*] [void (double*)* @ipmul]

@.str = private unnamed_addr constant [6 x i8] c"x=%f\0A\00", align 1
@.str.1 = private unnamed_addr constant [7 x i8] c"xp=%f\0A\00", align 1

define void @ipmul(double* %x) {
entry:
  %0 = load double, double* %x, align 8, !tbaa !2
  %mul = fmul fast double %0, %0
  store double %mul, double* %x
  ret void
}

; Function Attrs: noinline norecurse nounwind readonly uwtable
define dso_local double @mulglobal(double %x, i64 %idx) #0 {
entry:
  %alloc = alloca double
  store double %x, double* %alloc
  %arrayidx = getelementptr inbounds [1 x void (double*)*], [1 x void (double*)*]* @global, i64 0, i64 %idx
  %fp = load void (double*)*, void (double*)** %arrayidx, align 8
  call void %fp(double* %alloc)
  %ret = load double, double* %alloc, !tbaa !2
  ret double %ret
}

; Function Attrs: noinline nounwind uwtable
define dso_local double @derivative(double %x) local_unnamed_addr #1 {
entry:
  %0 = tail call double (double (double, i64)*, ...) @__enzyme_fwddiff(double (double, i64)* nonnull @mulglobal, double %x, double 1.0, i64 0)
  ret double %0
}

; Function Attrs: nounwind
declare double @__enzyme_fwddiff(double (double, i64)*, ...) #2

; Function Attrs: nounwind uwtable
define dso_local void @main(i32 %argc, i8** nocapture readonly %argv) local_unnamed_addr #3 {
entry:
  %arrayidx = getelementptr inbounds i8*, i8** %argv, i64 1
  %0 = load i8*, i8** %arrayidx, align 8, !tbaa !6
  %call.i = tail call fast double @strtod(i8* nocapture nonnull %0, i8** null) #2
  %call1 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str, i64 0, i64 0), double %call.i)
  %call2 = tail call fast double @derivative(double %call.i)
  %call3 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.1, i64 0, i64 0), double %call2)
  ret void
}

; Function Attrs: nounwind
declare dso_local i32 @printf(i8* nocapture readonly, ...) local_unnamed_addr #4

; Function Attrs: nounwind
declare dso_local double @strtod(i8* readonly, i8** nocapture) local_unnamed_addr #4

attributes #0 = { noinline norecurse nounwind readonly uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #2 = { nounwind }
attributes #3 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #4 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }

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


; CHECK: @global_shadow = private unnamed_addr constant [1 x void (double*)*] [void (double*)* bitcast (void (double*, double*)** @"_enzyme_forward_ipmul'" to void (double*)*)]
; CHECK:@"_enzyme_forward_ipmul'" = internal constant void (double*, double*)* @fwddiffeipmul

; CHECK: define internal double @fwddiffemulglobal(double %x, double %"x'", i64 %idx)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"alloc'ipa" = alloca double
; CHECK-NEXT:   store double 0.000000e+00, double* %"alloc'ipa"
; CHECK-NEXT:   %alloc = alloca double
; CHECK-NEXT:   store double %x, double* %alloc
; CHECK-NEXT:   store double %"x'", double* %"alloc'ipa"
; CHECK-NEXT:   %"arrayidx'ipg" = getelementptr inbounds [1 x void (double*)*], [1 x void (double*)*]* @global_shadow, i64 0, i64 %idx
; CHECK-NEXT:   %arrayidx = getelementptr inbounds [1 x void (double*)*], [1 x void (double*)*]* @global, i64 0, i64 %idx
; CHECK-NEXT:   %"fp'ipl" = load void (double*)*, void (double*)** %"arrayidx'ipg", align 8
; CHECK-NEXT:   %fp = load void (double*)*, void (double*)** %arrayidx, align 8
; CHECK-NEXT:   %0 = bitcast void (double*)* %fp to i8*
; CHECK-NEXT:   %1 = bitcast void (double*)* %"fp'ipl" to i8*
; CHECK-NEXT:   %2 = icmp eq i8* %0, %1
; CHECK-NEXT:   br i1 %2, label %error.i, label %__enzyme_runtimeinactiveerr.exit

; CHECK: error.i:                                          ; preds = %entry
; CHECK-NEXT:   %3 = call i32 @puts(i8* getelementptr inbounds ([79 x i8], [79 x i8]* @.str.2, i32 0, i32 0))
; CHECK-NEXT:   call void @exit(i32 1)
; CHECK-NEXT:   unreachable

; CHECK: __enzyme_runtimeinactiveerr.exit:
; CHECK-NEXT:   %[[a0:.+]] = bitcast void (double*)* %"fp'ipl" to void (double*, double*)**
; CHECK-NEXT:   %[[a1:.+]] = load void (double*, double*)*, void (double*, double*)** %[[a0]]
; CHECK-NEXT:   call void %[[a1]](double* %alloc, double* %"alloc'ipa")
; CHECK-NEXT:   %[[i2:.+]] = load double, double* %"alloc'ipa"
; CHECK-NEXT:   ret double %[[i2]]
; CHECK-NEXT: }
