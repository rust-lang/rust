; RUN: opt < %s %loadEnzyme -enzyme -enzyme_preopt=false -mem2reg -instsimplify -simplifycfg -correlated-propagation -adce -instcombine -loop-unroll -instcombine -simplifycfg -S | FileCheck %s

; Function Attrs: noinline nounwind uwtable
define dso_local double @f(double* %x, i64 %n) #0 {
entry:
  %retval = alloca double, align 8
  %x.addr = alloca double*, align 8
  %n.addr = alloca i64, align 8
  %data = alloca double, align 8
  %i = alloca i32, align 4
  %cleanup.dest.slot = alloca i32, align 4
  store double* %x, double** %x.addr, align 8, !tbaa !2
  store i64 %n, i64* %n.addr, align 8, !tbaa !6
  %0 = bitcast double* %data to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %0) #4
  store double 0.000000e+00, double* %data, align 8, !tbaa !8
  %1 = bitcast i32* %i to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %1) #4
  store i32 0, i32* %i, align 4, !tbaa !10
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %2 = load i32, i32* %i, align 4, !tbaa !10
  %conv = sext i32 %2 to i64
  %3 = load i64, i64* %n.addr, align 8, !tbaa !6
  %cmp = icmp ule i64 %conv, %3
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  store i32 2, i32* %cleanup.dest.slot, align 4
  br label %cleanup

for.body:                                         ; preds = %for.cond
  %4 = load double, double* %data, align 8, !tbaa !8
  %cmp2 = fcmp fast ogt double %4, 1.000000e+01
  br i1 %cmp2, label %if.then, label %if.end

if.then:                                          ; preds = %for.body
  store i32 2, i32* %cleanup.dest.slot, align 4
  br label %cleanup

if.end:                                           ; preds = %for.body
  %5 = load double*, double** %x.addr, align 8, !tbaa !2
  %6 = load i32, i32* %i, align 4, !tbaa !10
  %idxprom = sext i32 %6 to i64
  %arrayidx = getelementptr inbounds double, double* %5, i64 %idxprom
  %7 = load double, double* %arrayidx, align 8, !tbaa !8
  %8 = load double, double* %data, align 8, !tbaa !8
  %add = fadd fast double %8, %7
  store double %add, double* %data, align 8, !tbaa !8
  br label %for.inc

for.inc:                                          ; preds = %if.end
  %9 = load i32, i32* %i, align 4, !tbaa !10
  %inc = add nsw i32 %9, 1
  store i32 %inc, i32* %i, align 4, !tbaa !10
  br label %for.cond

cleanup:                                          ; preds = %if.then, %for.cond.cleanup
  %10 = bitcast i32* %i to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %10) #4
  br label %for.end

for.end:                                          ; preds = %cleanup
  %11 = bitcast double* %data to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %11) #4
  %12 = load double, double* %retval, align 8
  ret double %12
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

; Function Attrs: noinline nounwind uwtable
define dso_local double @dsumsquare(double* %x, double* %xp, i64 %n) #0 {
entry:
  %call = call fast double @__enzyme_autodiff(i8* bitcast (double (double*, i64)* @f to i8*), double* %x, double* %xp, i64 %n)
  ret double %call
}

declare dso_local double @__enzyme_autodiff(i8*, double*, double*, i64) #2

attributes #0 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #3 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #4 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0}
!3 = !{!"any pointer", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
!6 = !{!7, !7, i64 0}
!7 = !{!"long long", !4, i64 0}
!8 = !{!9, !9, i64 0}
!9 = !{!"double", !4, i64 0}
!10 = !{!11, !11, i64 0}
!11 = !{!"int", !4, i64 0}

; CHECK: define internal {{(dso_local )?}}{} @diffef(double* %x, double* %"x'", i64 %n, double %differeturn)
