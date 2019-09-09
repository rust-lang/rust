; RUN: opt < %s %loadEnzyme -enzyme -mem2reg -instsimplify -simplifycfg -early-cse -dce -correlated-propagation -instcombine -loop-deletion -simplifycfg -S | FileCheck %s

@.str = private unnamed_addr constant [21 x i8] c"ffib'(n=%d, i=1)=%f\0A\00", align 1

; Function Attrs: nounwind readnone uwtable
define dso_local double @ffib(i32 %n, double %x) #0 {
entry:
  %cmp7 = icmp slt i32 %n, 3
  br i1 %cmp7, label %return, label %if.end

if.end:                                           ; preds = %if.end, %entry
  %n.tr9 = phi i32 [ %sub1, %if.end ], [ %n, %entry ]
  %accumulator.tr8 = phi double [ %add, %if.end ], [ %x, %entry ]
  %sub = add nsw i32 %n.tr9, -1
  %call = tail call fast double @ffib(i32 %sub, double %x)
  %sub1 = add nsw i32 %n.tr9, -2
  %add = fadd fast double %call, %accumulator.tr8
  %cmp = icmp slt i32 %n.tr9, 5
  br i1 %cmp, label %return, label %if.end

return:                                           ; preds = %if.end, %entry
  %accumulator.tr.lcssa = phi double [ %x, %entry ], [ %add, %if.end ]
  ret double %accumulator.tr.lcssa
}

; Function Attrs: nounwind uwtable
define dso_local i32 @main(i32 %argc, i8** nocapture readonly %argv) local_unnamed_addr #1 {
entry:
  %arrayidx = getelementptr inbounds i8*, i8** %argv, i64 1
  %0 = load i8*, i8** %arrayidx, align 8, !tbaa !2
  %call.i = tail call i64 @strtol(i8* nocapture nonnull %0, i8** null, i32 10) #3
  %conv.i = trunc i64 %call.i to i32
  %1 = tail call double (double (i32, double)*, ...) @__enzyme_autodiff(double (i32, double)* nonnull @ffib, i32 %conv.i, double 1.000000e+00)
  %call3 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([21 x i8], [21 x i8]* @.str, i64 0, i64 0), i32 %conv.i, double %1)
  ret i32 0
}

; Function Attrs: nounwind
declare dso_local i32 @printf(i8* nocapture readonly, ...) local_unnamed_addr #2

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (i32, double)*, ...) #3

; Function Attrs: nounwind
declare dso_local i64 @strtol(i8* readonly, i8** nocapture, i32) local_unnamed_addr #2

attributes #0 = { nounwind readnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #2 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #3 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0}
!3 = !{!"any pointer", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}

; CHECK: define internal {{(dso_local )?}}{ double } @diffeffib.1(i32 %n
