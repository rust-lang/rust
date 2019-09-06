; RUN: opt < %s %loadEnzyme -enzyme -mem2reg -sroa -instsimplify -simplifycfg -S | FileCheck %s

@.str = private unnamed_addr constant [4 x i8] c"%f\0A\00", align 1
@.str.1 = private unnamed_addr constant [5 x i8] c"%f \0A\00", align 1

; Function Attrs: noinline nounwind uwtable
declare dso_local double @read() local_unnamed_addr #0

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

; Function Attrs: nounwind
declare dso_local i32 @scanf(i8* nocapture readonly, ...) local_unnamed_addr #2

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

; Function Attrs: noinline nounwind uwtable
define dso_local double @sub(double %x) local_unnamed_addr #0 {
entry:
  %call = tail call fast double @read()
  %mul = fmul fast double %call, %x
  ret double %mul
}

; Function Attrs: noinline nounwind uwtable
declare dso_local double @read2() local_unnamed_addr #0

; Function Attrs: noinline nounwind uwtable
define dso_local double @foo(double %x) #0 {
entry:
  %call = tail call fast double @sub(double %x)
  %call1 = tail call fast double @read2()
  %add = fadd fast double %call1, %call
  ret double %add
}

; Function Attrs: nounwind uwtable
define dso_local double @dsumsquare(double %x) local_unnamed_addr #3 {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_autodiff(double (double)* nonnull @foo, double %x)
  ret double %0
}

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double)*, ...) #4

attributes #0 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #3 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #4 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0}
!3 = !{!"double", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}

; CHECK: define internal { double } @diffefoo(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call { { double }, double } @augmented_sub(double %x)
; CHECK-NEXT:   %[[tape:.+]] = extractvalue { { double }, double } %0, 0
; CHECK-NEXT:   %call1 = tail call fast double @read2()
; CHECK-NEXT:   %[[result:.+]] = call { double } @diffesub(double %x, double %differeturn, { double } %[[tape]])
; CHECK-NEXT:   ret { double } %[[result]]
; CHECK-NEXT: }

; CHECK: define internal { { double }, double } @augmented_sub(double %x)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %call = tail call fast double @read()
; CHECK-NEXT:   %mul = fmul fast double %call, %x
; CHECK-NEXT:   %[[insertcache:.+]] = insertvalue { { double }, double } undef, double %call, 0, 0
; CHECK-NEXT:   %[[insertreturn:.+]] = insertvalue { { double }, double } %[[insertcache]], double %mul, 1
; CHECK-NEXT:   ret { { double }, double } %[[insertreturn]]
; CHECK-NEXT: }

; CHECK: define internal { double } @diffesub(double %x, double %differeturn, { double } %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[extract:.+]] = extractvalue { double } %tapeArg, 0
; CHECK-NEXT:   %[[fmul:.+]] = fmul fast double %differeturn, %[[extract]]
; CHECK-NEXT:   %[[ret:.+]] = insertvalue { double } undef, double %[[fmul]], 0
; CHECK-NEXT:   ret { double } %[[ret]]
; CHECK-NEXT: }
