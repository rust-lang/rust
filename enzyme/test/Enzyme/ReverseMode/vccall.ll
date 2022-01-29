; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s
source_filename = "vccall.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline norecurse nounwind readnone uwtable
define dso_local double @h(double %x) local_unnamed_addr #0 {
entry:
  %square = fmul fast double %x, %x
  ret double %square
}

; Function Attrs: noinline norecurse nounwind readnone uwtable
define dso_local double @f(double %x) #0 {
entry:
  %call = tail call fast double @h(double %x)
  %mul = fmul fast double %call, %x
  ret double %mul
}

; Function Attrs: nounwind uwtable
define dso_local double @dsumsquare(double %x) local_unnamed_addr #1 {
entry:
  %call = tail call fast double @__enzyme_autodiff(i8* bitcast (double (double)* @f to i8*), double %x) #3
  ret double %call
}

declare dso_local double @__enzyme_autodiff(i8*, double) local_unnamed_addr #2

attributes #0 = { noinline norecurse nounwind readnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #3 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}

; CHECK: define internal {{(dso_local )?}}{ double } @diffef(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %call = tail call fast double @h(double %x)
; CHECK-NEXT:   %m0diffecall = fmul fast double %differeturn, %x
; CHECK-NEXT:   %m1diffex = fmul fast double %differeturn, %call
; CHECK-NEXT:   %0 = call { double } @diffeh(double %x, double %m0diffecall)
; CHECK-NEXT:   %1 = extractvalue { double } %0, 0
; CHECK-NEXT:   %2 = fadd fast double %m1diffex, %1
; CHECK-NEXT:   %3 = insertvalue { double } undef, double %2, 0
; CHECK-NEXT:   ret { double } %3
; CHECK-NEXT: }
