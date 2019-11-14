; RUN: %opt < %s %loadEnzyme -enzyme -enzyme_preopt=false -mem2reg -instsimplify -simplifycfg -S -early-cse -instcombine -instsimplify | FileCheck %s

; Function Attrs: norecurse nounwind readnone uwtable
define dso_local double @square(double %x) #0 {
entry:
  %mul = fmul fast double %x, %x
  ret double %mul
}

; Function Attrs: noinline nounwind uwtable
define dso_local double @indirect(double (double)* nocapture %callee, double %x) local_unnamed_addr #1 {
entry:
  %call = tail call fast double %callee(double %x) #2
  ret double %call
}

; Function Attrs: noinline nounwind uwtable
define dso_local double @create(double %x) #1 {
entry:
  %call = tail call fast double @indirect(double (double)* nonnull @square, double %x)
  ret double %call
}

; Function Attrs: noinline nounwind uwtable
define dso_local double @derivative(double %x) local_unnamed_addr #1 {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_autodiff(double (double)* nonnull @create, double %x)
  ret double %0
}

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double)*, ...) #2

attributes #0 = { norecurse nounwind readnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #2 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}

; CHECK: define dso_local double @derivative(double %x)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call { double } @diffecreate(double %x, double 1.000000e+00)
; CHECK-NEXT:   %1 = extractvalue { double } %0, 0
; CHECK-NEXT:   ret double %1
; CHECK-NEXT: }

; CHECK: define internal {{(dso_local )?}}{ double } @diffecreate(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call { double } @diffeindirect(double (double)* nonnull @square, double (double)* bitcast ({ double } (double, double)* @diffesquare to double (double)*), double %x, double %differeturn)
; CHECK-NEXT:   ret { double } %0
; CHECK-NEXT: }

; CHECK: define internal {{(dso_local )?}}{ double } @diffeindirect(double (double)* nocapture %callee, double (double)* %"callee'", double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %call = tail call fast double %callee(double %x)
; CHECK-NEXT:   %0 = bitcast double (double)* %"callee'" to { double } (double, double)*
; CHECK-NEXT:   %1 = call { double } %0(double %x, double %differeturn)
; CHECK-NEXT:   ret { double } %1
; CHECK-NEXT: }
