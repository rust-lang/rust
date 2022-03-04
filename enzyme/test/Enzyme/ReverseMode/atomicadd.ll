; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -adce -loop-deletion -correlated-propagation -simplifycfg -S | FileCheck %s

; Function Attrs: norecurse nounwind readonly uwtable
define dso_local double @sum(i64* nocapture %n, double %x) #0 {
entry:
  %res = atomicrmw add i64* %n, i64 1 monotonic
  %fp = uitofp i64 %res to double
  %mul = fmul double %fp, %x
  ret double %mul
}

; Function Attrs: nounwind uwtable
define dso_local void @dsum(i64* %x, i64* %xp, double %n) local_unnamed_addr #1 {
entry:
  %0 = tail call double (double (i64*, double)*, ...) @__enzyme_autodiff(double (i64*, double)* nonnull @sum, i64* %x, double %n)
  ret void
}

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (i64*, double)*, ...) #2

attributes #0 = { norecurse nounwind readonly uwtable }
attributes #1 = { nounwind uwtable }
attributes #2 = { nounwind }

; CHECK: define internal { double } @diffesum(i64* nocapture %n, double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %res = atomicrmw add i64* %n, i64 1 monotonic
; CHECK-NEXT:   %fp = uitofp i64 %res to double
; CHECK-NEXT:   %m1diffex = fmul fast double %differeturn, %fp
; CHECK-NEXT:   %0 = insertvalue { double } undef, double %m1diffex, 0
; CHECK-NEXT:   ret { double } %0
; CHECK-NEXT: }
