; RUN: opt < %s %loadEnzyme -enzyme -inline -mem2reg -instsimplify -adce -loop-deletion -correlated-propagation -simplifycfg -S | FileCheck %s

; Function Attrs: norecurse nounwind readonly uwtable
define dso_local double @sum(double* nocapture readonly %x, i64 %n) #0 {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret double %add

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %extra ]
  %total.07 = phi double [ 0.000000e+00, %entry ], [ %add, %extra ]
  %arrayidx = getelementptr inbounds double, double* %x, i64 %indvars.iv
  %0 = load double, double* %arrayidx, align 8
  %add = fadd fast double %0, %total.07
  %indvars.iv.next = add nuw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv, %n
  br i1 %exitcond, label %for.cond.cleanup, label %extra

extra:
  br label %for.body
}

; Function Attrs: nounwind uwtable
define dso_local void @dsum(double* %x, double* %xp, i64 %n) local_unnamed_addr #1 {
entry:
  %0 = tail call double (double (double*, i64)*, ...) @__enzyme_autodiff(double (double*, i64)* nonnull @sum, double* %x, double* %xp, i64 %n)
  ret void
}

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double*, i64)*, ...) #2

attributes #0 = { norecurse nounwind readonly uwtable }
attributes #1 = { nounwind uwtable } 
attributes #2 = { nounwind }


; CHECK: define dso_local void @dsum(double* %x, double* %xp, i64 %n) 
; CHECK-NEXT: entry:
; CHECK-NEXT:   br label %invertfor.body.i
; CHECK: invertfor.body.i:                                 ; preds = %invertfor.body.i, %entry
; CHECK-NEXT:   %"indvars.iv'phi.i" = phi i64 [ %n, %entry ], [ %0, %invertfor.body.i ]
; CHECK-NEXT:   %0 = sub i64 %"indvars.iv'phi.i", 1
; CHECK-NEXT:   %"arrayidx'ipg.i" = getelementptr double, double* %xp, i64 %"indvars.iv'phi.i"
; CHECK-NEXT:   %1 = load double, double* %"arrayidx'ipg.i"
; CHECK-NEXT:   %2 = fadd fast double %1, 1.000000e+00
; CHECK-NEXT:   store double %2, double* %"arrayidx'ipg.i"
; CHECK-NEXT:   %3 = icmp ne i64 %"indvars.iv'phi.i", 0
; CHECK-NEXT:   br i1 %3, label %invertfor.body.i, label %diffesum.exit
; CHECK: diffesum.exit:                                    ; preds = %invertfor.body.i
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
