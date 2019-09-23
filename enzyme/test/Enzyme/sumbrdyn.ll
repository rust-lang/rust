; RUN: opt < %s %loadEnzyme -enzyme -enzyme_preopt=false -inline -mem2reg -instsimplify -adce -loop-deletion -correlated-propagation -S | FileCheck %s

declare i1 @exitcond();

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
  %exitcond = call i1 @exitcond();
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
; CHECK-NEXT:   br label %for.body.i

; CHECK: for.cond.cleanup.i:                               ; preds = %for.body.i
; CHECK-NEXT:   br label %invertfor.body.i

; CHECK: for.body.i:                                       ; preds = %extra.i, %entry
; CHECK-NEXT:   %0 = phi i64 [ %1, %extra.i ], [ 0, %entry ]
; CHECK-NEXT:   %exitcond.i = call i1 @exitcond() #2
; CHECK-NEXT:   br i1 %exitcond.i, label %for.cond.cleanup.i, label %extra.i

; CHECK: extra.i:                                          ; preds = %for.body.i
; CHECK-NEXT:   %1 = add nuw i64 %0, 1
; CHECK-NEXT:   br label %for.body.i

; CHECK: invertfor.body.i:                                 ; preds = %invertextra.i, %for.cond.cleanup.i
; CHECK-NEXT:   %"'phi.i" = phi i64 [ %0, %for.cond.cleanup.i ], [ %2, %invertextra.i ]
; CHECK-NEXT:   %2 = sub i64 %"'phi.i", 1
; CHECK-NEXT:   %"arrayidx'ipg.i" = getelementptr double, double* %xp, i64 %"'phi.i"
; CHECK-NEXT:   %3 = load double, double* %"arrayidx'ipg.i"
; CHECK-NEXT:   %4 = fadd fast double %3, 1.000000e+00
; CHECK-NEXT:   store double %4, double* %"arrayidx'ipg.i"
; CHECK-NEXT:   %5 = icmp ne i64 %"'phi.i", 0
; CHECK-NEXT:   br i1 %5, label %invertextra.i, label %diffesum.exit

; CHECK: invertextra.i:                                    ; preds = %invertfor.body.i
; CHECK-NEXT:   br label %invertfor.body.i

; CHECK: diffesum.exit:                                    ; preds = %invertfor.body.i
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
