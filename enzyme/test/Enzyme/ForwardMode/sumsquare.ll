; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -inline -mem2reg -instsimplify -adce -loop-deletion -correlated-propagation -simplifycfg -early-cse -S | FileCheck %s

; Function Attrs: norecurse nounwind readonly uwtable
define dso_local double @sumsquare(double* nocapture readonly %x, i64 %n) #0 {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret double %add

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %total.011 = phi double [ 0.000000e+00, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds double, double* %x, i64 %indvars.iv
  %0 = load double, double* %arrayidx, align 8
  %mul = fmul fast double %0, %0
  %add = fadd fast double %mul, %total.011
  %indvars.iv.next = add nuw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv, %n
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

; Function Attrs: nounwind uwtable
define dso_local void @dsumsquare(double* %x, double* %xp, i64 %n) local_unnamed_addr #1 {
entry:
  %0 = tail call double (double (double*, i64)*, ...) @__enzyme_autodiff(double (double*, i64)* nonnull @sumsquare, double* %x, double* %xp, i64 %n)
  ret void
}

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double*, i64)*, ...) #2

attributes #0 = { norecurse nounwind readonly uwtable }
attributes #1 = { nounwind uwtable }
attributes #2 = { nounwind }


; CHECK: define dso_local void @dsumsquare(double* %x, double* %xp, i64 %n)
; CHECK-NEXT: entry:
; CHECK-NEXT:   br label %invertfor.body.i

; CHECK: invertfor.body.i:
; CHECK-NEXT:   %[[antiiv:.+]] = phi i64 [ %n, %entry ], [ %[[antiivnext:.+]], %incinvertfor.body.i ]
; CHECK-NEXT:   %[[ptr:.+]] = getelementptr inbounds double, double* %x, i64 %[[antiiv]]
; CHECK-NEXT:   %[[prev:.+]] = load double, double* %[[ptr]]
; CHECK-NEXT:   %[[times2:.+]] = fadd fast double %[[prev]], %[[prev]]
; CHECK-NEXT:   %[[arrayidxipgi:.+]] = getelementptr inbounds double, double* %xp, i64 %[[antiiv]]
; CHECK-NEXT:   %[[loaded:.+]] = load double, double* %[[arrayidxipgi]]
; CHECK-NEXT:   %[[tostore:.+]] = fadd fast double %[[loaded:.+]], %[[times2]]
; CHECK-NEXT:   store double %[[tostore]], double* %[[arrayidxipgi]]
; CHECK-NEXT:   %[[cmp:.+]] = icmp eq i64 %[[antiiv]], 0
; CHECK-NEXT:   br i1 %[[cmp]], label %diffesumsquare.exit, label %incinvertfor.body.i

; CHECK: incinvertfor.body.i:
; CHECK-NEXT:   %[[antiivnext]] = add nsw i64 %[[antiiv]], -1
; CHECK-NEXT:   br label %invertfor.body.i

; CHECK: diffesumsquare.exit:                                    ; preds = %invertfor.body.i
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
