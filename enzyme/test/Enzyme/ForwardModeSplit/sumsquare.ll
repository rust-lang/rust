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
define dso_local double @dsumsquare(double* %x, double* %xp, i64 %n, i8* %tapeArg) local_unnamed_addr #1 {
entry:
  %0 = tail call double (double (double*, i64)*, ...) @__enzyme_fwdsplit(double (double*, i64)* nonnull @sumsquare, metadata !"enzyme_nofree", double* %x, double* %xp, i64 %n, i8* %tapeArg)
  ret double %0
}

; Function Attrs: nounwind
declare double @__enzyme_fwdsplit(double (double*, i64)*, ...) #2

attributes #0 = { norecurse nounwind readonly uwtable }
attributes #1 = { nounwind uwtable }
attributes #2 = { nounwind }


; CHECK: define {{(dso_local )?}}double @dsumsquare(double* %x, double* %xp, i64 %n, i8* %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = bitcast i8* %tapeArg to double**
; CHECK-NEXT:   %truetape.i = load double*, double** %0
; CHECK-NEXT:   br label %for.body.i

; CHECK: for.body.i:                                       ; preds = %for.body.i, %entry
; CHECK-DAG:   %iv.i = phi i64 [ %iv.next.i, %for.body.i ], [ 0, %entry ]
; CHECK-DAG:   %[[total011:.+]] = phi {{(fast )?}}double [ 0.000000e+00, %entry ], [ %[[i6:.+]], %for.body.i ]
; CHECK-NEXT:   %iv.next.i = add nuw nsw i64 %iv.i, 1
; CHECK-NEXT:   %"arrayidx'ipg.i" = getelementptr inbounds double, double* %xp, i64 %iv.i
; CHECK-NEXT:   %[[i3:.+]] = load double, double* %"arrayidx'ipg.i", align 8
; CHECK-NEXT:   %[[i1:.+]] = getelementptr inbounds double, double* %truetape.i, i64 %iv.i
; CHECK-NEXT:   %[[i2:.+]] = load double, double* %[[i1]], align 8, !invariant.group !1
; CHECK-NEXT:   %[[i4:.+]] = fmul fast double %[[i3]], %[[i2]]
; CHECK-NEXT:   %[[i5:.+]] = fadd fast double %[[i4]], %[[i4]]
; CHECK-NEXT:   %[[i6]] = fadd fast double %[[i5]], %[[total011]]
; CHECK-NEXT:   %exitcond.i = icmp eq i64 %iv.i, %n
; CHECK-NEXT:   br i1 %exitcond.i, label %fwddiffesumsquare.exit, label %for.body.i

; CHECK: fwddiffesumsquare.exit:                              ; preds = %for.body.i
; CHECK-NEXT:   ret double %[[i6]]
; CHECK-NEXT: }
