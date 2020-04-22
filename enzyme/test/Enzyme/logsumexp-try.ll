; RUN: %opt < %s %loadEnzyme -enzyme -enzyme_preopt=false -inline -mem2reg -instsimplify -adce -loop-deletion -correlated-propagation -simplifycfg -S | FileCheck %s
; XFAIL: *

; Function Attrs: nounwind uwtable
define dso_local void @dsincos(double* noalias %x, double* noalias %xp, i64 %n) local_unnamed_addr #0 {
entry:
  %0 = tail call double (double (double*, i64)*, ...) @__enzyme_autodiff(double (double*, i64)* nonnull @logsumexp, double* %x, double* %xp, i64 %n)
  ret void
}

; Function Attrs: noinline nounwind readonly uwtable
define internal double @logsumexp(double* noalias nocapture readonly %x, i64 %n) #1 {
entry:
  %0 = load double, double* %x, align 8
  %cmp55 = icmp eq i64 %n, 0
  br i1 %cmp55, label %for.cond.cleanup22, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  %exitcond6473 = icmp eq i64 %n, 1
  br i1 %exitcond6473, label %for.cond.cleanup, label %for.body.for.body_crit_edge

for.cond.cleanup:                                 ; preds = %for.body.for.body_crit_edge, %for.body.preheader
  %cond.i.lcssa = phi double [ %0, %for.body.preheader ], [ %cond.i, %for.body.for.body_crit_edge ]
  %vla = alloca double, i64 %n, align 16
  br i1 %cmp55, label %for.cond.cleanup22, label %for.body9.preheader

for.body9.preheader:                              ; preds = %for.cond.cleanup
  %sub70 = fsub fast double %0, %cond.i.lcssa
  %1 = tail call fast double @llvm.exp.f64(double %sub70)
  store double %1, double* %vla, align 16
  %exitcond6171 = icmp eq i64 %n, 1
  br i1 %exitcond6171, label %for.cond18.preheader, label %for.body9.for.body9_crit_edge

for.body.for.body_crit_edge:                      ; preds = %for.body.preheader, %for.body.for.body_crit_edge
  %indvars.iv.next6375 = phi i64 [ %indvars.iv.next63, %for.body.for.body_crit_edge ], [ 1, %for.body.preheader ]
  %cond.i74 = phi double [ %cond.i, %for.body.for.body_crit_edge ], [ %0, %for.body.preheader ]
  %arrayidx2.phi.trans.insert = getelementptr inbounds double, double* %x, i64 %indvars.iv.next6375
  %.pre = load double, double* %arrayidx2.phi.trans.insert, align 8
  %cmp.i = fcmp fast ogt double %cond.i74, %.pre
  %cond.i = select i1 %cmp.i, double %cond.i74, double %.pre
  %indvars.iv.next63 = add nuw i64 %indvars.iv.next6375, 1
  %exitcond64 = icmp eq i64 %indvars.iv.next63, %n
  br i1 %exitcond64, label %for.cond.cleanup, label %for.body.for.body_crit_edge

for.cond18.preheader:                             ; preds = %for.body9.for.body9_crit_edge, %for.body9.preheader
  br i1 %cmp55, label %for.cond.cleanup22, label %for.body23

for.body9.for.body9_crit_edge:                    ; preds = %for.body9.preheader, %for.body9.for.body9_crit_edge
  %indvars.iv.next6072 = phi i64 [ %indvars.iv.next60, %for.body9.for.body9_crit_edge ], [ 1, %for.body9.preheader ]
  %arrayidx11.phi.trans.insert = getelementptr inbounds double, double* %x, i64 %indvars.iv.next6072
  %.pre65 = load double, double* %arrayidx11.phi.trans.insert, align 8
  %sub = fsub fast double %.pre65, %cond.i.lcssa
  %2 = tail call fast double @llvm.exp.f64(double %sub)
  %arrayidx13 = getelementptr inbounds double, double* %vla, i64 %indvars.iv.next6072
  store double %2, double* %arrayidx13, align 8
  %indvars.iv.next60 = add nuw i64 %indvars.iv.next6072, 1
  %exitcond61 = icmp eq i64 %indvars.iv.next60, %n
  br i1 %exitcond61, label %for.cond18.preheader, label %for.body9.for.body9_crit_edge

for.cond.cleanup22:                               ; preds = %for.body23, %entry, %for.cond.cleanup, %for.cond18.preheader
  %A.0.lcssa6769 = phi double [ %cond.i.lcssa, %for.cond18.preheader ], [ %cond.i.lcssa, %for.cond.cleanup ], [ %0, %entry ], [ %cond.i.lcssa, %for.body23 ]
  %sema.0.lcssa = phi double [ 0.000000e+00, %for.cond18.preheader ], [ 0.000000e+00, %for.cond.cleanup ], [ 0.000000e+00, %entry ], [ %add, %for.body23 ]
  %3 = tail call fast double @llvm.log.f64(double %sema.0.lcssa)
  %add29 = fadd fast double %3, %A.0.lcssa6769
  ret double %add29

for.body23:                                       ; preds = %for.cond18.preheader, %for.body23
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body23 ], [ 0, %for.cond18.preheader ]
  %sema.051 = phi double [ %add, %for.body23 ], [ 0.000000e+00, %for.cond18.preheader ]
  %arrayidx25 = getelementptr inbounds double, double* %vla, i64 %indvars.iv
  %4 = load double, double* %arrayidx25, align 8
  %add = fadd fast double %4, %sema.051
  %indvars.iv.next = add nuw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond, label %for.cond.cleanup22, label %for.body23
}

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double*, i64)*, ...) #2

; Function Attrs: nounwind readnone speculatable
declare double @llvm.exp.f64(double) #3

; Function Attrs: nounwind readnone speculatable
declare double @llvm.log.f64(double) #3

attributes #0 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { noinline nounwind readonly uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #2 = { nounwind }
attributes #3 = { nounwind readnone speculatable }

; CHECK: define dso_local void @dsincos(double* noalias %x, double* noalias %xp, i64 %n)
