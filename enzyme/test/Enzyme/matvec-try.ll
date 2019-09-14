; RUN: opt < %s %loadEnzyme -enzyme -enzyme_preopt=false -inline -mem2reg -instsimplify -adce -loop-deletion -correlated-propagation -simplifycfg -S | FileCheck %s

; Function Attrs: noinline norecurse nounwind uwtable
define dso_local void @matvec(i64 %N, i64 %M, double* noalias nocapture readonly %mat, double* noalias nocapture readonly %vec, double* noalias nocapture %out) #0 {
entry:
  %out43 = bitcast double* %out to i8*
  %cmp33 = icmp eq i64 %N, 0
  br i1 %cmp33, label %for.cond.cleanup, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %entry
  %cmp431 = icmp eq i64 %M, 0
  br i1 %cmp431, label %for.body.preheader, label %for.body.us

for.body.preheader:                               ; preds = %for.body.lr.ph
  %0 = shl i64 %N, 3
  call void @llvm.memset.p0i8.i64(i8* align 8 %out43, i8 0, i64 %0, i1 false)
  br label %for.cond.cleanup

for.body.us:                                      ; preds = %for.body.lr.ph, %for.cond2.for.cond.cleanup6_crit_edge.us
  %indvars.iv36 = phi i64 [ %indvars.iv.next37, %for.cond2.for.cond.cleanup6_crit_edge.us ], [ 0, %for.body.lr.ph ]
  %arrayidx.us = getelementptr inbounds double, double* %out, i64 %indvars.iv36
  store double 0.000000e+00, double* %arrayidx.us, align 8, !tbaa !2
  %mul.us = mul i64 %indvars.iv36, %M
  br label %for.body7.us

for.body7.us:                                     ; preds = %for.body7.us, %for.body.us
  %indvars.iv = phi i64 [ 0, %for.body.us ], [ %indvars.iv.next, %for.body7.us ]
  %1 = phi double [ 0.000000e+00, %for.body.us ], [ %add16.us, %for.body7.us ]
  %add.us = add i64 %indvars.iv, %mul.us
  %arrayidx10.us = getelementptr inbounds double, double* %mat, i64 %add.us
  %2 = load double, double* %arrayidx10.us, align 8, !tbaa !2
  %arrayidx12.us = getelementptr inbounds double, double* %vec, i64 %indvars.iv
  %3 = load double, double* %arrayidx12.us, align 8, !tbaa !2
  %mul13.us = fmul fast double %3, %2
  %add16.us = fadd fast double %1, %mul13.us
  %indvars.iv.next = add nuw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %M
  br i1 %exitcond, label %for.cond2.for.cond.cleanup6_crit_edge.us, label %for.body7.us

for.cond2.for.cond.cleanup6_crit_edge.us:         ; preds = %for.body7.us
  store double %add16.us, double* %arrayidx.us, align 8, !tbaa !2
  %indvars.iv.next37 = add nuw i64 %indvars.iv36, 1
  %exitcond38 = icmp eq i64 %indvars.iv.next37, %N
  br i1 %exitcond38, label %for.cond.cleanup, label %for.body.us

for.cond.cleanup:                                 ; preds = %for.cond2.for.cond.cleanup6_crit_edge.us, %for.body.preheader, %entry
  ret void
}

; Function Attrs: nounwind uwtable
define dso_local void @dsincos(i64 %N, i64 %M, double* noalias %mat, double* noalias %matp, double* noalias %vec, double* noalias %vecp, double* noalias %out, double* noalias %outp) local_unnamed_addr #1 {
entry:
  %0 = tail call double (void (i64, i64, double*, double*, double*)*, ...) @__enzyme_autodiff(void (i64, i64, double*, double*, double*)* nonnull @matvec, i64 %N, i64 %M, double* %mat, double* %matp, double* %vec, double* %vecp, double* %out, double* %outp)
  ret void
}

; Function Attrs: nounwind
declare double @__enzyme_autodiff(void (i64, i64, double*, double*, double*)*, ...) #2

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1) #3

attributes #0 = { noinline norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #2 = { nounwind }
attributes #3 = { argmemonly nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0}
!3 = !{!"double", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}

; CHECK: define dso_local void @dsincos(i64 %N, i64 %M, double* noalias %mat, double* noalias %matp, double* noalias %vec, double* noalias %vecp, double* noalias %out, double* noalias %outp)
