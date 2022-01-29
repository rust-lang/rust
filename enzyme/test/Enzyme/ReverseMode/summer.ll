; RUN: if [ %llvmver -le 12 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -inline -mem2reg -instsimplify -adce -loop-deletion -correlated-propagation -simplifycfg -adce -S | FileCheck %s ; fi
; RUN: if [ %llvmver -ge 13 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -inline -mem2reg -instsimplify -adce -loop-deletion -correlated-propagation -simplifycfg -adce -S | FileCheck %s --check-prefix=POST ; fi

@.str = private unnamed_addr constant [25 x i8] c"xs[%d] = %f xp[%d] = %f\0A\00", align 1
@.str.1 = private unnamed_addr constant [7 x i8] c"n != 0\00", align 1
@.str.2 = private unnamed_addr constant [9 x i8] c"summer.c\00", align 1
@__PRETTY_FUNCTION__.summer = private unnamed_addr constant [40 x i8] c"double summer(double *restrict, size_t)\00", align 1
@.str.3 = private unnamed_addr constant [19 x i8] c"i print things %f\0A\00", align 1
@.str.4 = private unnamed_addr constant [7 x i8] c"n != 1\00", align 1

; Function Attrs: noinline nounwind uwtable
define dso_local void @derivative(double* noalias %x, double* noalias %xp, i64 %n) local_unnamed_addr #0 {
entry:
  %0 = tail call double (double (double*, i64)*, ...) @__enzyme_autodiff(double (double*, i64)* nonnull @summer, double* %x, double* %xp, i64 %n)
  ret void
}

; Function Attrs: noinline nounwind uwtable
define internal double @summer(double* noalias nocapture readonly %x, i64 %n) #0 {
entry:
  %cmp = icmp eq i64 %n, 0
  br i1 %cmp, label %cond.false, label %cond.end

cond.false:                                       ; preds = %entry
  tail call void @__assert_fail(i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.1, i64 0, i64 0), i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str.2, i64 0, i64 0), i32 11, i8* getelementptr inbounds ([40 x i8], [40 x i8]* @__PRETTY_FUNCTION__.summer, i64 0, i64 0)) #6
  unreachable

cond.end:                                         ; preds = %entry
  %call = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([19 x i8], [19 x i8]* @.str.3, i64 0, i64 0), double 0.000000e+00)
  %cmp1 = icmp eq i64 %n, 1
  br i1 %cmp1, label %cond.false3, label %for.body.preheader

cond.false3:                                      ; preds = %cond.end
  tail call void @__assert_fail(i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.4, i64 0, i64 0), i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str.2, i64 0, i64 0), i32 13, i8* getelementptr inbounds ([40 x i8], [40 x i8]* @__PRETTY_FUNCTION__.summer, i64 0, i64 0)) #6
  unreachable

for.body.preheader:                               ; preds = %cond.end
  %0 = load double, double* %x, align 8, !tbaa !2
  br label %for.body.for.body_crit_edge

for.cond.cleanup:                                 ; preds = %for.body.for.body_crit_edge
  %sub = fsub fast double %0, %cond.i
  ret double %sub

for.body.for.body_crit_edge:                      ; preds = %for.body.for.body_crit_edge, %for.body.preheader
  %indvars.iv.next29 = phi i64 [ 1, %for.body.preheader ], [ %indvars.iv.next, %for.body.for.body_crit_edge ]
  %cond.i28 = phi double [ %0, %for.body.preheader ], [ %cond.i, %for.body.for.body_crit_edge ]
  %arrayidx9.phi.trans.insert = getelementptr inbounds double, double* %x, i64 %indvars.iv.next29
  %.pre = load double, double* %arrayidx9.phi.trans.insert, align 8, !tbaa !2
  %cmp.i = fcmp fast ogt double %cond.i28, %.pre
  %cond.i = select i1 %cmp.i, double %cond.i28, double %.pre
  %indvars.iv.next = add nuw i64 %indvars.iv.next29, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond, label %for.cond.cleanup, label %for.body.for.body_crit_edge
}

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double*, i64)*, ...) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #3

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #3

; Function Attrs: nounwind
declare dso_local i32 @printf(i8* nocapture readonly, ...) local_unnamed_addr #4

; Function Attrs: noreturn nounwind
declare dso_local void @__assert_fail(i8*, i8*, i32, i8*) local_unnamed_addr #5

attributes #0 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { nounwind }
attributes #2 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #3 = { argmemonly nounwind }
attributes #4 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #5 = { noreturn nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #6 = { noreturn nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0}
!3 = !{!"double", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}


; CHECK: define internal {{(dso_local )?}}void @diffesummer(double* noalias nocapture readonly %x, double* nocapture %"x'", i64 %n, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %cmp = icmp eq i64 %n, 0
; CHECK-NEXT:   br i1 %cmp, label %cond.false, label %cond.end

; CHECK: cond.false:                                       ; preds = %entry
; CHECK-NEXT:   tail call void @__assert_fail(i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.1, i64 0, i64 0), i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str.2, i64 0, i64 0), i32 11, i8* getelementptr inbounds ([40 x i8], [40 x i8]* @__PRETTY_FUNCTION__.summer, i64 0, i64 0))
; CHECK-NEXT:   unreachable

; CHECK: cond.end:                                         ; preds = %entry
; CHECK-NEXT:   %call = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([19 x i8], [19 x i8]* @.str.3, i64 0, i64 0), double 0.000000e+00)
; CHECK-NEXT:   %cmp1 = icmp eq i64 %n, 1
; CHECK-NEXT:   br i1 %cmp1, label %cond.false3, label %for.body.preheader

; CHECK: cond.false3:                                      ; preds = %cond.end
; CHECK-NEXT:   tail call void @__assert_fail(i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.4, i64 0, i64 0), i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str.2, i64 0, i64 0), i32 13, i8* getelementptr inbounds ([40 x i8], [40 x i8]* @__PRETTY_FUNCTION__.summer, i64 0, i64 0))
; CHECK-NEXT:   unreachable

; CHECK: for.body.preheader:                               ; preds = %cond.end
; CHECK-NEXT:   %0 = load double, double* %x, align 8, !tbaa !2
; CHECK-NEXT:   %[[nm2:.+]] = add i64 %n, -2
; CHECK-NEXT:   br label %for.body.for.body_crit_edge

; CHECK: for.body.for.body_crit_edge:                      ; preds = %for.body.for.body_crit_edge, %for.body.preheader
; CHECK-NEXT:   %[[idx:.+]] = phi i64 [ 0, %for.body.preheader ], [ %[[idx2:.+]], %for.body.for.body_crit_edge ]
; CHECK-NEXT:   %[[iv:.+]] = phi i64 [ %[[idxadd:.+]], %for.body.for.body_crit_edge ], [ 0, %for.body.preheader ]
; CHECK-NEXT:   %cond.i28 = phi double [ %0, %for.body.preheader ], [ %cond.i, %for.body.for.body_crit_edge ]
; CHECK-NEXT:   %[[idxadd:.+]] = add nuw nsw i64 %[[iv]], 1
; CHECK-NEXT:   %arrayidx9.phi.trans.insert = getelementptr inbounds double, double* %x, i64 %[[idxadd]]
; CHECK-NEXT:   %.pre = load double, double* %arrayidx9.phi.trans.insert, align 8, !tbaa !2
; CHECK-NEXT:   %cmp.i = fcmp fast ogt double %cond.i28, %.pre
; CHECK-NEXT:   %[[idx2]] = select i1 %cmp.i, i64 %[[idx]], i64 %iv.next
; CHECK-NEXT:   %cond.i = select{{( fast)?}} i1 %cmp.i, double %cond.i28, double %.pre
; CHECK-NEXT:   %indvars.iv.next = add nuw i64 %[[idxadd]], 1
; CHECK-NEXT:   %[[pcond:.+]] = icmp eq i64 %indvars.iv.next, %n
; CHECK-NEXT:   br i1 %[[pcond]], label %invertfor.cond.cleanup, label %for.body.for.body_crit_edge

; CHECK: invertfor.body.preheader:
; CHECK-NEXT:   %[[lastload:.+]] = load double, double* %"x'"
; CHECK-NEXT:   %[[output:.+]] = fadd fast double %[[lastload]], %[[decarry:.+]]
; CHECK-NEXT:   store double %[[output]], double* %"x'"
; CHECK-NEXT:   ret void

; CHECK: invertfor.cond.cleanup:
; CHECK-NEXT:   %[[negdiff:.+]] = {{(fsub fast double 0.000000e\+00,|fneg fast double)}} %differeturn
; CHECK-NEXT:   br label %invertfor.body.for.body_crit_edge

; CHECK: invertfor.body.for.body_crit_edge:
; CHECK-NEXT:   %[[antivar:.+]] = phi i64 [ %[[nm2]], %invertfor.cond.cleanup ], [ %[[subd:.+]], %incinvertfor.body.for.body_crit_edge ]
; CHECK-NEXT:   %[[nidx2:.+]] = add nuw nsw i64 %[[antivar]], 1
; CHECK-NEXT:   %[[reload:.+]] = icmp eq i64 %[[idx2]], %[[nidx2]]
; CHECK-NEXT:   %[[diffepre:.+]] = select{{( fast)?}} i1 %[[reload]], double %[[negdiff]], double 0.000000e+00
; CHECK-NEXT:   %[[arrayidx9phitransinsertipg:.+]] = getelementptr inbounds double, double* %"x'", i64 %[[nidx2]]
; CHECK-NEXT:   %[[loaded:.+]] = load double, double* %[[arrayidx9phitransinsertipg]]
; CHECK-NEXT:   %[[tostore:.+]] = fadd fast double %[[loaded]], %[[diffepre]]
; CHECK-NEXT:   store double %[[tostore]], double* %[[arrayidx9phitransinsertipg]]
; CHECK-NEXT:   %[[lcond:.+]] = icmp eq i64 %[[antivar]], 0
; CHECK-NEXT:   %[[first:.+]] = icmp eq i64 %[[idx2]], 0
; CHECK-NEXT:   %[[diffecond:.+]] = select{{( fast)?}} i1 %[[first]], double %[[negdiff]], double 0.000000e+00
; CHECK-NEXT:   %[[decarry]] = fadd fast double %differeturn, %[[diffecond]]
; CHECK-NEXT:   br i1 %[[lcond]], label %invertfor.body.preheader, label %incinvertfor.body.for.body_crit_edge

; CHECK: incinvertfor.body.for.body_crit_edge:
; CHECK-NEXT:   %[[subd]] = add nsw i64 %[[antivar]], -1
; CHECK-NEXT:   br label %invertfor.body.for.body_crit_edge
; CHECK-NEXT: }

; POST: define internal {{(dso_local )?}}void @diffesummer(double* noalias nocapture readonly %x, double* nocapture %"x'", i64 %n, double %differeturn)
; POST-NEXT: entry:
; POST-NEXT:   %cmp = icmp eq i64 %n, 0
; POST-NEXT:   %0 = xor i1 %cmp, true
; POST-NEXT:   call void @llvm.assume(i1 %0)
; POST-NEXT:   %call = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([19 x i8], [19 x i8]* @.str.3, i64 0, i64 0), double 0.000000e+00)
; POST-NEXT:   %cmp1 = icmp eq i64 %n, 1
; POST-NEXT:   %1 = xor i1 %cmp1, true
; POST-NEXT:   call void @llvm.assume(i1 %1)
; POST-NEXT:   %[[i0:.+]] = load double, double* %x, align 8, !tbaa !2
; POST-NEXT:   %[[nm2:.+]] = add i64 %n, -2
; POST-NEXT:   br label %for.body.for.body_crit_edge

; POST: for.body.for.body_crit_edge:
; POST-NEXT:   %[[idx:.+]] = phi i64 [ 0, %entry ], [ %[[idx2:.+]], %for.body.for.body_crit_edge ]
; POST-NEXT:   %[[iv:.+]] = phi i64 [ %[[idxadd:.+]], %for.body.for.body_crit_edge ], [ 0, %entry ]
; POST-NEXT:   %cond.i28 = phi double [ %[[i0]], %entry ], [ %cond.i, %for.body.for.body_crit_edge ]
; POST-NEXT:   %[[idxadd:.+]] = add nuw nsw i64 %[[iv]], 1
; POST-NEXT:   %arrayidx9.phi.trans.insert = getelementptr inbounds double, double* %x, i64 %[[idxadd]]
; POST-NEXT:   %.pre = load double, double* %arrayidx9.phi.trans.insert, align 8, !tbaa !2
; POST-NEXT:   %cmp.i = fcmp fast ogt double %cond.i28, %.pre
; POST-NEXT:   %[[idx2]] = select i1 %cmp.i, i64 %[[idx]], i64 %iv.next
; POST-NEXT:   %cond.i = select{{( fast)?}} i1 %cmp.i, double %cond.i28, double %.pre
; POST-NEXT:   %indvars.iv.next = add nuw i64 %[[idxadd]], 1
; POST-NEXT:   %[[pcond:.+]] = icmp eq i64 %indvars.iv.next, %n
; POST-NEXT:   br i1 %[[pcond]], label %invertfor.cond.cleanup, label %for.body.for.body_crit_edge

; POST: invertfor.body.preheader:
; POST-NEXT:   %[[lastload:.+]] = load double, double* %"x'"
; POST-NEXT:   %[[output:.+]] = fadd fast double %[[lastload]], %[[decarry:.+]]
; POST-NEXT:   store double %[[output]], double* %"x'"
; POST-NEXT:   ret void

; POST: invertfor.cond.cleanup:
; POST-NEXT:   %[[negdiff:.+]] = {{(fsub fast double 0.000000e\+00,|fneg fast double)}} %differeturn
; POST-NEXT:   br label %invertfor.body.for.body_crit_edge

; POST: invertfor.body.for.body_crit_edge:
; POST-NEXT:   %[[antivar:.+]] = phi i64 [ %[[nm2]], %invertfor.cond.cleanup ], [ %[[subd:.+]], %incinvertfor.body.for.body_crit_edge ]
; POST-NEXT:   %[[nidx2:.+]] = add nuw nsw i64 %[[antivar]], 1
; POST-NEXT:   %[[reload:.+]] = icmp eq i64 %[[idx2]], %[[nidx2]]
; POST-NEXT:   %[[diffepre:.+]] = select{{( fast)?}} i1 %[[reload]], double %[[negdiff]], double 0.000000e+00
; POST-NEXT:   %[[arrayidx9phitransinsertipg:.+]] = getelementptr inbounds double, double* %"x'", i64 %[[nidx2]]
; POST-NEXT:   %[[loaded:.+]] = load double, double* %[[arrayidx9phitransinsertipg]]
; POST-NEXT:   %[[tostore:.+]] = fadd fast double %[[loaded]], %[[diffepre]]
; POST-NEXT:   store double %[[tostore]], double* %[[arrayidx9phitransinsertipg]]
; POST-NEXT:   %[[lcond:.+]] = icmp eq i64 %[[antivar]], 0
; POST-NEXT:   %[[first:.+]] = icmp eq i64 %[[idx2]], 0
; POST-NEXT:   %[[diffecond:.+]] = select{{( fast)?}} i1 %[[first]], double %[[negdiff]], double 0.000000e+00
; POST-NEXT:   %[[decarry]] = fadd fast double %differeturn, %[[diffecond]]
; POST-NEXT:   br i1 %[[lcond]], label %invertfor.body.preheader, label %incinvertfor.body.for.body_crit_edge

; POST: incinvertfor.body.for.body_crit_edge:
; POST-NEXT:   %[[subd]] = add nsw i64 %[[antivar]], -1
; POST-NEXT:   br label %invertfor.body.for.body_crit_edge
; POST-NEXT: }
