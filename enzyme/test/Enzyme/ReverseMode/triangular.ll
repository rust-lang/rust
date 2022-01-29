; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -inline -mem2reg -gvn -instsimplify -adce -loop-deletion -correlated-propagation -simplifycfg -licm -early-cse -simplifycfg -instsimplify -S | FileCheck %s

; Function Attrs: noinline norecurse nounwind uwtable
define dso_local double @get(double* nocapture %x, i64 %i, i64 %j) local_unnamed_addr #0 {
entry:
  %arrayidx = getelementptr inbounds double, double* %x, i64 %i
  %0 = load double, double* %arrayidx, align 8
  ret double %0
}

; Function Attrs: noinline norecurse nounwind uwtable
define dso_local double @f(double* nocapture %x, i64 %n) #0 {
entry:
  br label %for.cond3.preheader

for.cond3.preheader:                              ; preds = %entry, %for.cond.cleanup6
  %i = phi i64 [ %i_inc, %for.cond.cleanup6 ], [ 0, %entry ]
  %outersum = phi double [ %sum.1.lcssa, %for.cond.cleanup6 ], [ 0.000000e+00, %entry ]
  %i_inc = add nuw i64 %i, 1
; note this is now technically not exactly triangular
;  %cmp423 = icmp eq i64 %i, 0
;  br i1 %cmp423, label %for.cond.cleanup6, label %for.body7
  br label %for.body7

for.body7:                                        ; preds = %for.cond3.preheader, %for.body7
  %j = phi i64 [ %j_inc, %for.body7 ], [ 0, %for.cond3.preheader ]
  %innersum = phi double [ %add, %for.body7 ], [ %outersum, %for.cond3.preheader ]
  %call = tail call fast double @get(double* %x, i64 undef, i64 %j)
  %mul = fmul fast double %call, %call
  %add = fadd fast double %mul, %innersum
  %j_inc = add nuw i64 %j, 1
  %exitcond = icmp eq i64 %j, %i
  br i1 %exitcond, label %for.cond.cleanup6, label %for.body7

for.cond.cleanup6:                                ; preds = %for.body7
  %sum.1.lcssa = phi double [ %add, %for.body7 ]
  %cmp1 = icmp eq i64 %i, %n
  br i1 %cmp1, label %return, label %for.cond3.preheader

return:                                           ; preds = %for.cond.cleanup6
  %retval.0 = phi double [ %sum.1.lcssa, %for.cond.cleanup6 ]
  ret double %retval.0
}

; Function Attrs: noinline nounwind uwtable
define dso_local double @dsumsquare(double* %x, double* %xp, i64 %n) local_unnamed_addr #1 {
entry:
  %call = tail call fast double @__enzyme_autodiff(i8* bitcast (double (double*, i64)* @f to i8*), double* %x, double* %xp, i64 %n)
  ret double %call
}

declare dso_local double @__enzyme_autodiff(i8*, double*, double*, i64) local_unnamed_addr

attributes #0 = { noinline norecurse nounwind uwtable }
attributes #1 = { noinline nounwind uwtable }

; CHECK: define internal void @diffef(double* nocapture %x, double* nocapture %"x'", i64 %n, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[np1:.+]] = add nuw i64 %n, 1
; CHECK-NEXT:   %mallocsize = mul nuw nsw i64 %[[np1]], 8
; CHECK-NEXT:   %malloccall = tail call noalias nonnull i8* @malloc(i64 %mallocsize)
; CHECK-NEXT:   %call_malloccache = bitcast i8* %malloccall to double**
; CHECK-NEXT:   br label %for.cond3.preheader

; CHECK: for.cond3.preheader:                              ; preds = %for.cond.cleanup6, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %for.cond.cleanup6 ], [ 0, %entry ]
; CHECK-NEXT:   %iv.next = add nuw i64 %iv, 1
; CHECK-NEXT:   %[[mallocgep1:.+]] = getelementptr inbounds double*, double** %call_malloccache, i64 %iv
; CHECK-NEXT:   %[[mallocsize1:.+]] = mul nuw nsw i64 %iv.next, 8
; CHECK-NEXT:   %[[malloccall2:.+]] = tail call noalias nonnull i8* @malloc(i64 %[[mallocsize1]])
; CHECK-NEXT:   %[[call_malloccache3:.+]] = bitcast i8* %[[malloccall2]] to double*
; CHECK-NEXT:   store double* %[[call_malloccache3]], double** %[[mallocgep1]], align 8, !invariant.group !0
; CHECK-NEXT:   br label %for.body7

; CHECK: for.body7:                                        ; preds = %for.body7, %for.cond3.preheader
; CHECK-NEXT:   %iv1 = phi i64 [ %iv.next2, %for.body7 ], [ 0, %for.cond3.preheader ]
; CHECK-NEXT:   %iv.next2 = add nuw nsw i64 %iv1, 1
; CHECK-NEXT:   %[[augmented:.+]] = call fast double @augmented_get(double* %x, double* %"x'", i64 undef, i64 %iv1)
; CHECK-NEXT:   %[[mallocgep2:.+]] = getelementptr inbounds double, double* %[[call_malloccache3]], i64 %iv1
; CHECK-NEXT:   store double %[[augmented]], double* %[[mallocgep2]], align 8, !invariant.group !1
; CHECK-NEXT:   %exitcond = icmp eq i64 %iv1, %iv
; CHECK-NEXT:   br i1 %exitcond, label %for.cond.cleanup6, label %for.body7

; CHECK: for.cond.cleanup6:                                ; preds = %for.body7
; CHECK-NEXT:   %cmp1 = icmp eq i64 %iv, %n
; CHECK-NEXT:   br i1 %cmp1, label %invertfor.cond.cleanup6, label %for.cond3.preheader

; CHECK: invertentry:                                      ; preds = %invertfor.cond3.preheader
; CHECK-NEXT:   tail call void @free(i8* nonnull %malloccall)
; CHECK-NEXT:   ret void

; CHECK: invertfor.cond3.preheader:
; CHECK-NEXT:   %[[done1:.+]] = icmp eq i64 %[[iv:.+]], 0
; CHECK-NEXT:   %[[innerdatai8:.+]] = bitcast double* %[[innerdata:.+]] to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[innerdatai8]])
; CHECK-NEXT:   br i1 %[[done1]], label %invertentry, label %incinvertfor.cond3.preheader

; CHECK: incinvertfor.cond3.preheader:
; CHECK-NEXT:   %[[subouter:.+]] = add nsw i64 %[[iv]], -1
; CHECK-NEXT:   br label %invertfor.cond.cleanup6

; CHECK: invertfor.body7:
; CHECK-NEXT:   %[[iv1:.+]] = phi i64 [ %[[iv]], %invertfor.cond.cleanup6 ], [ %[[subinner:.+]], %incinvertfor.body7 ]
; CHECK-NEXT:   %[[invertedgep2:.+]] = getelementptr inbounds double, double* %[[innerdata]], i64 %[[iv1]]
; CHECK-NEXT:   %[[cached:.+]] = load double, double* %[[invertedgep2]], align 8, !invariant.group !1
; CHECK-NEXT:   %m0diffecall = fmul fast double %differeturn, %[[cached]]
; CHECK-NEXT:   %[[innerdiffe:.+]] = fadd fast double %m0diffecall, %m0diffecall
; CHECK-NEXT:   call void @diffeget(double* %x, double* %"x'", i64 undef, i64 %[[iv1]], double %[[innerdiffe]])
; CHECK-NEXT:   %[[done2:.+]] = icmp eq i64 %[[iv1]], 0
; CHECK-NEXT:   br i1 %[[done2]], label %invertfor.cond3.preheader, label %incinvertfor.body7

; CHECK: incinvertfor.body7:
; CHECK-NEXT:   %[[subinner]] = add nsw i64 %[[iv1]], -1
; CHECK-NEXT:   br label %invertfor.body7

; CHECK: invertfor.cond.cleanup6:
; CHECK-NEXT:   %[[iv]] = phi i64 [ %[[subouter]], %incinvertfor.cond3.preheader ], [ %n, %for.cond.cleanup6 ]
; CHECK-NEXT:   %[[invertedgep1:.+]] = getelementptr inbounds double*, double** %call_malloccache, i64 %[[iv]]
; CHECK-NEXT:   %[[innerdata]] = load double*, double** %[[invertedgep1]], align 8, !invariant.group !0
; CHECK-NEXT:   br label %invertfor.body7
; CHECK-NEXT: }
