; RUN: %opt < %s %loadEnzyme -enzyme -enzyme_preopt=false -inline -mem2reg -gvn -instsimplify -adce -loop-deletion -correlated-propagation -simplifycfg -licm -early-cse -simplifycfg -instsimplify -S | FileCheck %s

; Function Attrs: noinline norecurse nounwind uwtable
define dso_local double @get(double* nocapture %x, i64 %i) local_unnamed_addr #0 {
entry:
  %arrayidx = getelementptr inbounds double, double* %x, i64 %i
  %0 = load double, double* %arrayidx, align 8
  store double 0.000000e+00, double* %arrayidx, align 8
  ret double %0
}

; Function Attrs: noinline norecurse nounwind uwtable
define dso_local double @f(double %x, i64 %n) #0 {
entry:
  br label %for.outerbody

for.outerbody:
  %i = phi i64 [ %i_inc, %for.body ], [ 0, %entry ]
  %outeradd = phi double [ %innersum, %for.body ], [ 0.000000e+00, %entry ]
  %i_inc = add nuw i64 %i, 1
  %iand = and i64 %i, 5678
  %exitcondi = icmp eq i64 %iand, %n
  br i1 %exitcondi, label %return, label %for.body.ph

for.body.ph:
  br label %for.body

for.body:
  %j = phi i64 [ %j_inc, %for.body ], [ 0, %for.body.ph ]
  %innersum = phi double [ %add, %for.body ], [ %outeradd, %for.body.ph ]
  %phi = phi double [ %phiadd, %for.body ], [ %x, %for.body.ph ]
  %mul = fmul fast double %phi, %phi
  %add = fadd fast double %mul, %innersum
  %phiadd = fadd fast double %phi, 1.000000e+00
  %j_inc = add nuw i64 %j, 1
  %jand = and i64 %j, 1234
  %exitcond = icmp eq i64 %jand, %n
  br i1 %exitcond, label %for.outerbody, label %for.body

return:
  %retval.0 = phi double [ %outeradd, %for.outerbody ]
  ret double %retval.0
}

; Function Attrs: noinline nounwind uwtable
define dso_local double @dsumsquare(double %x, i64 %n) local_unnamed_addr #1 {
entry:
  %call = tail call fast double @__enzyme_autodiff(i8* bitcast (double (double, i64)* @f to i8*), double %x, i64 %n)
  ret double %call
}

declare dso_local double @__enzyme_autodiff(i8*, double, i64) local_unnamed_addr

attributes #0 = { noinline norecurse nounwind uwtable }
attributes #1 = { noinline nounwind uwtable }

; CHECK: define internal { double } @diffef(double %x, i64 %n, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   br label %for.outerbody

; CHECK: for.outerbody.loopexit:                           ; preds = %for.body
; CHECK-NEXT:   %0 = getelementptr inbounds i64, i64* %loopLimit_realloccast, i64 %iv
; CHECK-NEXT:   store i64 %iv1, i64* %0, align 8, !invariant.group !0
; CHECK-NEXT:   br label %for.outerbody

; CHECK: for.outerbody:                                    ; preds = %for.outerbody.loopexit, %entry
; CHECK-NEXT:   %loopLimit_cache3.0 = phi i64* [ null, %entry ], [ %loopLimit_realloccast, %for.outerbody.loopexit ]
; CHECK-NEXT:   %phi_cache.0 = phi double** [ null, %entry ], [ %phi_realloccast, %for.outerbody.loopexit ]
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %for.outerbody.loopexit ], [ 0, %entry ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %1 = bitcast double** %phi_cache.0 to i8*
; CHECK-NEXT:   %2 = mul nuw nsw i64 8, %iv.next
; CHECK-NEXT:   %phi_realloccache = call i8* @realloc(i8* %1, i64 %2)
; CHECK-NEXT:   %phi_realloccast = bitcast i8* %phi_realloccache to double**
; CHECK-NEXT:   %3 = bitcast i64* %loopLimit_cache3.0 to i8*
; CHECK-NEXT:   %loopLimit_realloccache = call i8* @realloc(i8* %3, i64 %2)
; CHECK-NEXT:   %loopLimit_realloccast = bitcast i8* %loopLimit_realloccache to i64*
; CHECK-NEXT:   %iand = and i64 %iv, 5678
; CHECK-NEXT:   %exitcondi = icmp eq i64 %iand, %n
; CHECK-NEXT:   br i1 %exitcondi, label %invertfor.outerbody, label %for.body.ph

; CHECK: for.body.ph:                                      ; preds = %for.outerbody
; CHECK-NEXT:   %4 = getelementptr inbounds double*, double** %phi_realloccast, i64 %iv
; CHECK-NEXT:   store double* null, double** %4
; CHECK-NEXT:   br label %for.body

; CHECK: for.body:                                         ; preds = %for.body.for.body_crit_edge, %for.body.ph
; CHECK-NEXT:   %5 = phi double* [ %.pre, %for.body.for.body_crit_edge ], [ null, %for.body.ph ]
; CHECK-NEXT:   %iv1 = phi i64 [ %iv.next2, %for.body.for.body_crit_edge ], [ 0, %for.body.ph ]
; CHECK-NEXT:   %phi = phi double [ %phiadd, %for.body.for.body_crit_edge ], [ %x, %for.body.ph ]
; CHECK-NEXT:   %iv.next2 = add nuw nsw i64 %iv1, 1
; CHECK-NEXT:   %6 = bitcast double* %5 to i8*
; CHECK-NEXT:   %7 = mul nuw nsw i64 8, %iv.next2
; CHECK-NEXT:   %phi_realloccache4 = call i8* @realloc(i8* %6, i64 %7)
; CHECK-NEXT:   %phi_realloccast5 = bitcast i8* %phi_realloccache4 to double*
; CHECK-NEXT:   store double* %phi_realloccast5, double** %4, align 8
; CHECK-NEXT:   %8 = getelementptr inbounds double, double* %phi_realloccast5, i64 %iv1
; CHECK-NEXT:   store double %phi, double* %8, align 8
; CHECK-NEXT:   %phiadd = fadd fast double %phi, 1.000000e+00
; CHECK-NEXT:   %jand = and i64 %iv1, 1234
; CHECK-NEXT:   %exitcond = icmp eq i64 %jand, %n
; CHECK-NEXT:   br i1 %exitcond, label %for.outerbody.loopexit, label %for.body.for.body_crit_edge

; CHECK: for.body.for.body_crit_edge:                      ; preds = %for.body
; CHECK-NEXT:   %.pre = load double*, double** %4
; CHECK-NEXT:   br label %for.body

; CHECK: invertentry:                                      ; preds = %invertfor.outerbody
; CHECK-NEXT:   %9 = insertvalue { double } undef, double %"x'de.0", 0
; CHECK-NEXT:   tail call void @free(i8* nonnull %phi_realloccache)
; CHECK-NEXT:   ret { double } %9

; CHECK: invertfor.outerbody:                              ; preds = %for.outerbody, %invertfor.body.ph
; CHECK-NEXT:   %"outeradd'de.0" = phi double [ %16, %invertfor.body.ph ], [ %differeturn, %for.outerbody ]
; CHECK-NEXT:   %"x'de.0" = phi double [ %14, %invertfor.body.ph ], [ 0.000000e+00, %for.outerbody ]
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %11, %invertfor.body.ph ], [ %iv, %for.outerbody ]
; CHECK-NEXT:   %10 = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %10, label %invertentry, label %incinvertfor.outerbody

; CHECK: incinvertfor.outerbody:                           ; preds = %invertfor.outerbody
; CHECK-NEXT:   %11 = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   %12 = getelementptr inbounds i64, i64* %loopLimit_realloccast, i64 %11
; CHECK-NEXT:   %13 = load i64, i64* %12, align 8, !invariant.group !0
; CHECK-NEXT:   %.phi.trans.insert = getelementptr inbounds double*, double** %phi_realloccast, i64 %11
; CHECK-NEXT:   %[[pre6:.+]] = load double*, double** %.phi.trans.insert, align 8, !invariant.group !2
; CHECK-NEXT:   br label %invertfor.body

; CHECK: invertfor.body.ph:                                ; preds = %invertfor.body
; CHECK-NEXT:   %14 = fadd fast double %"x'de.0", %20
; CHECK-NEXT:   %15 = bitcast double* %[[pre6]] to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %15)
; CHECK-NEXT:   br label %invertfor.outerbody

; CHECK: invertfor.body:                                   ; preds = %incinvertfor.body, %incinvertfor.outerbody
; CHECK-NEXT:   %"add'de.1" = phi double [ 0.000000e+00, %incinvertfor.outerbody ], [ %16, %incinvertfor.body ]
; CHECK-NEXT:   %"phiadd'de.1" = phi double [ 0.000000e+00, %incinvertfor.outerbody ], [ %20, %incinvertfor.body ]
; CHECK-NEXT:   %"innersum'de.1" = phi double [ %"outeradd'de.0", %incinvertfor.outerbody ], [ 0.000000e+00, %incinvertfor.body ]
; CHECK-NEXT:   %"iv1'ac.0" = phi i64 [ %13, %incinvertfor.outerbody ], [ %22, %incinvertfor.body ]
; CHECK-NEXT:   %16 = fadd fast double %"innersum'de.1", %"add'de.1"
; CHECK-NEXT:   %17 = getelementptr inbounds double, double* %[[pre6]], i64 %"iv1'ac.0"
; CHECK-NEXT:   %18 = load double, double* %17, align 8, !invariant.group !1
; CHECK-NEXT:   %m0diffephi = fmul fast double %"add'de.1", %18
; CHECK-NEXT:   %19 = fadd fast double %"phiadd'de.1", %m0diffephi
; CHECK-NEXT:   %20 = fadd fast double %19, %m0diffephi
; CHECK-NEXT:   %21 = icmp eq i64 %"iv1'ac.0", 0
; CHECK-NEXT:   br i1 %21, label %invertfor.body.ph, label %incinvertfor.body

; CHECK: incinvertfor.body:                                ; preds = %invertfor.body
; CHECK-NEXT:   %22 = add nsw i64 %"iv1'ac.0", -1
; CHECK-NEXT:   br label %invertfor.body
; CHECK-NEXT: }
