; RUN: if [ %llvmver -lt 15 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -inline -mem2reg -gvn -instsimplify -adce -loop-deletion -correlated-propagation -simplifycfg -licm -early-cse -simplifycfg -instsimplify -S | FileCheck %s -check-prefixes LL14,CHECK; fi
; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -inline -mem2reg -gvn -instsimplify -adce -loop-deletion -correlated-propagation -simplifycfg -licm -early-cse -simplifycfg -instsimplify -S | FileCheck %s -check-prefixes LL15,CHECK; fi

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

; CHECK: for.outerbody.loopexit:
; CHECK-NEXT:   %0 = getelementptr inbounds i64, i64* %[[loopLimit_realloccast:.+]], i64 %iv
; CHECK-NEXT:   store i64 %iv1, i64* %0, align 8, !invariant.group !0
; CHECK-NEXT:   br label %for.outerbody

; CHECK: for.outerbody: 
; CHECK-NEXT:   %[[loopLimit_cache3:.+]] = phi i64* [ null, %entry ], [ %[[loopLimit_realloccast]], %for.outerbody.loopexit ]
; CHECK-NEXT:   %phi_cache.0 = phi double** [ null, %entry ], [ %[[phi_realloccast:.+]], %for.outerbody.loopexit ]
; CHECK-NEXT:   %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.outerbody.loopexit ] 
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %1 = bitcast double** %phi_cache.0 to i8*
; CHECK-NEXT:   %2 = and i64 %iv.next, 1
; CHECK-NEXT:   %3 = icmp ne i64 %2, 0
; CHECK-NEXT:   %4 = call i64 @llvm.ctpop.i64(i64 %iv.next)
; CHECK-NEXT:   %5 = icmp ult i64 %4, 3
; CHECK-NEXT:   %6 = and i1 %5, %3
; CHECK-NEXT:   br i1 %6, label %grow.i, label %__enzyme_exponentialallocation.exit

; CHECK: grow.i: 
; CHECK-NEXT:   %7 = call i64 @llvm.ctlz.i64(i64 %iv.next, i1 true)
; CHECK-NEXT:   %8 = sub nuw nsw i64 64, %7
; CHECK-NEXT:   %9 = shl i64 8, %8
; CHECK-NEXT:   %10 = call i8* @realloc(i8* %1, i64 %9)
; CHECK-NEXT:   br label %__enzyme_exponentialallocation.exit

; CHECK: __enzyme_exponentialallocation.exit:
; CHECK-NEXT:   %[[phi_realloc:.+]] = phi i8* [ %10, %grow.i ], [ %1, %for.outerbody ]
; CHECK-NEXT:   %[[phi_realloccache:.+]] = bitcast i8* %[[phi_realloc]] to double**
; CHECK-NEXT:   %13 = bitcast i64* %[[loopLimit_cache3]] to i8*
; CHECK-NEXT:   br i1 %6, label %[[growi7:.+]], label %[[nouterbody:.+]]

; CHECK: [[growi7]]:  
; CHECK-NEXT:   %14 = call i64 @llvm.ctlz.i64(i64 %iv.next, i1 true)
; CHECK-NEXT:   %15 = sub nuw nsw i64 64, %14
; CHECK-NEXT:   %16 = shl i64 8, %15
; CHECK-NEXT:   %17 = call i8* @realloc(i8* %13, i64 %16)
; CHECK-NEXT:   br label %[[nouterbody]]

; CHECK: [[nouterbody]]:
; CHECK-NEXT:   %[[loopLimit_realloccache:.+]] = phi i8* [ %17, %[[growi7]] ], [ %13, %__enzyme_exponentialallocation.exit ]
; CHECK-NEXT:   %[[loopLimit_realloccast]] = bitcast i8* %[[loopLimit_realloccache]] to i64*

; CHECK-NEXT:   %iand = and i64 %iv, 5678
; CHECK-NEXT:   %exitcondi = icmp eq i64 %iand, %n
; CHECK-NEXT:   br i1 %exitcondi, label %invertfor.outerbody, label %for.body.ph

; CHECK: for.body.ph:
; CHECK-NEXT:   %[[a4:.+]] = getelementptr inbounds double*, double** %[[phi_realloccast]], i64 %iv
; CHECK-NEXT:   store double* null, double** %[[a4]]
; CHECK-NEXT:   br label %for.body

; CHECK: for.body: 
; LL14-NEXT:   %[[a5:.+]] = phi double* [ %.pre, %[[crit:.+]] ], [ null, %for.body.ph ]
; LL15-NEXT:   %iv1 = phi i64 [ %iv.next2, %[[crit:.+]] ], [ 0, %for.body.ph ]
; LL14-NEXT:   %iv1 = phi i64 [ %iv.next2, %[[crit]] ], [ 0, %for.body.ph ]
; CHECK-NEXT:   %phi = phi double [ %phiadd, %[[crit]] ], [ %x, %for.body.ph ]
; CHECK-NEXT:   %iv.next2 = add nuw nsw i64 %iv1, 1
; LL15-NEXT:   %[[a5:.+]] = load double*, double** %[[a4]]
; CHECK-NEXT:   %[[a6:.+]] = bitcast double* %[[a5]] to i8*

; CHECK-NEXT:   %23 = and i64 %iv.next2, 1
; CHECK-NEXT:   %24 = icmp ne i64 %23, 0
; CHECK-NEXT:   %25 = call i64 @llvm.ctpop.i64(i64 %iv.next2)
; CHECK-NEXT:   %26 = icmp ult i64 %25, 3
; CHECK-NEXT:   %27 = and i1 %26, %24
; LL14-NEXT:   br i1 %27, label %[[growi9:.+]], label %[[exit10:.+]]
; LL15-NEXT:   br i1 %27, label %[[growi9:.+]], label %[[crit:.+]]

; CHECK: [[growi9]]:
; CHECK-NEXT:   %28 = call i64 @llvm.ctlz.i64(i64 %iv.next2, i1 true)
; CHECK-NEXT:   %29 = sub nuw nsw i64 64, %28
; CHECK-NEXT:   %30 = shl i64 8, %29
; CHECK-NEXT:   %31 = call i8* @realloc(i8* %22, i64 %30)
; LL14-NEXT:   br label %[[exit10]]
; LL15-NEXT:   br label %[[crit]]

; LL14: [[exit10]]: 
; LL15: [[crit]]: 
; CHECK-NEXT:   %[[phi_realloccache4:.+]] = phi i8* [ %31, %[[growi9]] ], [ %22, %for.body ]
; CHECK-NEXT:   %[[phi_realloccast5:.+]] = bitcast i8* %[[phi_realloccache4]] to double*


; CHECK-NEXT:   store double* %[[phi_realloccast5]], double** %[[a4]], align 8
; CHECK-NEXT:   %[[a8:.+]] = getelementptr inbounds double, double* %[[phi_realloccast5]], i64 %iv1
; CHECK-NEXT:   store double %phi, double* %[[a8]], align 8
; CHECK-NEXT:   %phiadd = fadd fast double %phi, 1.000000e+00
; CHECK-NEXT:   %jand = and i64 %iv1, 1234
; CHECK-NEXT:   %exitcond = icmp eq i64 %jand, %n
; LL14-NEXT:   br i1 %exitcond, label %for.outerbody.loopexit, label %[[crit]]
; LL15-NEXT:   br i1 %exitcond, label %for.outerbody.loopexit, label %for.body

; LL14: [[crit]]:
; LL14-NEXT:   %.pre = load double*, double** %[[a4]]
; LL14-NEXT:   br label %for.body

; CHECK: invertentry:                                      ; preds = %invertfor.outerbody
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[loopLimit_realloccache]])
; CHECK-NEXT:   %[[a9:.+]] = insertvalue { double } undef, double %"x'de.0", 0
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[phi_realloc]])
; CHECK-NEXT:   ret { double } %[[a9]]

; CHECK: invertfor.outerbody: 
; CHECK-NEXT:   %"x'de.0" = phi double [ %[[a14:.+]], %invertfor.body.ph ], [ 0.000000e+00, %[[nouterbody]] ]
; CHECK-NEXT:   %"outeradd'de.0" = phi double [ %[[a16:.+]], %invertfor.body.ph ], [ %differeturn, %[[nouterbody]] ]
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %[[a11:.+]], %invertfor.body.ph ], [ %iv, %[[nouterbody]] ]
; CHECK-NEXT:   %[[a10:.+]] = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %[[a10]], label %invertentry, label %incinvertfor.outerbody

; CHECK: incinvertfor.outerbody:                           ; preds = %invertfor.outerbody
; CHECK-NEXT:   %[[a11]] = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   %[[a12:.+]] = getelementptr inbounds i64, i64* %[[loopLimit_realloccast]], i64 %[[a11]]
; CHECK-NEXT:   %[[a13:.+]] = load i64, i64* %[[a12]], align 8, !invariant.group !0
; CHECK-NEXT:   %.phi.trans.insert = getelementptr inbounds double*, double** %[[phi_realloccast]], i64 %[[a11]]
; CHECK-NEXT:   %[[pre6:.+]] = load double*, double** %.phi.trans.insert, align 8, !invariant.group !2
; CHECK-NEXT:   br label %invertfor.body

; CHECK: invertfor.body.ph:                                ; preds = %invertfor.body
; CHECK-NEXT:   %[[a14]] = fadd fast double %"x'de.0", %[[a20:.+]]
; CHECK-NEXT:   %[[a15:.+]] = bitcast double* %[[pre6]] to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[a15]])
; CHECK-NEXT:   br label %invertfor.outerbody

; CHECK: invertfor.body:                                   ; preds = %incinvertfor.body, %incinvertfor.outerbody
; CHECK-NEXT:   %"add'de.1" = phi double [ 0.000000e+00, %incinvertfor.outerbody ], [ %[[a16:.+]], %incinvertfor.body ]
; CHECK-NEXT:   %"phiadd'de.1" = phi double [ 0.000000e+00, %incinvertfor.outerbody ], [ %[[a20]], %incinvertfor.body ]
; CHECK-NEXT:   %"innersum'de.1" = phi double [ %"outeradd'de.0", %incinvertfor.outerbody ], [ 0.000000e+00, %incinvertfor.body ]
; CHECK-NEXT:   %"iv1'ac.0" = phi i64 [ %[[a13]], %incinvertfor.outerbody ], [ %[[a22:.+]], %incinvertfor.body ]
; CHECK-NEXT:   %[[a16]] = fadd fast double %"innersum'de.1", %"add'de.1"
; CHECK-NEXT:   %[[a17:.+]] = getelementptr inbounds double, double* %[[pre6]], i64 %"iv1'ac.0"
; CHECK-NEXT:   %[[a18:.+]] = load double, double* %[[a17]], align 8, !invariant.group !1
; CHECK-NEXT:   %m0diffephi = fmul fast double %"add'de.1", %[[a18]]
; CHECK-NEXT:   %[[a19:.+]] = fadd fast double %"phiadd'de.1", %m0diffephi
; CHECK-NEXT:   %[[a20]] = fadd fast double %[[a19]], %m0diffephi
; CHECK-NEXT:   %[[a21:.+]] = icmp eq i64 %"iv1'ac.0", 0
; CHECK-NEXT:   br i1 %[[a21]], label %invertfor.body.ph, label %incinvertfor.body

; CHECK: incinvertfor.body:                                ; preds = %invertfor.body
; CHECK-NEXT:   %[[a22]] = add nsw i64 %"iv1'ac.0", -1
; CHECK-NEXT:   br label %invertfor.body
; CHECK-NEXT: }
