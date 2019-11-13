; RUN: opt < %s %loadEnzyme -enzyme -enzyme_preopt=false -inline -mem2reg -instsimplify -adce -loop-deletion -correlated-propagation -simplifycfg -S -early-cse -simplifycfg | FileCheck %s

; Function Attrs: noinline nounwind uwtable
define dso_local void @f(double* %x, double** %y, i64 %n) #0 {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %i.0 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %cmp = icmp ule i64 %i.0, %n
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds double, double* %x, i64 0
  %0 = load double, double* %arrayidx
  %1 = load double*, double** %y
  %2 = load double, double* %1
  %add = fadd fast double %2, %0
  store double %add, double* %1
  %inc = add i64 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

; Function Attrs: noinline nounwind uwtable
define dso_local double @dsumsquare(double* %x, double* %xp, double** %y, double** %yp, i64 %n) #0 {
entry:
  %call = call fast double @__enzyme_autodiff(i8* bitcast (void (double*, double**, i64)* @f to i8*), double* %x, double* %xp, double** %y, double** %yp, i64 %n)
  ret double %call
}

declare dso_local double @__enzyme_autodiff(i8*, double*, double*, double**, double**, i64)


attributes #0 = { noinline nounwind uwtable }

; CHECK: define internal {{(dso_local )?}}{} @diffef(double* %x, double* %"x'", double** %y, double** %"y'", i64 %n)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = add i64 %n, 1
; CHECK-NEXT:   %1 = add nuw i64 %0, 1
; CHECK-NEXT:   %mallocsize = mul i64 %1, 8
; CHECK-NEXT:   %malloccall = tail call noalias nonnull i8* @malloc(i64 %mallocsize)
; CHECK-NEXT:   %"'ipl_malloccache" = bitcast i8* %malloccall to double**
; CHECK-NEXT:   br label %for.cond

; CHECK: for.cond:
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
; CHECK-NEXT:   %iv.next = add nuw i64 %iv, 1
; CHECK-NEXT:   %cmp = icmp ne i64 %iv, %0
; CHECK-NEXT:   br i1 %cmp, label %for.body, label %invertfor.cond

; CHECK: for.body: 
; CHECK-NEXT:   %2 = load double, double* %x
; CHECK-NEXT:   %"'ipl" = load double*, double** %"y'"
; CHECK-NEXT:   %3 = getelementptr double*, double** %"'ipl_malloccache", i64 %iv
; CHECK-NEXT:   store double* %"'ipl", double** %3
; CHECK-NEXT:   %4 = load double*, double** %y
; CHECK-NEXT:   %5 = load double, double* %4
; CHECK-NEXT:   %add = fadd fast double %5, %2
; CHECK-NEXT:   store double %add, double* %4
; CHECK-NEXT:   br label %for.cond

; CHECK: invertentry:
; CHECK-NEXT:   tail call void @free(i8* nonnull %malloccall)
; CHECK-NEXT:   ret {} undef

; CHECK: invertfor.cond:
; CHECK-NEXT:   %[[ivp:.+]] = phi i64 [ %[[sub:.+]], %incinvertfor.cond ], [ %0, %for.cond ] 
; CHECK-NEXT:   %[[cmp:.+]] = icmp eq i64 %[[ivp]], 0
; CHECK-NEXT:   br i1 %[[cmp]], label %invertentry, label %incinvertfor.cond

; CHECK: incinvertfor.cond:
; CHECK-NEXT:   %[[sub]] = sub nuw nsw i64 %[[ivp]], 1
; CHECK-NEXT:   %8 = getelementptr double*, double** %"'ipl_malloccache", i64 %[[sub]]
; CHECK-NEXT:   %9 = load double*, double** %8, !invariant.load !0
; CHECK-NEXT:   %10 = load double, double* %9
; CHECK-NEXT:   store double %10, double* %9
; CHECK-NEXT:   %11 = load double, double* %"x'"
; CHECK-NEXT:   %12 = fadd fast double %11, %10
; CHECK-NEXT:   store double %12, double* %"x'"
; CHECK-NEXT:   br label %invertfor.cond
; CHECK-NEXT: }


; CHECK: !0 = !{}
