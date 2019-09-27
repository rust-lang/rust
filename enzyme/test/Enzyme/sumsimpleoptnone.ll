; RUN: opt < %s %loadEnzyme -enzyme -enzyme_preopt=false -inline -mem2reg -instsimplify -adce -loop-deletion -correlated-propagation -simplifycfg -S -early-cse | FileCheck %s

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


attributes #0 = { noinline nounwind uwtable optnone }

; CHECK: define internal {{(dso_local )?}}{} @diffef(double* %x, double* %"x'", double** %y, double** %"y'", i64 %n)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = add nuw i64 %n, 1
; CHECK-NEXT:   %mallocsize = mul i64 %0, 8
; CHECK-NEXT:   %malloccall = tail call noalias nonnull i8* @malloc(i64 %mallocsize)
; CHECK-NEXT:   %"'ipl_malloccache" = bitcast i8* %malloccall to double**
; CHECK-NEXT:   br label %for.body

; CHECK: for.body:                                         ; preds = %for.body, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
; CHECK-NEXT:   %iv.next = add nuw i64 %iv, 1
; CHECK-NEXT:   %1 = load double, double* %x
; CHECK-NEXT:   %"'ipl" = load double*, double** %"y'"
; CHECK-NEXT:   %2 = getelementptr double*, double** %"'ipl_malloccache", i64 %iv
; CHECK-NEXT:   store double* %"'ipl", double** %2
; CHECK-NEXT:   %3 = load double*, double** %y
; CHECK-NEXT:   %4 = load double, double* %3
; CHECK-NEXT:   %add = fadd fast double %4, %1
; CHECK-NEXT:   store double %add, double* %3
; CHECK-NEXT:   %cmp = icmp ule i64 %iv.next, %n
; CHECK-NEXT:   br i1 %cmp, label %for.body, label %invertfor.body

; CHECK: invertentry:                                      ; preds = %invertfor.body
; CHECK-NEXT:   tail call void @free(i8* nonnull %malloccall)
; CHECK-NEXT:   ret {} undef

; CHECK: invertfor.body:
; CHECK-NEXT:   %"iv'phi" = phi i64 [ %5, %invertfor.body ], [ %n, %for.body ]
; CHECK-NEXT:   %5 = sub i64 %"iv'phi", 1
; CHECK-NEXT:   %6 = getelementptr double*, double** %"'ipl_malloccache", i64 %"iv'phi"
; CHECK-NEXT:   %7 = load double*, double** %6, !invariant.load !0
; CHECK-NEXT:   %8 = load double, double* %7
; CHECK-NEXT:   store double %8, double* %7
; CHECK-NEXT:   %9 = load double, double* %"x'"
; CHECK-NEXT:   %10 = fadd fast double %9, %8
; CHECK-NEXT:   store double %10, double* %"x'"
; CHECK-NEXT:   %11 = icmp eq i64 %"iv'phi", 0
; CHECK-NEXT:   br i1 %11, label %invertentry, label %invertfor.body
; CHECK-NEXT: }

; CHECK: !0 = !{}
