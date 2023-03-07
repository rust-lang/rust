; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -instsimplify -adce -S | FileCheck %s
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(instsimplify,adce)" -enzyme-preopt=false -S | FileCheck %s

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
  %call = call fast double @__enzyme_fwddiff(i8* bitcast (void (double*, double**, i64)* @f to i8*), double* %x, double* %xp, double** %y, double** %yp, i64 %n)
  ret double %call
}

declare dso_local double @__enzyme_fwddiff(i8*, double*, double*, double**, double**, i64)


attributes #0 = { noinline nounwind uwtable optnone }


; CHECK: define internal void @fwddiffef(double* %x, double* %"x'", double** %y, double** %"y'", i64 %n)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = add {{(nuw )?}}i64 %n, 1
; CHECK-NEXT:   br label %for.cond

; CHECK: for.cond:                                         ; preds = %for.body, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %cmp = icmp ne i64 %iv, %0
; CHECK-NEXT:   br i1 %cmp, label %for.body, label %for.end

; CHECK: for.body:                                         ; preds = %for.cond
; CHECK-NEXT:   %[[i2:.+]] = load double, double* %"x'"
; CHECK-NEXT:   %[[i1:.+]] = load double, double* %x
; CHECK-NEXT:   %[[ipl:.+]] = load double*, double** %"y'"
; CHECK-NEXT:   %[[i3:.+]] = load double*, double** %y
; CHECK-NEXT:   %[[i5:.+]] = load double, double* %[[ipl]]
; CHECK-NEXT:   %[[i4:.+]] = load double, double* %[[i3]]
; CHECK-NEXT:   %[[add:.+]] = fadd fast double %[[i4]], %[[i1]]
; CHECK-NEXT:   %[[i6:.+]] = fadd fast double %[[i5]], %[[i2]]
; CHECK-NEXT:   store double %[[add]], double* %[[i3]]
; CHECK-NEXT:   store double %[[i6]], double* %[[ipl]]
; CHECK-NEXT:   br label %for.cond

; CHECK: for.end:                                          ; preds = %for.cond
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
