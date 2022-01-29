; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instcombine -correlated-propagation -adce -instcombine -simplifycfg -early-cse -simplifycfg -loop-unroll -instcombine -simplifycfg -gvn -jump-threading -instcombine -simplifycfg -S | FileCheck %s

; Function Attrs: noinline nounwind uwtable
define dso_local double @f(double* nocapture readonly %x, i64 %n) #0 {
entry:
  br label %for.body

for.body:                                         ; preds = %if.end, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %if.end ]
  %data.016 = phi double [ 0.000000e+00, %entry ], [ %add5, %if.end ]
  %cmp2 = fcmp fast ogt double %data.016, 1.000000e+01
  br i1 %cmp2, label %if.then, label %if.end

if.then:                                          ; preds = %for.body
  %arrayidx = getelementptr inbounds double, double* %x, i64 %n
  %0 = load double, double* %arrayidx, align 8
  %add = fadd fast double %0, %data.016
  br label %cleanup

if.end:                                           ; preds = %for.body
  %arrayidx4 = getelementptr inbounds double, double* %x, i64 %indvars.iv
  %1 = load double, double* %arrayidx4, align 8
  %add5 = fadd fast double %1, %data.016
  %indvars.iv.next = add nuw i64 %indvars.iv, 1
  %cmp = icmp ult i64 %indvars.iv, %n
  br i1 %cmp, label %for.body, label %cleanup

cleanup:                                          ; preds = %if.end, %if.then
  %data.1 = phi double [ %add, %if.then ], [ %add5, %if.end ]
  ret double %data.1
}

; Function Attrs: noinline nounwind uwtable
define dso_local double @dsumsquare(double* %x, double* %xp, i64 %n) #0 {
entry:
  %call = call fast double @__enzyme_autodiff(i8* bitcast (double (double*, i64)* @f to i8*), double* %x, double* %xp, i64 %n)
  ret double %call
}

declare dso_local double @__enzyme_autodiff(i8*, double*, double*, i64)


attributes #0 = { noinline nounwind uwtable }

; CHECK: define internal {{(dso_local )?}}void @diffef(double* nocapture readonly %x, double* nocapture %"x'", i64 %n, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   br label %for.body

; CHECK: for.body:                                         ; preds = %if.end, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %if.end ], [ 0, %entry ]
; CHECK-NEXT:   %data.016 = phi double [ %add5, %if.end ], [ 0.000000e+00, %entry ]
; CHECK-NEXT:   %cmp2 = fcmp fast ogt double %data.016, 1.000000e+01
; CHECK-NEXT:   br i1 %cmp2, label %invertif.then, label %if.end

; CHECK: if.end:                                           ; preds = %for.body
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %arrayidx4 = getelementptr inbounds double, double* %x, i64 %iv
; CHECK-NEXT:   %0 = load double, double* %arrayidx4, align 8
; CHECK-NEXT:   %add5 = fadd fast double %0, %data.016
; CHECK-NEXT:   %cmp = icmp ult i64 %iv, %n
; CHECK-NEXT:   br i1 %cmp, label %for.body, label %invertif.end

; CHECK: invertentry:
; CHECK-NEXT:   ret void

; CHECK: invertfor.body:
; CHECK-NEXT:   %[[ivp0:.+]] = phi i64 [ %iv, %invertif.then ], [ %[[ivp1:.+]], %invertif.end ]
; CHECK-NEXT:   %[[icmp:.+]] = icmp eq i64 %[[ivp0]], 0
; CHECK-NEXT:   br i1 %[[icmp]], label %invertentry, label %incinvertfor.body

; CHECK: incinvertfor.body:
; CHECK-NEXT:   %[[ivp0m1:.+]] = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertif.end

; CHECK: invertif.then:
; CHECK-NEXT:   %[[arrayidxipg:.+]] = getelementptr inbounds double, double* %"x'", i64 %n
; CHECK-NEXT:   %[[loadit:.+]] = load double, double* %[[arrayidxipg]], align 8
; CHECK-NEXT:   %[[tostoreit:.+]] = fadd fast double %[[loadit]], %differeturn
; CHECK-NEXT:   store double %[[tostoreit]], double* %[[arrayidxipg]], align 8
; CHECK-NEXT:   br label %invertfor.body

; CHECK: invertif.end:
; CHECK-NEXT:   %[[ivp1]] = phi i64 [ %[[ivp0m1]], %incinvertfor.body ], [ %iv, %if.end ]
; CHECK-NEXT:   %[[arrayidx4ipg:.+]] = getelementptr inbounds double, double* %"x'", i64 %[[ivp1]]
; CHECK-NEXT:   %[[loaditp:.+]] = load double, double* %[[arrayidx4ipg]], align 8
; CHECK-NEXT:   %[[tostoreitp:.+]] = fadd fast double %[[loaditp]], %differeturn
; CHECK-NEXT:   store double %[[tostoreitp]], double* %[[arrayidx4ipg]], align 8
; CHECK-NEXT:   br label %invertfor.body
; CHECK-NEXT: }
