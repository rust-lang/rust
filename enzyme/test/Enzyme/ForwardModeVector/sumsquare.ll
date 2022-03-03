; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -inline -mem2reg -instsimplify -adce -loop-deletion -correlated-propagation -simplifycfg -early-cse -S | FileCheck %s

%struct.Gradients = type { double, double, double }

; Function Attrs: nounwind
declare %struct.Gradients @__enzyme_fwddiff(double (double*, i64)*, ...)

; Function Attrs: norecurse nounwind readonly uwtable
define dso_local double @sumsquare(double* nocapture readonly %x, i64 %n) #0 {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret double %add

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %total.011 = phi double [ 0.000000e+00, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds double, double* %x, i64 %indvars.iv
  %0 = load double, double* %arrayidx
  %mul = fmul fast double %0, %0
  %add = fadd fast double %mul, %total.011
  %indvars.iv.next = add nuw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv, %n
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

; Function Attrs: nounwind uwtable
define dso_local %struct.Gradients @dsumsquare(double* %x, double* %xp1, double* %xp2, double* %xp3, i64 %n) local_unnamed_addr #1 {
entry:
  %0 = tail call %struct.Gradients (double (double*, i64)*, ...) @__enzyme_fwddiff(double (double*, i64)* nonnull @sumsquare, metadata !"enzyme_width", i64 3, double* %x, double* %xp1, double* %xp2, double* %xp3, i64 %n)
  ret %struct.Gradients %0
}


attributes #0 = { norecurse nounwind readonly uwtable }
attributes #1 = { nounwind uwtable }
attributes #2 = { nounwind }


; CHECK: define dso_local %struct.Gradients @dsumsquare(double* %x, double* %xp1, double* %xp2, double* %xp3, i64 %n)
; CHECK-NEXT: entry:
; CHECK-NEXT:   br label %for.body.i

; CHECK: for.body.i:                                       ; preds = %for.body.i, %entry
; CHECK-NEXT:   %iv.i = phi i64 [ %iv.next.i, %for.body.i ], [ 0, %entry ]
; CHECK-NEXT:   %"total.011'.i" = phi {{(fast )?}}[3 x double] [ zeroinitializer, %entry ], [ %18, %for.body.i ]
; CHECK-NEXT:   %iv.next.i = add nuw nsw i64 %iv.i, 1
; CHECK-NEXT:   %"arrayidx'ipg.i" = getelementptr inbounds double, double* %xp1, i64 %iv.i
; CHECK-NEXT:   %"arrayidx'ipg1.i" = getelementptr inbounds double, double* %xp2, i64 %iv.i
; CHECK-NEXT:   %"arrayidx'ipg2.i" = getelementptr inbounds double, double* %xp3, i64 %iv.i
; CHECK-NEXT:   %arrayidx.i = getelementptr inbounds double, double* %x, i64 %iv.i
; CHECK-NEXT:   %0 = load double, double* %arrayidx.i
; CHECK-NEXT:   %1 = load double, double* %"arrayidx'ipg.i"
; CHECK-NEXT:   %2 = load double, double* %"arrayidx'ipg1.i"
; CHECK-NEXT:   %3 = load double, double* %"arrayidx'ipg2.i"
; CHECK-NEXT:   %4 = fmul fast double %1, %0
; CHECK-NEXT:   %5 = fadd fast double %4, %4
; CHECK-NEXT:   %6 = fmul fast double %2, %0
; CHECK-NEXT:   %7 = fadd fast double %6, %6
; CHECK-NEXT:   %8 = fmul fast double %3, %0
; CHECK-NEXT:   %9 = fadd fast double %8, %8
; CHECK-NEXT:   %10 = extractvalue [3 x double] %"total.011'.i", 0
; CHECK-NEXT:   %11 = fadd fast double %5, %10
; CHECK-NEXT:   %12 = insertvalue [3 x double] undef, double %11, 0
; CHECK-NEXT:   %13 = extractvalue [3 x double] %"total.011'.i", 1
; CHECK-NEXT:   %14 = fadd fast double %7, %13
; CHECK-NEXT:   %15 = insertvalue [3 x double] %12, double %14, 1
; CHECK-NEXT:   %16 = extractvalue [3 x double] %"total.011'.i", 2
; CHECK-NEXT:   %17 = fadd fast double %9, %16
; CHECK-NEXT:   %18 = insertvalue [3 x double] %15, double %17, 2
; CHECK-NEXT:   %exitcond.i = icmp eq i64 %iv.i, %n
; CHECK-NEXT:   br i1 %exitcond.i, label %fwddiffe3sumsquare.exit, label %for.body.i

; CHECK: fwddiffe3sumsquare.exit:                          ; preds = %for.body.i
; CHECK-NEXT:   %19 = insertvalue %struct.Gradients zeroinitializer, double %11, 0
; CHECK-NEXT:   %20 = insertvalue %struct.Gradients %19, double %14, 1
; CHECK-NEXT:   %21 = insertvalue %struct.Gradients %20, double %17, 2
; CHECK-NEXT:   ret %struct.Gradients %21
; CHECK-NEXT: }