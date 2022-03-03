; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instcombine -correlated-propagation -adce -instcombine -simplifycfg -early-cse -simplifycfg -loop-unroll -instcombine -simplifycfg -gvn -jump-threading -instcombine -simplifycfg -S | FileCheck %s

%struct.Gradients = type { double, double, double }

; Function Attrs: nounwind
declare %struct.Gradients @__enzyme_fwddiff(i8*, ...)

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
define dso_local %struct.Gradients @dsumsquare(double* %x, double* %xp1, double* %xp2, double* %xp3, i64 %n) #0 {
entry:
  %call = call %struct.Gradients (i8*, ...) @__enzyme_fwddiff(i8* bitcast (double (double*, i64)* @f to i8*), metadata !"enzyme_width", i64 3, double* %x, double* %xp1, double* %xp2, double* %xp3, i64 %n)
  ret %struct.Gradients %call
}


attributes #0 = { noinline nounwind uwtable }


; CHECK: define internal [3 x double] @fwddiffe3f(double* nocapture readonly %x, [3 x double*] %"x'", i64 %n)
; CHECK-NEXT: entry:
; CHECK-NEXT:   br label %for.body

; CHECK: for.body:                                         ; preds = %if.end, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %if.end ], [ 0, %entry ]
; CHECK-NEXT:   %data.016 = phi double [ %add5, %if.end ], [ 0.000000e+00, %entry ]
; CHECK-NEXT:   %"data.016'" = phi {{(fast )?}}[3 x double] [ %30, %if.end ], [ zeroinitializer, %entry ]
; CHECK-NEXT:   %cmp2 = fcmp fast ogt double %data.016, 1.000000e+01
; CHECK-NEXT:   br i1 %cmp2, label %if.then, label %if.end

; CHECK: if.then:                                          ; preds = %for.body
; CHECK-NEXT:   %0 = extractvalue [3 x double*] %"x'", 0
; CHECK-NEXT:   %"arrayidx'ipg" = getelementptr inbounds double, double* %0, i64 %n
; CHECK-NEXT:   %1 = extractvalue [3 x double*] %"x'", 1
; CHECK-NEXT:   %"arrayidx'ipg1" = getelementptr inbounds double, double* %1, i64 %n
; CHECK-NEXT:   %2 = extractvalue [3 x double*] %"x'", 2
; CHECK-NEXT:   %"arrayidx'ipg2" = getelementptr inbounds double, double* %2, i64 %n
; CHECK-NEXT:   %3 = load double, double* %"arrayidx'ipg", align 8
; CHECK-NEXT:   %4 = load double, double* %"arrayidx'ipg1", align 8
; CHECK-NEXT:   %5 = load double, double* %"arrayidx'ipg2", align 8
; CHECK-NEXT:   %6 = extractvalue [3 x double] %"data.016'", 0
; CHECK-NEXT:   %7 = fadd fast double %3, %6
; CHECK-NEXT:   %8 = insertvalue [3 x double] undef, double %7, 0
; CHECK-NEXT:   %9 = extractvalue [3 x double] %"data.016'", 1
; CHECK-NEXT:   %10 = fadd fast double %4, %9
; CHECK-NEXT:   %11 = insertvalue [3 x double] %8, double %10, 1
; CHECK-NEXT:   %12 = extractvalue [3 x double] %"data.016'", 2
; CHECK-NEXT:   %13 = fadd fast double %5, %12
; CHECK-NEXT:   %14 = insertvalue [3 x double] %11, double %13, 2
; CHECK-NEXT:   br label %cleanup

; CHECK: if.end:                                           ; preds = %for.body
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %15 = extractvalue [3 x double*] %"x'", 0
; CHECK-NEXT:   %"arrayidx4'ipg" = getelementptr inbounds double, double* %15, i64 %iv
; CHECK-NEXT:   %16 = extractvalue [3 x double*] %"x'", 1
; CHECK-NEXT:   %"arrayidx4'ipg3" = getelementptr inbounds double, double* %16, i64 %iv
; CHECK-NEXT:   %17 = extractvalue [3 x double*] %"x'", 2
; CHECK-NEXT:   %"arrayidx4'ipg4" = getelementptr inbounds double, double* %17, i64 %iv
; CHECK-NEXT:   %arrayidx4 = getelementptr inbounds double, double* %x, i64 %iv
; CHECK-NEXT:   %18 = load double, double* %arrayidx4, align 8
; CHECK-NEXT:   %19 = load double, double* %"arrayidx4'ipg", align 8
; CHECK-NEXT:   %20 = load double, double* %"arrayidx4'ipg3", align 8
; CHECK-NEXT:   %21 = load double, double* %"arrayidx4'ipg4", align 8
; CHECK-NEXT:   %add5 = fadd fast double %18, %data.016
; CHECK-NEXT:   %22 = extractvalue [3 x double] %"data.016'", 0
; CHECK-NEXT:   %23 = fadd fast double %19, %22
; CHECK-NEXT:   %24 = insertvalue [3 x double] undef, double %23, 0
; CHECK-NEXT:   %25 = extractvalue [3 x double] %"data.016'", 1
; CHECK-NEXT:   %26 = fadd fast double %20, %25
; CHECK-NEXT:   %27 = insertvalue [3 x double] %24, double %26, 1
; CHECK-NEXT:   %28 = extractvalue [3 x double] %"data.016'", 2
; CHECK-NEXT:   %29 = fadd fast double %21, %28
; CHECK-NEXT:   %30 = insertvalue [3 x double] %27, double %29, 2
; CHECK-NEXT:   %cmp = icmp ult i64 %iv, %n
; CHECK-NEXT:   br i1 %cmp, label %for.body, label %cleanup

; CHECK: cleanup:                                          ; preds = %if.end, %if.then
; CHECK-NEXT:   %"data.1'" = phi {{(fast )?}}[3 x double] [ %14, %if.then ], [ %30, %if.end ]
; CHECK-NEXT:   ret [3 x double] %"data.1'"
; CHECK-NEXT: }