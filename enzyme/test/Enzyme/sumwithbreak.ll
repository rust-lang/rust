; RUN: opt < %s %loadEnzyme -enzyme -enzyme_preopt=false -inline -mem2reg -correlated-propagation -instcombine -correlated-propagation -instsimplify -adce -loop-deletion -simplifycfg -S | FileCheck %s

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

; CHECK: define internal {} @diffef(double* nocapture readonly %x, double* %"x'", i64 %n, double %differeturn) #0 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   br label %for.body

; CHECK: for.body:                                         ; preds = %if.end, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %if.end ], [ 0, %entry ]
; CHECK-NEXT:   %data.016 = phi double [ 0.000000e+00, %entry ], [ %add5, %if.end ]
; CHECK-NEXT:   %iv.next = add nuw i64 %iv, 1
; CHECK-NEXT:   %cmp2 = fcmp fast ogt double %data.016, 1.000000e+01
; CHECK-NEXT:   br i1 %cmp2, label %invertcleanup, label %if.end

; CHECK: if.end:                                           ; preds = %for.body
; CHECK-NEXT:   %arrayidx4 = getelementptr inbounds double, double* %x, i64 %iv
; CHECK-NEXT:   %0 = load double, double* %arrayidx4, align 8
; CHECK-NEXT:   %add5 = fadd fast double %0, %data.016
; CHECK-NEXT:   %cmp = icmp ult i64 %iv, %n
; CHECK-NEXT:   br i1 %cmp, label %for.body, label %invertcleanup

; CHECK: invertentry:                                      ; preds = %invertfor.body
; CHECK-NEXT:   ret {} undef

; CHECK: invertfor.body:                                   ; preds = %loopMerge, %invertif.end
; CHECK-NEXT:   %"'de1.0" = phi double [ 0.000000e+00, %invertif.end ], [ %"'de1.1", %loopMerge ]
; CHECK-NEXT:   %"add5'de.0" = phi double [ 0.000000e+00, %invertif.end ], [ %"add5'de.2", %loopMerge ]
; CHECK-NEXT:   %"data.016'de.0" = phi double [ %6, %invertif.end ], [ %"data.016'de.2", %loopMerge ]
; CHECK-NEXT:   %1 = icmp eq i64 %"iv'phi", 0
; CHECK-NEXT:   %2 = fadd fast double %"add5'de.0", %"data.016'de.0"
; CHECK-NEXT:   br i1 %1, label %invertentry, label %loopMerge

; CHECK: invertif.then:                                    ; preds = %invertcleanup
; CHECK-NEXT:   %"arrayidx'ipg" = getelementptr double, double* %"x'", i64 %n
; CHECK-NEXT:   %3 = load double, double* %"arrayidx'ipg"
; CHECK-NEXT:   %4 = fadd fast double %3, %9
; CHECK-NEXT:   store double %4, double* %"arrayidx'ipg"
; CHECK-NEXT:   br label %loopMerge.preheader

; CHECK: loopMerge.preheader:                              ; preds = %invertcleanup, %invertif.then
; CHECK-NEXT:   %"add5'de.1" = phi double [ 0.000000e+00, %invertif.then ], [ %10, %invertcleanup ]
; CHECK-NEXT:   %"data.016'de.1" = phi double [ %9, %invertif.then ], [ 0.000000e+00, %invertcleanup ]
; CHECK-NEXT:   br label %loopMerge

; CHECK: invertif.end:                                     ; preds = %loopMerge
; CHECK-NEXT:   %5 = fadd fast double %"'de1.1", %"add5'de.2"
; CHECK-NEXT:   %6 = fadd fast double %"data.016'de.2", %"add5'de.2"
; CHECK-NEXT:   %"arrayidx4'ipg" = getelementptr double, double* %"x'", i64 %"iv'phi"
; CHECK-NEXT:   %7 = load double, double* %"arrayidx4'ipg"
; CHECK-NEXT:   %8 = fadd fast double %7, %5
; CHECK-NEXT:   store double %8, double* %"arrayidx4'ipg"
; CHECK-NEXT:   br label %invertfor.body

; CHECK: invertcleanup:                                    ; preds = %for.body, %if.end
; CHECK-NEXT:   %iv3 = phi i64 [ %iv, %for.body ], [ %iv, %if.end ]
; CHECK-NEXT:   %loopender_cache.0 = phi i8 [ 0, %for.body ], [ 1, %if.end ]
; CHECK-NEXT:   %"cmp2!manual_lcssa" = phi i1 [ true, %for.body ], [ false, %if.end ]
; CHECK-NEXT:   %9 = select i1 %"cmp2!manual_lcssa", double %differeturn, double 0.000000e+00
; CHECK-NEXT:   %10 = select i1 %"cmp2!manual_lcssa", double 0.000000e+00, double %differeturn
; CHECK-NEXT:   br i1 %"cmp2!manual_lcssa", label %invertif.then, label %loopMerge.preheader

; CHECK: loopMerge:                                        ; preds = %loopMerge.preheader, %invertfor.body
; CHECK-NEXT:   %"'de1.1" = phi double [ 0.000000e+00, %loopMerge.preheader ], [ %"'de1.0", %invertfor.body ]
; CHECK-NEXT:   %"add5'de.2" = phi double [ %"add5'de.1", %loopMerge.preheader ], [ %2, %invertfor.body ]
; CHECK-NEXT:   %"data.016'de.2" = phi double [ %"data.016'de.1", %loopMerge.preheader ], [ 0.000000e+00, %invertfor.body ]
; CHECK-NEXT:   %"iv'phi" = phi i64 [ %11, %invertfor.body ], [ %iv3, %loopMerge.preheader ]
; CHECK-NEXT:   %11 = sub i64 %"iv'phi", 1
; CHECK-NEXT:   %cond = icmp eq i8 %loopender_cache.0, 0
; CHECK-NEXT:   br i1 %cond, label %invertfor.body, label %invertif.end
; CHECK-NEXT: }
