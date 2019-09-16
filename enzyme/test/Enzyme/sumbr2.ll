; RUN: opt < %s %loadEnzyme -enzyme -enzyme_preopt=false -inline -mem2reg -instsimplify -adce -loop-deletion -correlated-propagation -instcombine -S | FileCheck %s

; Function Attrs: norecurse nounwind readonly uwtable
define dso_local double @sum(double* nocapture readonly %x, i64 %n) #0 {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret double %add

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %extra ]
  %total.07 = phi double [ 0.000000e+00, %entry ], [ %add2, %extra ]
  %arrayidx = getelementptr inbounds double, double* %x, i64 %indvars.iv
  %0 = load double, double* %arrayidx, align 8
  %add = fadd fast double %0, %total.07
  %indvars.iv.next = add nuw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv, %n
  br i1 %exitcond, label %for.cond.cleanup, label %extra

extra:
  %res = uitofp i64 %indvars.iv to double
  %add2 = fmul fast double %add, %res
  br label %for.body
}

; Function Attrs: nounwind uwtable
define dso_local void @dsum(double* %x, double* %xp, i64 %n) local_unnamed_addr #1 {
entry:
  %0 = tail call double (double (double*, i64)*, ...) @__enzyme_autodiff(double (double*, i64)* nonnull @sum, double* %x, double* %xp, i64 %n)
  ret void
}

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double*, i64)*, ...) #2

attributes #0 = { norecurse nounwind readonly uwtable }
attributes #1 = { nounwind uwtable } 
attributes #2 = { nounwind }


; CHECK: define dso_local void @dsum(double* %x, double* %xp, i64 %n) 
; CHECK-NEXT: entry:
; CHECK-NEXT:   br label %invertfor.body.i
; CHECK: invertfor.body.i: 
; CHECK-NEXT:   %"add'de.0.i" = phi double [ 1.000000e+00, %entry ], [ %m0diffeadd.i, %invertextra.i ]
; CHECK-NEXT:   %"indvars.iv'phi.i" = phi i64 [ %n, %entry ], [ %3, %invertextra.i ]
; CHECK-NEXT:   %"arrayidx'ipg.i" = getelementptr double, double* %xp, i64 %"indvars.iv'phi.i"
; CHECK-NEXT:   %0 = load double, double* %"arrayidx'ipg.i", align 8
; CHECK-NEXT:   %1 = fadd fast double %0, %"add'de.0.i"
; CHECK-NEXT:   store double %1, double* %"arrayidx'ipg.i", align 8
; CHECK-NEXT:   %2 = icmp eq i64 %"indvars.iv'phi.i", 0
; CHECK-NEXT:   br i1 %2, label %diffesum.exit, label %invertextra.i
; CHECK: invertextra.i:  
; CHECK-NEXT:   %3 = add i64 %"indvars.iv'phi.i", -1
; CHECK-NEXT:   %res_unwrap.i = uitofp i64 %"indvars.iv'phi.i" to double
; CHECK-NEXT:   %m0diffeadd.i = fmul fast double %"add'de.0.i", %res_unwrap.i
; CHECK-NEXT:   br label %invertfor.body.i
; CHECK: diffesum.exit:                                    ; preds = %invertfor.body.i
; CHECK-NEXT:   ret void
; CHECK-NEXT: }


