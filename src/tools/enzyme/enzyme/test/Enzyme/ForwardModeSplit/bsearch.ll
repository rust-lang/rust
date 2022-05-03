; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -inline -mem2reg -gvn -instsimplify -correlated-propagation -adce -simplifycfg -S | FileCheck %s

; Function Attrs: noinline norecurse nounwind uwtable
define  double @f(double* nocapture %x, i64 %n) #0 {
entry:
  br label %loop

loop:
  %j = phi i64 [ %nj, %end ], [ 0, %entry ]
  %sum = phi double [ %nsum, %end ], [ 0.000000e+00, %entry ]
  %nj = add nsw nuw i64 %j, 1
  %g0 = getelementptr inbounds double, double* %x, i64 %j
  br label %body

body:                              ; preds = %entry, %for.cond.cleanup6
  %i = phi i64 [ %next, %body ], [ 0, %loop ]
  %gep = getelementptr inbounds double, double* %g0, i64 %i
  %ld = load double, double* %gep, align 8
  %cmp = fcmp oeq double %ld, 3.141592e+00
  %next = add nuw i64 %i, 1
  br i1 %cmp, label %body, label %end

end:
  %gep2 = getelementptr inbounds double, double* %x, i64 %i
  %ld2 = load double, double* %gep2, align 8
  %nsum = fadd double %ld2, %sum
  %cmp2 = icmp ne i64 %nj, 10
  br i1 %cmp2, label %loop, label %exit

exit:
  ret double %nsum
}

; Function Attrs: noinline nounwind uwtable
define dso_local double @dsumsquare(double* %x, double* %xp, i64 %n) local_unnamed_addr #1 {
entry:
  %call = tail call double (...) @__enzyme_fwdsplit(i8* bitcast (double (double*, i64)* @f to i8*), metadata !"enzyme_nofree", double* %x, double* %xp, i64 %n, i8* null)
  ret double %call
}

declare dso_local double @__enzyme_fwdsplit(...)

attributes #0 = { noinline norecurse nounwind uwtable }
attributes #1 = { noinline nounwind uwtable }


; CHECK: define internal double @fwddiffef(double* nocapture %x, double* nocapture %"x'", i64 %n, i8* %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = bitcast i8* %tapeArg to i1***
; CHECK-NEXT:   %truetape = load i1**, i1*** %0
; CHECK-NEXT:   br label %loop

; CHECK: loop:                                             ; preds = %end, %entry
; CHECK-DAG:   %iv = phi i64 [ %iv.next, %end ], [ 0, %entry ]
; CHECK-DAG:   %[[dsum:.+]] = phi {{(fast )?}}double [ %[[i4:.+]], %end ], [ 0.000000e+00, %entry ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %[[i1:.+]] = getelementptr inbounds i1*, i1** %truetape, i64 %iv
; CHECK-NEXT:   %.pre = load i1*, i1** %[[i1]], align 8, !invariant.group !1
; CHECK-NEXT:   br label %body

; CHECK: body:                                             ; preds = %body, %loop
; CHECK-NEXT:   %iv1 = phi i64 [ %iv.next2, %body ], [ 0, %loop ]
; CHECK-NEXT:   %iv.next2 = add nuw nsw i64 %iv1, 1
; CHECK-NEXT:   %[[i2:.+]] = getelementptr inbounds i1, i1* %.pre, i64 %iv1
; CHECK-NEXT:   %cmp = load i1, i1* %[[i2]], align 1, !invariant.group !2
; CHECK-NEXT:   br i1 %cmp, label %body, label %end

; CHECK: end:                                              ; preds = %body
; CHECK-NEXT:   %"gep2'ipg" = getelementptr inbounds double, double* %"x'", i64 %iv1
; CHECK-NEXT:   %[[i3:.+]] = load double, double* %"gep2'ipg"
; CHECK-NEXT:   %[[i4]] = fadd fast double %[[i3]], %[[dsum]]
; CHECK-NEXT:   %cmp2 = icmp ne i64 %iv.next, 10
; CHECK-NEXT:   br i1 %cmp2, label %loop, label %exit

; CHECK: exit:                                             ; preds = %end
; CHECK-NEXT:   ret double %[[i4]]
; CHECK-NEXT: }
