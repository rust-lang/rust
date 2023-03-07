; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -inline -mem2reg -gvn -instsimplify -adce -correlated-propagation -simplifycfg -S | FileCheck %s

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
  %idx = phi i64 [ %nidx, %body ], [ 0, %loop ]
  %gep = getelementptr inbounds double, double* %g0, i64 %i
  %ld = load double, double* %gep, align 8
  %cmp = fcmp oeq double %ld, 3.141592e+00
  %next = add nuw i64 %i, 1
  %int = fptoui double %ld to i64
  %nidx = add nuw i64 %idx, %int
  br i1 %cmp, label %body, label %end

end:
  %gep2 = getelementptr inbounds double, double* %x, i64 %idx
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
; CHECK-NEXT:   %0 = bitcast i8* %tapeArg to { i64**, double** }*
; CHECK-NEXT:   %truetape = load { i64**, double** }, { i64**, double** }* %0
; CHECK-DAG:    %[[i1:.+]] = extractvalue { i64**, double** } %truetape, 0
; CHECK-DAG:    %[[i2:.+]] = extractvalue { i64**, double** } %truetape, 1
; CHECK-NEXT:   br label %loop

; CHECK: loop:                                             ; preds = %end, %entry
; CHECK-DAG:   %iv = phi i64 [ %iv.next, %end ], [ 0, %entry ]
; CHECK-DAG:   %[[dsum:.+]] = phi {{(fast )?}}double [ %[[i8:.+]], %end ], [ 0.000000e+00, %entry ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-DAG:    %[[i3:.+]] = getelementptr inbounds i64*, i64** %[[i1]], i64 %iv
; CHECK-DAG:    %[[i4:.+]] = getelementptr inbounds double*, double** %[[i2]], i64 %iv
; CHECK-NEXT:   %.pre = load i64*, i64** %[[i3]], align 8, !invariant.group !1
; CHECK-NEXT:   %[[pre2:.+]] = load double*, double** %[[i4]], align 8, !invariant.group !2
; CHECK-NEXT:   br label %body

; CHECK: body:                                             ; preds = %body, %loop
; CHECK-NEXT:   %iv1 = phi i64 [ %iv.next2, %body ], [ 0, %loop ]
; CHECK-NEXT:   %[[i5:.+]] = getelementptr inbounds i64, i64* %.pre, i64 %iv1
; CHECK-NEXT:   %idx = load i64, i64* %[[i5]], align 8, !invariant.group !3
; CHECK-NEXT:   %iv.next2 = add nuw nsw i64 %iv1, 1
; CHECK-NEXT:   %[[i6:.+]] = getelementptr inbounds double, double* %[[pre2]], i64 %iv1
; CHECK-NEXT:   %ld = load double, double* %[[i6]], align 8, !invariant.group !4
; CHECK-NEXT:   %cmp = fcmp oeq double %ld, 0x400921FAFC8B007A
; CHECK-NEXT:   br i1 %cmp, label %body, label %end

; CHECK: end:                                              ; preds = %body
; CHECK-NEXT:   %"gep2'ipg" = getelementptr inbounds double, double* %"x'", i64 %idx
; CHECK-NEXT:   %[[i7:.+]] = load double, double* %"gep2'ipg"
; CHECK-NEXT:   %[[i8]] = fadd fast double %[[i7]], %[[dsum]]
; CHECK-NEXT:   %cmp2 = icmp ne i64 %iv.next, 10
; CHECK-NEXT:   br i1 %cmp2, label %loop, label %exit

; CHECK: exit:                                             ; preds = %end
; CHECK-NEXT:   ret double %[[i8]]
; CHECK-NEXT: }
