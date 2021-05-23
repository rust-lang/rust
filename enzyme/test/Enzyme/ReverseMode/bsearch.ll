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
  %call = tail call fast double @__enzyme_autodiff(i8* bitcast (double (double*, i64)* @f to i8*), double* %x, double* %xp, i64 %n)
  ret double %call
}

declare dso_local double @__enzyme_autodiff(i8*, double*, double*, i64) local_unnamed_addr

attributes #0 = { noinline norecurse nounwind uwtable }
attributes #1 = { noinline nounwind uwtable }

; CHECK: define internal void @diffef(double* nocapture %x, double* nocapture %"x'", i64 %n, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %malloccall = tail call noalias nonnull dereferenceable(80) dereferenceable_or_null(80) i8* @malloc(i64 80)
; CHECK-NEXT:   %loopLimit_malloccache = bitcast i8* %malloccall to i64*
; CHECK-NEXT:   br label %loop

; CHCEK: loop:                                             ; preds = %end, %entry
; CHCEK-NEXT:   %iv = phi i64 [ %iv.next, %end ], [ 0, %entry ]
; CHCEK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHCEK-NEXT:   %g0 = getelementptr inbounds double, double* %x, i64 %iv
; CHCEK-NEXT:   br label %body

; CHECK: body: 
; CHECK-NEXT:   %iv1 = phi i64 [ %iv.next2, %body ], [ 0, %loop ]
; CHECK-NEXT:   %iv.next2 = add nuw nsw i64 %iv1, 1
; CHECK-NEXT:   %gep = getelementptr inbounds double, double* %g0, i64 %iv1
; CHECK-NEXT:   %ld = load double, double* %gep, align 8
; CHECK-NEXT:   %cmp = fcmp oeq double %ld, 0x400921FAFC8B007A
; CHECK-NEXT:   br i1 %cmp, label %body, label %end

; CHECK: end:                                              ; preds = %body
; CHECK-NEXT:   %0 = getelementptr inbounds i64, i64* %loopLimit_malloccache, i64 %iv
; CHECK-NEXT:   store i64 %iv1, i64* %0, align 8
; CHECK-NEXT:   %cmp2 = icmp ne i64 %iv.next, 10
; CHECK-NEXT:   br i1 %cmp2, label %loop, label %invertend

; CHECK: invertentry:                                      ; preds = %invertloop
; CHECK-NEXT:   tail call void @free(i8* nonnull %malloccall)
; CHECK-NEXT:   ret void

; CHECK: invertloop:                                       ; preds = %invertbody
; CHECK-NEXT:   %[[icmp0:.+]] = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %[[icmp0]], label %invertentry, label %incinvertloop

; CHECK: incinvertloop:                                    ; preds = %invertloop
; CHECK-NEXT:   %2 = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertend

; CHECK: invertbody:                                       ; preds = %invertend, %incinvertbody
; CHECK-NEXT:   %"iv1'ac.0" = phi i64 [ 0, %invertend ], [ %4, %incinvertbody ]
; CHECK-NEXT:   %3 = icmp eq i64 %"iv1'ac.0", 0
; CHECK-NEXT:   br i1 %3, label %invertloop, label %incinvertbody

; CHECK: incinvertbody:                                    ; preds = %invertbody
; CHECK-NEXT:   %4 = add nsw i64 %"iv1'ac.0", -1
; CHECK-NEXT:   br label %invertbody

; CHECK: invertend:                                        ; preds = %end, %incinvertloop
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %2, %incinvertloop ], [ 9, %end ]
; CHECK-NEXT:   %5 = getelementptr inbounds i64, i64* %loopLimit_malloccache, i64 %"iv'ac.0"
; CHECK-NEXT:   %6 = load i64, i64* %5, align 8, !invariant.group !0
; CHECK-NEXT:   %"gep2'ipg_unwrap" = getelementptr inbounds double, double* %"x'", i64 %6
; CHECK-NEXT:   %7 = load double, double* %"gep2'ipg_unwrap", align 8
; CHECK-NEXT:   %8 = fadd fast double %7, %differeturn
; CHECK-NEXT:   store double %8, double* %"gep2'ipg_unwrap", align 8
; CHECK-NEXT:   br label %invertbody
; CHECK-NEXT: }