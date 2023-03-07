; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -adce -loop-deletion -correlated-propagation -simplifycfg -S | FileCheck %s

declare i8* @malloc(i64)
declare void @free(i8*)

; Function Attrs: norecurse nounwind readonly uwtable
define dso_local double @subsum(i64** %off, double* nocapture readonly %x, i64 %n) #0 {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret double %add

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %total.07 = phi double [ 0.000000e+00, %entry ], [ %add, %for.body ]
  %pidx = getelementptr inbounds i64*, i64** %off, i64 %indvars.iv
  %qidx = load i64*, i64** %pidx, align 8
  %idx = load i64, i64* %qidx, align 8
  %arrayidx = getelementptr inbounds double, double* %x, i64 %idx
  %0 = load double, double* %arrayidx, align 8
  %add = fadd fast double %0, %total.07
  %indvars.iv.next = add nuw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv, %n
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

define dso_local double @sum(i64** %off, double* nocapture readonly %x, i64 %n) #0 {
entry:
  %res = call double @subsum(i64** %off, double* %x, i64 %n)
  store double 0.000000e+00, double* %x
  store i64* null, i64** %off
  ret double %res
}

; Function Attrs: nounwind uwtable
define dso_local void @dsum(i64** %off, i64** %doff, double* %x, double* %xp, i64 %n) local_unnamed_addr #1 {
entry:
  %0 = tail call double (double (i64**, double*, i64)*, ...) @__enzyme_autodiff(double (i64**, double*, i64)* nonnull @sum, metadata !"enzyme_dup", i64** %off, i64** %doff, double* %x, double* %xp, i64 %n)
  ret void
}

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (i64**, double*, i64)*, ...) #2

attributes #0 = { norecurse nounwind readonly uwtable }
attributes #1 = { nounwind uwtable }
attributes #2 = { nounwind }

; CHECK: define internal i64* @augmented_subsum(i64** %off, i64** %"off'", double* nocapture readonly %x, double* nocapture %"x'", i64 %n)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = add nuw i64 %n, 1
; CHECK-NEXT:   %mallocsize = mul nuw nsw i64 %0, 8
; CHECK-NEXT:   %malloccall = tail call noalias nonnull i8* @malloc(i64 %mallocsize)
; CHECK-NEXT:   %idx_malloccache = bitcast i8* %malloccall to i64*
; CHECK-NEXT:   br label %for.body

; CHECK: for.cond.cleanup:                                 ; preds = %for.body
; CHECK-NEXT:   ret i64* %idx_malloccache

; CHECK: for.body:                                         ; preds = %for.body, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %pidx = getelementptr inbounds i64*, i64** %off, i64 %iv
; CHECK-NEXT:   %qidx = load i64*, i64** %pidx, align 8
; CHECK-NEXT:   %idx = load i64, i64* %qidx, align 8
; CHECK-NEXT:   %1 = getelementptr inbounds i64, i64* %idx_malloccache, i64 %iv
; CHECK-NEXT:   store i64 %idx, i64* %1, align 8, !invariant.group !
; CHECK-NEXT:   %exitcond = icmp eq i64 %iv, %n
; CHECK-NEXT:   br i1 %exitcond, label %for.cond.cleanup, label %for.body
; CHECK-NEXT: }

; CHECK: define internal void @diffesubsum(i64** %off, i64** %"off'", double* nocapture readonly %x, double* nocapture %"x'", i64 %n, double %differeturn, i64* %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   br label %invertfor.body

; CHECK: invertentry:                                      ; preds = %invertfor.body
; CHECK-NEXT:   %0 = bitcast i64* %tapeArg to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %0)
; CHECK-NEXT:   ret void

; CHECK: invertfor.body:                                   ; preds = %incinvertfor.body, %entry
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %n, %entry ], [ %7, %incinvertfor.body ]
; CHECK-NEXT:   %1 = getelementptr inbounds i64, i64* %tapeArg, i64 %"iv'ac.0"
; CHECK-NEXT:   %2 = load i64, i64* %1, align 8, !invariant.group !
; CHECK-NEXT:   %"arrayidx'ipg_unwrap" = getelementptr inbounds double, double* %"x'", i64 %2
; CHECK-NEXT:   %3 = load double, double* %"arrayidx'ipg_unwrap", align 8
; CHECK-NEXT:   %4 = fadd fast double %3, %differeturn
; CHECK-NEXT:   store double %4, double* %"arrayidx'ipg_unwrap", align 8
; CHECK-NEXT:   %5 = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   %6 = select{{( fast)?}} i1 %5, double 0.000000e+00, double %differeturn
; CHECK-NEXT:   br i1 %5, label %invertentry, label %incinvertfor.body

; CHECK: incinvertfor.body:                                ; preds = %invertfor.body
; CHECK-NEXT:   %7 = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertfor.body
; CHECK-NEXT: }
