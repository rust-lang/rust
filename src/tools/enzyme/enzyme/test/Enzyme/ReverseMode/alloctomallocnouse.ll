; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -adce -loop-deletion -correlated-propagation -simplifycfg -S | FileCheck %s

declare i8* @malloc(i64)
declare void @free(i8*)

; Function Attrs: norecurse nounwind readonly uwtable
define dso_local double @subsum(double* nocapture readonly %x, i64 %n) #0 {
entry:
  %m = call i8* @malloc(i64 8)
  %v = bitcast i8* %m to double*
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  %res = load double, double* %v
  call void @free(i8* %m)
  ret double %res

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %total.07 = phi double [ 0.000000e+00, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds double, double* %x, i64 %indvars.iv
  %0 = load double, double* %arrayidx, align 8
  %add = fadd fast double %0, %total.07
  store double %add, double* %v
  %indvars.iv.next = add nuw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv, %n
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

define dso_local double @sum(double* nocapture readonly %x, i64 %n) #0 {
entry:
  %res = call double @subsum(double* %x, i64 %n)
  store double 0.000000e+00, double* %x
  ret double %res
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

; CHECK: define internal void @augmented_subsum(double* nocapture readonly %x, double* nocapture %"x'", i64 %n)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %m = call i8* @malloc(i64 8)
; CHECK-NEXT:   %v = bitcast i8* %m to double*
; CHECK-NEXT:   br label %for.body

; CHECK: for.cond.cleanup:                                 ; preds = %for.body
; CHECK-NEXT:   call void @free(i8* nonnull %m)
; CHECK-NEXT:   ret void

; CHECK: for.body:                                         ; preds = %for.body, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
; CHECK-NEXT:   %total.07 = phi double [ 0.000000e+00, %entry ], [ %add, %for.body ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %arrayidx = getelementptr inbounds double, double* %x, i64 %iv
; CHECK-NEXT:   %0 = load double, double* %arrayidx, align 8
; CHECK-NEXT:   %add = fadd fast double %0, %total.07
; CHECK-NEXT:   store double %add, double* %v
; CHECK-NEXT:   %exitcond = icmp eq i64 %iv, %n
; CHECK-NEXT:   br i1 %exitcond, label %for.cond.cleanup, label %for.body
; CHECK-NEXT: }


; CHECK: define internal void @diffesubsum(double* nocapture readonly %x, double* nocapture %"x'", i64 %n, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"m'mi" = call noalias nonnull dereferenceable(8) dereferenceable_or_null(8) i8* @malloc(i64 8)
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull dereferenceable(8) dereferenceable_or_null(8) %"m'mi", i8 0, i64 8, i1 false)
; CHECK-NEXT:   %"v'ipc" = bitcast i8* %"m'mi" to double*
; CHECK-NEXT:   %0 = load double, double* %"v'ipc"
; CHECK-NEXT:   %1 = fadd fast double %0, %differeturn
; CHECK-NEXT:   store double %1, double* %"v'ipc"
; CHECK-NEXT:   br label %invertfor.body

; CHECK: invertentry:                                      ; preds = %invertfor.body
; CHECK-NEXT:   tail call void @free(i8* nonnull %"m'mi")
; CHECK-NEXT:   ret void

; CHECK: invertfor.body:                                   ; preds = %incinvertfor.body, %entry
; CHECK-NEXT:   %"add'de.0" = phi double [ 0.000000e+00, %entry ], [ %3, %incinvertfor.body ]
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %n, %entry ], [ %8, %incinvertfor.body ]
; CHECK-NEXT:   %2 = load double, double* %"v'ipc"
; CHECK-NEXT:   store double 0.000000e+00, double* %"v'ipc"
; CHECK-NEXT:   %3 = fadd fast double %"add'de.0", %2
; CHECK-NEXT:   %"arrayidx'ipg_unwrap" = getelementptr inbounds double, double* %"x'", i64 %"iv'ac.0"
; CHECK-NEXT:   %4 = load double, double* %"arrayidx'ipg_unwrap", align 8
; CHECK-NEXT:   %5 = fadd fast double %4, %3
; CHECK-NEXT:   store double %5, double* %"arrayidx'ipg_unwrap", align 8
; CHECK-NEXT:   %6 = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   %7 = select{{( fast)?}} i1 %6, double 0.000000e+00, double %3
; CHECK-NEXT:   br i1 %6, label %invertentry, label %incinvertfor.body

; CHECK: incinvertfor.body:                                ; preds = %invertfor.body
; CHECK-NEXT:   %8 = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertfor.body
; CHECK-NEXT: }
