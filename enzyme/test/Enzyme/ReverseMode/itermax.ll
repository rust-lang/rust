; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -inline -mem2reg -correlated-propagation -adce -instcombine -instsimplify -early-cse-memssa -simplifycfg -correlated-propagation -adce -jump-threading -instsimplify -early-cse -simplifycfg -S | FileCheck %s

; #include <math.h>
; #include <stdio.h>
;
; static double max(double x, double y) {
;     return (x > y) ? x : y;
; }
;
; __attribute__((noinline))
; static double iterA(double *__restrict x, size_t n) {
;   double A = x[0];
;   for(int i=0; i<=n; i++) {
;     A = max(A, x[i]);
;   }
;   return A;
; }
;
; void dsincos(double *__restrict x, double *__restrict xp, size_t n) {
;     __builtin_autodiff(iterA, x, xp, n);
; }

; Function Attrs: nounwind uwtable
define dso_local void @dsincos(double* noalias %x, double* noalias %xp, i64 %n) local_unnamed_addr #0 {
entry:
  %0 = tail call double (double (double*, i64)*, ...) @__enzyme_autodiff(double (double*, i64)* nonnull @iterA, double* %x, double* %xp, i64 %n)
  ret void
}

; Function Attrs: noinline norecurse nounwind readonly uwtable
define internal double @iterA(double* noalias nocapture readonly %x, i64 %n) #1 {
entry:
  %0 = load double, double* %x, align 8, !tbaa !2
  %exitcond11 = icmp eq i64 %n, 0
  br i1 %exitcond11, label %for.cond.cleanup, label %for.body.for.body_crit_edge

for.cond.cleanup:                                 ; preds = %for.body.for.body_crit_edge, %entry
  %cond.i.lcssa = phi double [ %0, %entry ], [ %cond.i, %for.body.for.body_crit_edge ]
  ret double %cond.i.lcssa

for.body.for.body_crit_edge:                      ; preds = %entry, %for.body.for.body_crit_edge
  %indvars.iv.next13 = phi i64 [ %indvars.iv.next, %for.body.for.body_crit_edge ], [ 1, %entry ]
  %cond.i12 = phi double [ %cond.i, %for.body.for.body_crit_edge ], [ %0, %entry ]
  %arrayidx2.phi.trans.insert = getelementptr inbounds double, double* %x, i64 %indvars.iv.next13
  %.pre = load double, double* %arrayidx2.phi.trans.insert, align 8, !tbaa !2
  %cmp.i = fcmp fast ogt double %cond.i12, %.pre
  %cond.i = select i1 %cmp.i, double %cond.i12, double %.pre
  %indvars.iv.next = add nuw i64 %indvars.iv.next13, 1
  %exitcond = icmp eq i64 %indvars.iv.next13, %n
  br i1 %exitcond, label %for.cond.cleanup, label %for.body.for.body_crit_edge
}

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double*, i64)*, ...) #2

attributes #0 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { noinline norecurse nounwind readonly uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #2 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0}
!3 = !{!"double", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}

; CHECK: define internal void @diffeiterA(double* noalias nocapture readonly %x, double* nocapture %"x'", i64 %n, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %exitcond11 = icmp eq i64 %n, 0
; CHECK-NEXT:   br i1 %exitcond11, label %invertentry, label %for.body.for.body_crit_edge.preheader

; CHECK: for.body.for.body_crit_edge.preheader:            ; preds = %entry
; CHECK-NEXT:   %0 = load double, double* %x, align 8, !tbaa !2
; CHECK-NEXT:   br label %for.body.for.body_crit_edge

; CHECK: for.body.for.body_crit_edge:                      ; preds = %for.body.for.body_crit_edge, %for.body.for.body_crit_edge.preheader
; CHECK-NEXT:   %1 = phi i64 [ 0, %for.body.for.body_crit_edge.preheader ], [ %2, %for.body.for.body_crit_edge ]
; CHECK-NEXT:   %iv = phi i64 [ 0, %for.body.for.body_crit_edge.preheader ], [ %iv.next, %for.body.for.body_crit_edge ]
; CHECK-NEXT:   %cond.i12 = phi double [ %0, %for.body.for.body_crit_edge.preheader ], [ %cond.i, %for.body.for.body_crit_edge ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %arrayidx2.phi.trans.insert = getelementptr inbounds double, double* %x, i64 %iv.next
; CHECK-NEXT:   %.pre = load double, double* %arrayidx2.phi.trans.insert, align 8, !tbaa !2
; CHECK-NEXT:   %cmp.i = fcmp fast ogt double %cond.i12, %.pre
; CHECK-NEXT:   %2 = select i1 %cmp.i, i64 %1, i64 %iv.next
; CHECK-NEXT:   %cond.i = select i1 %cmp.i, double %cond.i12, double %.pre
; CHECK-NEXT:   %exitcond = icmp eq i64 %iv.next, %n
; CHECK-NEXT:   br i1 %exitcond, label %invertfor.body.for.body_crit_edge, label %for.body.for.body_crit_edge

; CHECK: invertentry:                                      ; preds = %entry, %invertfor.body.for.body_crit_edge.preheader
; CHECK-NEXT:   %"'de.0" = phi double [ %6, %invertfor.body.for.body_crit_edge.preheader ], [ %differeturn, %entry ]
; CHECK-NEXT:   %3 = load double, double* %"x'", align 8
; CHECK-NEXT:   %4 = fadd fast double %3, %"'de.0"
; CHECK-NEXT:   store double %4, double* %"x'", align 8
; CHECK-NEXT:   ret void

; CHECK: invertfor.body.for.body_crit_edge.preheader:  
; CHECK-NEXT:   %5 = icmp eq i64 %2, 0
; CHECK-NEXT:   %6 = select{{( fast)?}} i1 %5, double %differeturn, double 0.000000e+00
; CHECK-NEXT:   br label %invertentry

; CHECK: invertfor.body.for.body_crit_edge: 
; CHECK-NEXT:   %"iv'ac.0.in" = phi i64 [ %"iv'ac.0", %invertfor.body.for.body_crit_edge ], [ %n, %for.body.for.body_crit_edge ]
; CHECK-NEXT:   %"iv'ac.0" = add i64 %"iv'ac.0.in", -1
; CHECK-NEXT:   %7 = icmp eq i64 %2, %"iv'ac.0.in"
; CHECK-NEXT:   %8 = select{{( fast)?}} i1 %7, double %differeturn, double 0.000000e+00
; CHECK-NEXT:   %"arrayidx2.phi.trans.insert'ipg_unwrap" = getelementptr inbounds double, double* %"x'", i64 %"iv'ac.0.in"
; CHECK-NEXT:   %9 = load double, double* %"arrayidx2.phi.trans.insert'ipg_unwrap", align 8
; CHECK-NEXT:   %10 = fadd fast double %9, %8
; CHECK-NEXT:   store double %10, double* %"arrayidx2.phi.trans.insert'ipg_unwrap", align 8
; CHECK-NEXT:   %11 = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %11, label %invertfor.body.for.body_crit_edge.preheader, label %invertfor.body.for.body_crit_edge
; CHECK-NEXT: }
