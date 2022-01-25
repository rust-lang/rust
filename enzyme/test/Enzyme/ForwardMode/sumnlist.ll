; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -gvn -early-cse-memssa -instcombine -instsimplify -simplifycfg -adce -licm -correlated-propagation -instcombine -correlated-propagation -adce -instsimplify -correlated-propagation -jump-threading -instsimplify -early-cse -simplifycfg -S | FileCheck %s

; #include <stdlib.h>
; #include <stdio.h>
;
; struct n {
;     double *values;
;     struct n *next;
; };
;
; __attribute__((noinline))
; double sum_list(const struct n *__restrict node, unsigned long times) {
;     double sum = 0;
;     for(const struct n *val = node; val != 0; val = val->next) {
;         for(int i=0; i<=times; i++) {
;             sum += val->values[i];
;         }
;     }
;     return sum;
; }

%struct.n = type { double*, %struct.n* }

; Function Attrs: noinline norecurse nounwind readonly uwtable
define dso_local double @sum_list(%struct.n* noalias readonly %node, i64 %times) local_unnamed_addr #0 {
entry:
  %cmp18 = icmp eq %struct.n* %node, null
  br i1 %cmp18, label %for.cond.cleanup, label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.cond.cleanup4, %entry
  %val.020 = phi %struct.n* [ %1, %for.cond.cleanup4 ], [ %node, %entry ]
  %sum.019 = phi double [ %add, %for.cond.cleanup4 ], [ 0.000000e+00, %entry ]
  %values = getelementptr inbounds %struct.n, %struct.n* %val.020, i64 0, i32 0
  %0 = load double*, double** %values, align 8, !tbaa !2
  br label %for.body5

for.cond.cleanup:                                 ; preds = %for.cond.cleanup4, %entry
  %sum.0.lcssa = phi double [ 0.000000e+00, %entry ], [ %add, %for.cond.cleanup4 ]
  ret double %sum.0.lcssa

for.cond.cleanup4:                                ; preds = %for.body5
  %next = getelementptr inbounds %struct.n, %struct.n* %val.020, i64 0, i32 1
  %1 = load %struct.n*, %struct.n** %next, align 8, !tbaa !7
  %cmp = icmp eq %struct.n* %1, null
  br i1 %cmp, label %for.cond.cleanup, label %for.cond1.preheader

for.body5:                                        ; preds = %for.body5, %for.cond1.preheader
  %indvars.iv = phi i64 [ 0, %for.cond1.preheader ], [ %indvars.iv.next, %for.body5 ]
  %sum.116 = phi double [ %sum.019, %for.cond1.preheader ], [ %add, %for.body5 ]
  %arrayidx = getelementptr inbounds double, double* %0, i64 %indvars.iv
  %2 = load double, double* %arrayidx, align 8, !tbaa !8
  %add = fadd fast double %2, %sum.116
  %indvars.iv.next = add nuw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv, %times
  br i1 %exitcond, label %for.cond.cleanup4, label %for.body5
}

; Function Attrs: nounwind
declare dso_local noalias i8* @malloc(i64) local_unnamed_addr #2

; Function Attrs: noinline nounwind uwtable
define dso_local double @derivative(%struct.n* %x, %struct.n* %xp, i64 %n) {
entry:
  %0 = tail call double (double (%struct.n*, i64)*, ...) @__enzyme_fwddiff(double (%struct.n*, i64)* nonnull @sum_list, %struct.n* %x, %struct.n* %xp, i64 %n)
  ret double %0
}

; Function Attrs: nounwind
declare double @__enzyme_fwddiff(double (%struct.n*, i64)*, ...) #4


attributes #0 = { noinline norecurse nounwind readonly uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #2 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #3 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #4 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !4, i64 0}
!3 = !{!"n", !4, i64 0, !4, i64 8}
!4 = !{!"any pointer", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C/C++ TBAA"}
!7 = !{!3, !4, i64 8}
!8 = !{!9, !9, i64 0}
!9 = !{!"double", !5, i64 0}
!10 = !{!4, !4, i64 0}


; CHECK: define internal double @fwddiffesum_list(%struct.n* noalias readonly %node, %struct.n* %"node'", i64 %times)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %cmp18 = icmp eq %struct.n* %node, null
; CHECK-NEXT:   br i1 %cmp18, label %for.cond.cleanup, label %for.cond1.preheader

; CHECK: for.cond1.preheader:                              ; preds = %entry, %for.cond.cleanup4
; CHECK-NEXT:   %0 = phi %struct.n* [ %[[ipl4:.+]], %for.cond.cleanup4 ], [ %"node'", %entry ]
; CHECK-NEXT:   %val.020 = phi %struct.n* [ %1, %for.cond.cleanup4 ], [ %node, %entry ]
; CHECK-NEXT:   %"sum.019'" = phi {{(fast )?}}double [ %3, %for.cond.cleanup4 ], [ 0.000000e+00, %entry ]
; CHECK-NEXT:   %"values'ipg" = getelementptr inbounds %struct.n, %struct.n* %0, i64 0, i32 0
; CHECK-NEXT:   %"'ipl" = load double*, double** %"values'ipg", align 8
; CHECK-NEXT:   br label %for.body5

; CHECK: for.cond.cleanup:                                 ; preds = %for.cond.cleanup4, %entry
; CHECK-NEXT:   %"sum.0.lcssa'" = phi {{(fast )?}}double [ 0.000000e+00, %entry ], [ %3, %for.cond.cleanup4 ]
; CHECK-NEXT:   ret double %"sum.0.lcssa'

; CHECK: for.cond.cleanup4:                                ; preds = %for.body5
; CHECK-NEXT:   %"next'ipg" = getelementptr inbounds %struct.n, %struct.n* %0, i64 0, i32 1
; CHECK-NEXT:   %next = getelementptr inbounds %struct.n, %struct.n* %val.020, i64 0, i32 1
; CHECK-NEXT:   %[[ipl4]] = load %struct.n*, %struct.n** %"next'ipg", align 8
; CHECK-NEXT:   %1 = load %struct.n*, %struct.n** %next, align 8, !tbaa !7
; CHECK-NEXT:   %cmp = icmp eq %struct.n* %1, null
; CHECK-NEXT:   br i1 %cmp, label %for.cond.cleanup, label %for.cond1.preheader

; CHECK: for.body5:                                        ; preds = %for.body5, %for.cond1.preheader
; CHECK-NEXT:   %iv1 = phi i64 [ %iv.next2, %for.body5 ], [ 0, %for.cond1.preheader ]
; CHECK-NEXT:   %"sum.116'" = phi {{(fast )?}}double [ %3, %for.body5 ], [ %"sum.019'", %for.cond1.preheader ]
; CHECK-NEXT:   %iv.next2 = add nuw nsw i64 %iv1, 1
; CHECK-NEXT:   %"arrayidx'ipg" = getelementptr inbounds double, double* %"'ipl", i64 %iv1
; CHECK-NEXT:   %2 = load double, double* %"arrayidx'ipg", align 8
; CHECK-NEXT:   %3 = fadd fast double %2, %"sum.116'"
; CHECK-NEXT:   %exitcond = icmp eq i64 %iv1, %times
; CHECK-NEXT:   br i1 %exitcond, label %for.cond.cleanup4, label %for.body5
; CHECK-NEXT: }
