; RUN: %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=sum_list -o /dev/null | FileCheck %s

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

%struct.n = type { double, %struct.n* }

; Function Attrs: noinline norecurse nounwind readonly uwtable
define dso_local double @sum_list(%struct.n* noalias readonly %node, i64 %times) local_unnamed_addr #0 {
entry:
  br label %for

for:                              ; preds = %for.cond.cleanup4, %entry
  %val.020 = phi %struct.n* [ %l1, %for ], [ %node, %entry ]
  %sum.019 = phi double [ %add, %for ], [ 0.000000e+00, %entry ]
  %values = getelementptr inbounds %struct.n, %struct.n* %val.020, i64 0, i32 0
  %l0 = load double, double* %values, align 8, !tbaa !8
  %add = fadd fast double %l0, %sum.019
  %next = getelementptr inbounds %struct.n, %struct.n* %val.020, i64 0, i32 1
  %l1 = load %struct.n*, %struct.n** %next, align 8, !tbaa !7
  %cmp = icmp eq %struct.n* %l1, null
  br i1 %cmp, label %for.cond.cleanup, label %for

for.cond.cleanup:                                 ; preds = %for.cond.cleanup4, %entry
  ret double %add
}

; Function Attrs: nounwind
declare dso_local noalias i8* @malloc(i64) local_unnamed_addr #2

; Function Attrs: noinline nounwind uwtable
define dso_local double @derivative(%struct.n* %x, %struct.n* %xp, i64 %n) {
entry:
  %0 = tail call double (double (%struct.n*, i64)*, ...) @__enzyme_autodiff(double (%struct.n*, i64)* nonnull @sum_list, %struct.n* %x, %struct.n* %xp, i64 %n)
  ret double %0
}

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (%struct.n*, i64)*, ...) #4


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

; TODO nicely present recursive structure

; CHECK: sum_list - {} |{}:{} {}:{} 
; CHECK-NEXT: %struct.n* %node: {[-1]:Pointer, [-1,0]:Float@double, [-1,8]:Pointer, [-1,8,0]:Float@double, [-1,8,8]:Pointer, [-1,8,8,0]:Float@double, [-1,8,8,8]:Pointer, [-1,8,8,8,0]:Float@double, [-1,8,8,8,8]:Pointer, [-1,8,8,8,8,0]:Float@double, [-1,8,8,8,8,8]:Pointer}
; CHECK-NEXT: i64 %times: {}
; CHECK-NEXT: entry
; CHECK-NEXT:   br label %for: {}
; CHECK-NEXT: for
; CHECK-NEXT:   %val.020 = phi %struct.n* [ %l1, %for ], [ %node, %entry ]: {[-1]:Pointer, [-1,0]:Float@double, [-1,8]:Pointer, [-1,8,0]:Float@double, [-1,8,8]:Pointer, [-1,8,8,0]:Float@double, [-1,8,8,8]:Pointer, [-1,8,8,8,0]:Float@double, [-1,8,8,8,8]:Pointer, [-1,8,8,8,8,0]:Float@double, [-1,8,8,8,8,8]:Pointer}
; CHECK-NEXT:   %sum.019 = phi double [ %add, %for ], [ 0.000000e+00, %entry ]: {[-1]:Float@double}
; CHECK-NEXT:   %values = getelementptr inbounds %struct.n, %struct.n* %val.020, i64 0, i32 0: {[-1]:Pointer, [-1,0]:Float@double}
; CHECK-NEXT:   %l0 = load double, double* %values, align 8, !tbaa !2: {[-1]:Float@double}
; CHECK-NEXT:   %add = fadd fast double %l0, %sum.019: {[-1]:Float@double}
; CHECK-NEXT:   %next = getelementptr inbounds %struct.n, %struct.n* %val.020, i64 0, i32 1: {[-1]:Pointer, [-1,0]:Pointer, [-1,0,0]:Float@double, [-1,0,8]:Pointer, [-1,0,8,0]:Float@double, [-1,0,8,8]:Pointer, [-1,0,8,8,0]:Float@double, [-1,0,8,8,8]:Pointer, [-1,0,8,8,8,0]:Float@double, [-1,0,8,8,8,8]:Pointer}
; CHECK-NEXT:   %l1 = load %struct.n*, %struct.n** %next, align 8, !tbaa !6: {[-1]:Pointer, [-1,0]:Float@double, [-1,8]:Pointer, [-1,8,0]:Float@double, [-1,8,8]:Pointer, [-1,8,8,0]:Float@double, [-1,8,8,8]:Pointer, [-1,8,8,8,0]:Float@double, [-1,8,8,8,8]:Pointer, [-1,8,8,8,8,0]:Float@double, [-1,8,8,8,8,8]:Pointer}
; CHECK-NEXT:   %cmp = icmp eq %struct.n* %l1, null: {[-1]:Integer}
; CHECK-NEXT:   br i1 %cmp, label %for.cond.cleanup, label %for: {}
; CHECK-NEXT: for.cond.cleanup
; CHECK-NEXT:   ret double %add: {}