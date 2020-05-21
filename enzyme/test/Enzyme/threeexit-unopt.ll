; RUN: %opt < %s %loadEnzyme -enzyme -enzyme_preopt=false -mem2reg -instsimplify -adce -loop-deletion -correlated-propagation -simplifycfg -S | FileCheck %s
source_filename = "threeexit.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind readnone uwtable
define dso_local zeroext i1 @approx_fp_equality_float(float %f1, float %f2, double %threshold) local_unnamed_addr #0 {
entry:
  %sub = fsub fast float %f1, %f2
  %0 = tail call fast float @llvm.fabs.f32(float %sub)
  %1 = fpext float %0 to double
  %cmp = fcmp fast ule double %1, %threshold
  ret i1 %cmp
}

; Function Attrs: norecurse nounwind uwtable
define dso_local void @compute_loops(double* noalias nocapture readonly %in, double* noalias nocapture %out) #1 {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.inc
  ret void

for.body:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.inc ]
  %arrayidx = getelementptr inbounds double, double* %in, i64 %indvars.iv
  %0 = load double, double* %arrayidx, align 8, !tbaa !2
  %cmp1 = fcmp fast ogt double %0, 1.000000e+00
  br i1 %cmp1, label %if.then, label %if.else

if.then:                                          ; preds = %for.body
  %mul = fmul fast double %0, 2.000000e+00
  br label %for.inc

if.else:                                          ; preds = %for.body
  %cmp8 = fcmp fast ogt double %0, 0.000000e+00
  br i1 %cmp8, label %if.then9, label %for.inc

if.then9:                                         ; preds = %if.else
  %add = fadd fast double %0, 3.000000e+00
  br label %for.inc

for.inc:                                          ; preds = %if.else, %if.then, %if.then9
  %mul.sink = phi double [ %mul, %if.then ], [ %add, %if.then9 ], [ 4.000000e+00, %if.else ]
  %arrayidx5 = getelementptr inbounds double, double* %out, i64 %indvars.iv
  store double %mul.sink, double* %arrayidx5, align 8, !tbaa !2
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 100
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

; Function Attrs: nounwind uwtable
define dso_local i32 @main(i32 %argc, i8** nocapture readnone %argv) local_unnamed_addr #2 {
entry:
  %agg-temp = alloca [100 x double], align 8
  %agg-temp1 = alloca [100 x double], align 8
  %agg-temp2 = alloca [100 x double], align 8
  %agg-temp3 = alloca [100 x double], align 8
  %0 = getelementptr inbounds [100 x double], [100 x double]* %agg-temp, i64 0, i64 0
  %1 = getelementptr inbounds [100 x double], [100 x double]* %agg-temp1, i64 0, i64 0
  %2 = getelementptr inbounds [100 x double], [100 x double]* %agg-temp2, i64 0, i64 0
  %3 = getelementptr inbounds [100 x double], [100 x double]* %agg-temp3, i64 0, i64 0
  %4 = call double (...) @__enzyme_autodiff.f64(void (double*, double*)* nonnull @compute_loops, double* nonnull %0, double* nonnull %1, double* nonnull %2, double* nonnull %3) #4
  ret i32 0
}

declare double @__enzyme_autodiff.f64(...) local_unnamed_addr

; Function Attrs: nounwind readnone speculatable
declare float @llvm.fabs.f32(float) #3

attributes #0 = { nounwind readnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #2 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #3 = { nounwind readnone speculatable }
attributes #4 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0}
!3 = !{!"double", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}

; CHECK: define internal void @diffecompute_loops(double* noalias nocapture readonly %in, double* nocapture %"in'", double* noalias nocapture %out, double* nocapture %"out'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   br label %for.body

; CHECK: for.body:                                         ; preds = %for.inc, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %for.inc ], [ 0, %entry ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %arrayidx = getelementptr inbounds double, double* %in, i64 %iv
; CHECK-NEXT:   %0 = load double, double* %arrayidx, align 8, !tbaa !2, !invariant.group !6
; CHECK-NEXT:   %cmp1 = fcmp fast ogt double %0, 1.000000e+00
; CHECK-NEXT:   br i1 %cmp1, label %if.then, label %if.else

; CHECK: if.then:                                          ; preds = %for.body
; CHECK-NEXT:   %mul = fmul fast double %0, 2.000000e+00
; CHECK-NEXT:   br label %for.inc

; CHECK: if.else:                                          ; preds = %for.body
; CHECK-NEXT:   %cmp8 = fcmp fast ogt double %0, 0.000000e+00
; CHECK-NEXT:   br i1 %cmp8, label %if.then9, label %for.inc

; CHECK: if.then9:                                         ; preds = %if.else
; CHECK-NEXT:   %add = fadd fast double %0, 3.000000e+00
; CHECK-NEXT:   br label %for.inc

; CHECK: for.inc:                                          ; preds = %if.then9, %if.else, %if.then
; CHECK-NEXT:   %mul.sink = phi double [ %mul, %if.then ], [ %add, %if.then9 ], [ 4.000000e+00, %if.else ]
; CHECK-NEXT:   %arrayidx5 = getelementptr inbounds double, double* %out, i64 %iv
; CHECK-NEXT:   store double %mul.sink, double* %arrayidx5, align 8, !tbaa !2
; CHECK-NEXT:   %exitcond = icmp eq i64 %iv.next, 100
; CHECK-NEXT:   br i1 %exitcond, label %invertfor.inc, label %for.body

; CHECK: invertentry:                                      ; preds = %invertfor.inc
; CHECK-NEXT:   ret void

; CHECK: incinvertfor.body:                                ; preds = %invertfor.inc
; CHECK-NEXT:   %1 = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertfor.inc

; CHECK: invertfor.inc:                                    ; preds = %for.inc, %incinvertfor.body
; CHECK-NEXT:   %"add'de.2" = phi double [ %"add'de.0", %incinvertfor.body ], [ 0.000000e+00, %for.inc ]
; CHECK-NEXT:   %"mul'de.1" = phi double [ %"mul'de.0", %incinvertfor.body ], [ 0.000000e+00, %for.inc ]
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %1, %incinvertfor.body ], [ 99, %for.inc ]
; CHECK-NEXT:   %"arrayidx5'ipg_unwrap" = getelementptr inbounds double, double* %"out'", i64 %"iv'ac.0"
; CHECK-NEXT:   %2 = load double, double* %"arrayidx5'ipg_unwrap", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"arrayidx5'ipg_unwrap", align 8
; CHECK-NEXT:   %arrayidx_unwrap = getelementptr inbounds double, double* %in, i64 %"iv'ac.0"
; CHECK-NEXT:   %_unwrap = load double, double* %arrayidx_unwrap, align 8, !tbaa !2, !invariant.group !6, !enzyme_unwrapped !7
; CHECK-NEXT:   %cmp1_unwrap = fcmp fast ogt double %_unwrap, 1.000000e+00
; CHECK-NEXT:   %cmp8_unwrap = fcmp fast ogt double %_unwrap, 0.000000e+00
; CHECK-NEXT:   %anot1_ = xor i1 %cmp1_unwrap, true
; CHECK-NEXT:   %andVal = and i1 %cmp8_unwrap, %anot1_
; CHECK-NEXT:   %3 = fadd fast double %"add'de.2", %2
; CHECK-NEXT:   %4 = select i1 %andVal, double %3, double %"add'de.2"
; CHECK-NEXT:   %5 = fadd fast double %"mul'de.1", %2
; CHECK-NEXT:   %6 = select i1 %cmp1_unwrap, double %5, double %"mul'de.1"
; CHECK-NEXT:   %m0diffe = fmul fast double %6, 2.000000e+00
; CHECK-NEXT:   %"add'de.1" = select i1 %cmp8_unwrap, double 0.000000e+00, double %"add'de.2"
; CHECK-NEXT:   %"'de.1" = select i1 %cmp8_unwrap, double %4, double 0.000000e+00
; CHECK-NEXT:   %"add'de.0" = select i1 %cmp1_unwrap, double %4, double %"add'de.1"
; CHECK-NEXT:   %"mul'de.0" = select i1 %cmp1_unwrap, double 0.000000e+00, double %"mul'de.1"
; CHECK-NEXT:   %"'de.0" = select i1 %cmp1_unwrap, double %m0diffe, double %"'de.1"
; CHECK-NEXT:   %"arrayidx'ipg_unwrap" = getelementptr inbounds double, double* %"in'", i64 %"iv'ac.0"
; CHECK-NEXT:   %7 = load double, double* %"arrayidx'ipg_unwrap", align 8
; CHECK-NEXT:   %8 = fadd fast double %7, %"'de.0"
; CHECK-NEXT:   store double %8, double* %"arrayidx'ipg_unwrap", align 8
; CHECK-NEXT:   %9 = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %9, label %invertentry, label %incinvertfor.body
; CHECK-NEXT: }
