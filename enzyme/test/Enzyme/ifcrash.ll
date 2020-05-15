; RUN: %opt < %s %loadEnzyme -enzyme -enzyme_preopt=false -mem2reg -instsimplify -adce -loop-deletion -correlated-propagation -simplifycfg -S | FileCheck %s


; Function Attrs: norecurse nounwind uwtable
define dso_local float @insertsort_sum(float* nocapture %array) #0 {
entry:
  %arrayidx = getelementptr inbounds float, float* %array, i64 2
  %0 = load float, float* %arrayidx, align 4, !tbaa !2
  %arrayidx2 = getelementptr inbounds float, float* %array, i64 3
  %1 = load float, float* %arrayidx2, align 4, !tbaa !2
  %cmp = fcmp olt float %0, %1
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  store float %0, float* %arrayidx2, align 4, !tbaa !2
  store float %1, float* %arrayidx, align 4, !tbaa !2
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %.in = phi float [ %1, %if.then ], [ %0, %entry ]
  %arrayidx11 = getelementptr inbounds float, float* %array, i64 1
  %2 = load float, float* %arrayidx11, align 4, !tbaa !2
  %cmp14 = fcmp olt float %2, %.in
  br i1 %cmp14, label %if.then15, label %if.end21

if.then15:                                        ; preds = %if.end
  store float %2, float* %arrayidx, align 4, !tbaa !2
  store float %.in, float* %arrayidx11, align 4, !tbaa !2
  br label %if.end21

if.end21:                                         ; preds = %if.then15, %if.end
  %3 = load float, float* %array, align 4, !tbaa !2
  ret float %3
}

; Function Attrs: uwtable
define dso_local i32 @_Z10derivativePfS_(float* %array, float* %d_array) local_unnamed_addr #1 {
entry:
  %call = tail call double (...) @__enzyme_autodiff(float (float*)* nonnull @insertsort_sum, float* %array, float* %d_array)
  ret i32 0
}

declare dso_local double @__enzyme_autodiff(...) local_unnamed_addr

attributes #0 = { norecurse nounwind uwtable }
attributes #1 = { uwtable }

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0}
!3 = !{!"float", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C++ TBAA"}

; CHECK: define internal {{(dso_local )?}}void @diffeinsertsort_sum(float* nocapture %array
