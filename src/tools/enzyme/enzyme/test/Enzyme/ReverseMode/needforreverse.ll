; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s

source_filename = "badalloc.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@enzyme_dup = external dso_local local_unnamed_addr global i32, align 4
@enzyme_const = external dso_local local_unnamed_addr global i32, align 4
@enzyme_dupnoneed = external dso_local local_unnamed_addr global i32, align 4

define void @project(double* noalias nocapture readonly %cam, double* noalias nocapture %proj) {
entry:
  %Xcam = alloca [2 x double], align 16
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, double* %cam, i64 %indvars.iv
  %a4 = load double, double* %arrayidx, align 8, !tbaa !2
  %add = fadd fast double %a4, 2.000000e+00
  %arrayidx2 = getelementptr inbounds [2 x double], [2 x double]* %Xcam, i64 0, i64 %indvars.iv
  store double %add, double* %arrayidx2, align 8, !tbaa !2
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 2
  br i1 %exitcond, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  %arrayidx3 = getelementptr inbounds [2 x double], [2 x double]* %Xcam, i64 0, i64 0
  %a1 = load double, double* %arrayidx3, align 16, !tbaa !2
  %a2 = load double, double* %cam, align 8, !tbaa !2
  %mul = fmul fast double %a2, %a1
  store double %mul, double* %proj, align 8, !tbaa !2
  %arrayidx6 = getelementptr inbounds [2 x double], [2 x double]* %Xcam, i64 0, i64 1
  %a3 = load double, double* %arrayidx6, align 8, !tbaa !2
  %mul8 = fmul fast double %a3, %a2
  %arrayidx9 = getelementptr inbounds double, double* %proj, i64 1
  store double %mul8, double* %arrayidx9, align 8, !tbaa !2
  ret void
}

; Function Attrs: nounwind uwtable
define void @compute_reproj_error(double* noalias nocapture readonly %cam, double* noalias nocapture readonly %w, double* noalias nocapture readnone %feat, double* noalias nocapture %err) {
entry:
  %proj = alloca [2 x double], align 16
  %arraydecay = getelementptr inbounds [2 x double], [2 x double]* %proj, i64 0, i64 0
  call void @project(double* %cam, double* nonnull %arraydecay)
  %a1 = load double, double* %w, align 8, !tbaa !2
  %a2 = load double, double* %arraydecay, align 16, !tbaa !2
  %mul = fmul fast double %a2, %a1
  store double %mul, double* %err, align 8, !tbaa !2
  %arrayidx2 = getelementptr inbounds [2 x double], [2 x double]* %proj, i64 0, i64 1
  %a3 = load double, double* %arrayidx2, align 8, !tbaa !2
  %mul3 = fmul fast double %a3, %a1
  %arrayidx4 = getelementptr inbounds double, double* %err, i64 1
  store double %mul3, double* %arrayidx4, align 8, !tbaa !2
  ret void
}

; Function Attrs: nounwind uwtable
define void @dcompute_reproj_error(double* %cam, double* %dcam, double* %w, double* %wb, double* %feat, double* %err, double* %derr) {
entry:
  %0 = load i32, i32* @enzyme_dup, align 4, !tbaa !6
  %1 = load i32, i32* @enzyme_const, align 4, !tbaa !6
  %2 = load i32, i32* @enzyme_dupnoneed, align 4, !tbaa !6
  tail call void (i8*, ...) @__enzyme_autodiff(i8* bitcast (void (double*, double*, double*, double*)* @compute_reproj_error to i8*), i32 %0, double* %cam, double* %dcam, i32 %0, double* %w, double* %wb, i32 %1, double* %feat, i32 %2, double* %err, double* %derr) #4
  ret void
}

declare void @__enzyme_autodiff(i8*, ...) local_unnamed_addr #3

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0}
!3 = !{!"double", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
!6 = !{!7, !7, i64 0}
!7 = !{!"int", !4, i64 0}

; CHECK-LABEL: define internal void @diffeproject
; (double* noalias nocapture readonly %cam, double* nocapture %"cam'", double* noalias nocapture %proj, double* nocapture %"proj'")
; CHECK: %Xcam = alloca [2 x double], i64 1, align 16
; CHECK: %0 = bitcast [2 x double]* %Xcam to i8*
; CHECK: %2 = bitcast i8* %0 to [2 x double]*

; CHECK-LABEL: for.body
; CHECK: %arrayidx2 = getelementptr inbounds [2 x double], [2 x double]* %2, i64 0, i64 %iv
; CHECK: store double %add, double* %arrayidx2, align 8

