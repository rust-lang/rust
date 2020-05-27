; RUN: %opt < %s %loadEnzyme -enzyme -enzyme_preopt=false -S | FileCheck %s

source_filename = "badalloc.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@diffe_dup = external dso_local local_unnamed_addr global i32, align 4
@diffe_const = external dso_local local_unnamed_addr global i32, align 4
@diffe_dupnoneed = external dso_local local_unnamed_addr global i32, align 4

; Function Attrs: noinline nounwind uwtable
define dso_local void @project(double* noalias nocapture readonly %cam, double* noalias nocapture %proj) local_unnamed_addr #0 {
entry:
  %Xcam = alloca [2 x double], align 16
  %0 = bitcast [2 x double]* %Xcam to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %0) #4
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  %arrayidx3 = getelementptr inbounds [2 x double], [2 x double]* %Xcam, i64 0, i64 0
  %1 = load double, double* %arrayidx3, align 16, !tbaa !2
  %2 = load double, double* %cam, align 8, !tbaa !2
  %mul = fmul fast double %2, %1
  store double %mul, double* %proj, align 8, !tbaa !2
  %arrayidx6 = getelementptr inbounds [2 x double], [2 x double]* %Xcam, i64 0, i64 1
  %3 = load double, double* %arrayidx6, align 8, !tbaa !2
  %mul8 = fmul fast double %3, %2
  %arrayidx9 = getelementptr inbounds double, double* %proj, i64 1
  store double %mul8, double* %arrayidx9, align 8, !tbaa !2
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %0) #4
  ret void

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, double* %cam, i64 %indvars.iv
  %4 = load double, double* %arrayidx, align 8, !tbaa !2
  %add = fadd fast double %4, 2.000000e+00
  %arrayidx2 = getelementptr inbounds [2 x double], [2 x double]* %Xcam, i64 0, i64 %indvars.iv
  store double %add, double* %arrayidx2, align 8, !tbaa !2
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 2
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

; Function Attrs: nounwind uwtable
define dso_local void @compute_reproj_error(double* noalias nocapture readonly %cam, double* noalias nocapture readonly %w, double* noalias nocapture readnone %feat, double* noalias nocapture %err) #2 {
entry:
  %proj = alloca [2 x double], align 16
  %0 = bitcast [2 x double]* %proj to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %0) #4
  %arraydecay = getelementptr inbounds [2 x double], [2 x double]* %proj, i64 0, i64 0
  call void @project(double* %cam, double* nonnull %arraydecay)
  %1 = load double, double* %w, align 8, !tbaa !2
  %2 = load double, double* %arraydecay, align 16, !tbaa !2
  %mul = fmul fast double %2, %1
  store double %mul, double* %err, align 8, !tbaa !2
  %arrayidx2 = getelementptr inbounds [2 x double], [2 x double]* %proj, i64 0, i64 1
  %3 = load double, double* %arrayidx2, align 8, !tbaa !2
  %mul3 = fmul fast double %3, %1
  %arrayidx4 = getelementptr inbounds double, double* %err, i64 1
  store double %mul3, double* %arrayidx4, align 8, !tbaa !2
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %0) #4
  ret void
}

; Function Attrs: nounwind uwtable
define dso_local void @dcompute_reproj_error(double* %cam, double* %dcam, double* %w, double* %wb, double* %feat, double* %err, double* %derr) local_unnamed_addr #2 {
entry:
  %0 = load i32, i32* @diffe_dup, align 4, !tbaa !6
  %1 = load i32, i32* @diffe_const, align 4, !tbaa !6
  %2 = load i32, i32* @diffe_dupnoneed, align 4, !tbaa !6
  tail call void (i8*, ...) @__enzyme_autodiff(i8* bitcast (void (double*, double*, double*, double*)* @compute_reproj_error to i8*), i32 %0, double* %cam, double* %dcam, i32 %0, double* %w, double* %wb, i32 %1, double* %feat, i32 %2, double* %err, double* %derr) #4
  ret void
}

declare dso_local void @__enzyme_autodiff(i8*, ...) local_unnamed_addr #3

attributes #0 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #3 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #4 = { nounwind }

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

; CHECK: define internal void @diffeproject(double* noalias nocapture readonly %cam, double* nocapture %"cam'", double* noalias nocapture %proj, double* nocapture %"proj'", { i8*, i8* } %tapeArg)
; CHECK-NOT: %Xcam = alloca [2 x double]
