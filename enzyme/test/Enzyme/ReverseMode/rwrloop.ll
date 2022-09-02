; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -early-cse -simplifycfg -instsimplify -correlated-propagation -instsimplify -adce -S | FileCheck %s

; ModuleID = '../test/Integration/rwrloop.c'
source_filename = "../test/Integration/rwrloop.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }

@.str = private unnamed_addr constant [16 x i8] c"d_a[%d][%d]=%f\0A\00", align 1
@stderr = external dso_local local_unnamed_addr global %struct._IO_FILE*, align 8
@.str.1 = private unnamed_addr constant [68 x i8] c"Assertion Failed: fabs( [%s = %g] - [%s = %g] ) > %g at %s:%d (%s)\0A\00", align 1
@.str.2 = private unnamed_addr constant [10 x i8] c"d_a[i][j]\00", align 1
@.str.3 = private unnamed_addr constant [15 x i8] c"2. * (i*100+j)\00", align 1
@.str.4 = private unnamed_addr constant [30 x i8] c"../test/Integration/rwrloop.c\00", align 1
@__PRETTY_FUNCTION__.main = private unnamed_addr constant [23 x i8] c"int main(int, char **)\00", align 1

; Function Attrs: norecurse nounwind uwtable
define dso_local double @alldiv(double* noalias nocapture %a, i32* noalias nocapture %N) #0 {
entry:
  %0 = load i32, i32* %N, align 4, !tbaa !2
  %cmp233 = icmp sgt i32 %0, 0
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.cond.cleanup3, %entry
  %indvar = phi i64 [ 0, %entry ], [ %indvar.next, %for.cond.cleanup3 ]
  %sum.036 = phi double [ 0.000000e+00, %entry ], [ %sum.1.lcssa, %for.cond.cleanup3 ]
  br i1 %cmp233, label %for.body4.lr.ph, label %for.cond.cleanup3

for.body4.lr.ph:                                  ; preds = %for.cond1.preheader
  %1 = mul nuw nsw i64 %indvar, 10
  %2 = load i32, i32* %N, align 4, !tbaa !2
  %3 = sext i32 %2 to i64
  br label %for.body4

for.body4:                                        ; preds = %for.body4.lr.ph, %for.body4
  %indvars.iv = phi i64 [ 0, %for.body4.lr.ph ], [ %indvars.iv.next, %for.body4 ]
  %sum.134 = phi double [ %sum.036, %for.body4.lr.ph ], [ %add10, %for.body4 ]
  %4 = add nuw nsw i64 %indvars.iv, %1
  %arrayidx = getelementptr inbounds double, double* %a, i64 %4
  %5 = load double, double* %arrayidx, align 8, !tbaa !6
  %mul9 = fmul double %5, %5
  %add10 = fadd double %sum.134, %mul9
  store double 0.000000e+00, double* %arrayidx, align 8, !tbaa !6
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %cmp2 = icmp slt i64 %indvars.iv.next, %3
  br i1 %cmp2, label %for.body4, label %for.cond.cleanup3

for.cond.cleanup3:                                ; preds = %for.body4, %for.cond1.preheader
  %sum.1.lcssa = phi double [ %sum.036, %for.cond1.preheader ], [ %add10, %for.body4 ]
  %indvar.next = add nuw nsw i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar.next, 10
  br i1 %exitcond, label %for.cond.cleanup, label %for.cond1.preheader

for.cond.cleanup:                                 ; preds = %for.cond.cleanup3
  store i32 7, i32* %N, align 4, !tbaa !2
  ret double %sum.1.lcssa
}

define void @main(double* %a, double* %da, i32* %N) {
entry:
  %call = call double (i8*, ...) @__enzyme_autodiff(i8* bitcast (double (double*, i32*)* @alldiv to i8*), double* nonnull %a, double* nonnull %da, i32* nonnull %N)
  ret void
}

; Function Attrs: nounwind
declare i8* @llvm.stacksave() #3

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1) #1

declare dso_local double @__enzyme_autodiff(i8*, ...) local_unnamed_addr #4

; Function Attrs: nounwind
declare dso_local i32 @printf(i8* nocapture readonly, ...) local_unnamed_addr #5

; Function Attrs: nounwind
declare dso_local i32 @fflush(%struct._IO_FILE* nocapture) local_unnamed_addr #5

; Function Attrs: nounwind readnone speculatable
declare double @llvm.fabs.f64(double) #6

; Function Attrs: nounwind
declare dso_local i32 @fprintf(%struct._IO_FILE* nocapture, i8* nocapture readonly, ...) local_unnamed_addr #5

; Function Attrs: noreturn nounwind
declare dso_local void @abort() local_unnamed_addr #7

; Function Attrs: nounwind
declare void @llvm.stackrestore(i8*) #3

attributes #0 = { norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind }
attributes #4 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { nounwind readnone speculatable }
attributes #7 = { noreturn nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #8 = { cold }
attributes #9 = { noreturn nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.0.0 (trunk 336729)"}
!2 = !{!3, !3, i64 0}
!3 = !{!"int", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
!6 = !{!7, !7, i64 0}
!7 = !{!"double", !4, i64 0}
!8 = !{!9, !9, i64 0}
!9 = !{!"any pointer", !4, i64 0}

; CHECK: define internal void @diffealldiv(double* noalias nocapture %a, double* nocapture %"a'", i32* noalias nocapture %N, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[a0:.+]] = load i32, i32* %N, align 4, !tbaa !2
; CHECK-NEXT:   %cmp233 = icmp sgt i32 %[[a0]], 0
; CHECK-NEXT:   %[[_unwrap5:.+]] = sext i32 %[[a0]] to i64
; TODO-CHECK-NEXT:   %_unwrap6 = icmp sgt i64 %[[_unwrap5]], 1
; TODO-CHECK-NEXT:   %[[smax_unwrap:.+]] = select i1 %_unwrap6, i64 %[[_unwrap5]], i64 1
; CHECK:   %[[a1:.+]] = mul nuw nsw i64 %[[smax_unwrap:.+]], 10
; CHECK-NEXT:   %mallocsize = mul nuw nsw i64 %[[a1]], 8
; CHECK-NEXT:   %malloccall = tail call noalias nonnull i8* @malloc(i64 %mallocsize)
; CHECK-NEXT:   %_malloccache = bitcast i8* %malloccall to double*
; CHECK-NEXT:   %[[malloccall11:.+]] = tail call noalias nonnull dereferenceable(40) dereferenceable_or_null(40) i8* @malloc(i64 40)
; CHECK-NEXT:   %[[malloccache12:.+]] = bitcast i8* %[[malloccall11:.+]] to i32*
; CHECK-NEXT:   br label %for.cond1.preheader

; CHECK: for.cond1.preheader:                              ; preds = %for.cond.cleanup3, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %for.cond.cleanup3 ], [ 0, %entry ]
; CHECK-NEXT:   %[[a2:.+]] = mul {{(nuw nsw )?}}i64 %iv, 10
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   br i1 %cmp233, label %for.body4.lr.ph, label %for.cond.cleanup3

; CHECK: for.body4.lr.ph:                                  ; preds = %for.cond1.preheader
; CHECK-NEXT:   %[[a3:.+]] = load i32, i32* %N, align 4, !tbaa !2
; CHECK-NEXT:   %[[a4:.+]] = getelementptr inbounds i32, i32* %[[malloccache12]], i64 %iv
; CHECK-NEXT:   store i32 %[[a3]], i32* %[[a4]], align 4, !tbaa !2, !invariant.group !8
; CHECK-NEXT:   %[[a5:.+]] = sext i32 %[[a3]] to i64
; CHECK-NEXT:   br label %for.body4

; CHECK: for.body4:                                        ; preds = %for.body4, %for.body4.lr.ph
; CHECK-NEXT:   %iv1 = phi i64 [ %iv.next2, %for.body4 ], [ 0, %for.body4.lr.ph ]
; CHECK-NEXT:   %iv.next2 = add nuw nsw i64 %iv1, 1
; CHECK-NEXT:   %[[a6:.+]] = add nuw nsw i64 %iv1, %[[a2]]
; CHECK-NEXT:   %arrayidx = getelementptr inbounds double, double* %a, i64 %[[a6]]
; CHECK-NEXT:   %[[a7:.+]] = load double, double* %arrayidx, align 8, !tbaa !6
; CHECK-NEXT:   store double 0.000000e+00, double* %arrayidx, align 8, !tbaa !6
; CHECK-NEXT:   %[[a8:.+]] = mul nuw nsw i64 %iv, %[[smax_unwrap]]
; CHECK-NEXT:   %[[a9:.+]] = add nuw nsw i64 %iv1, %[[a8]]
; CHECK-NEXT:   %[[a10:.+]] = getelementptr inbounds double, double* %_malloccache, i64 %[[a9]]
; CHECK-NEXT:   store double %[[a7]], double* %[[a10]], align 8, !tbaa !6, !invariant.group ![[g9:[0-9]+]]
; CHECK-NEXT:   %cmp2 = icmp slt i64 %iv.next2, %[[a5]]
; CHECK-NEXT:   br i1 %cmp2, label %for.body4, label %for.cond.cleanup3

; CHECK: for.cond.cleanup3:                                ; preds = %for.body4, %for.cond1.preheader
; CHECK-NEXT:   %exitcond = icmp eq i64 %iv.next, 10
; CHECK-NEXT:   br i1 %exitcond, label %for.cond.cleanup, label %for.cond1.preheader

; CHECK: for.cond.cleanup:                                 ; preds = %for.cond.cleanup3
; CHECK-NEXT:   store i32 7, i32* %N, align 4, !tbaa !2
; CHECK-NEXT:   br label %invertfor.cond.cleanup3

; CHECK: invertentry:                                      ; preds = %invertfor.cond1.preheader
; CHECK-NEXT:   tail call void @free(i8* nonnull %malloccall)
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[malloccall11]])
; CHECK-NEXT:   ret void

; CHECK: invertfor.cond1.preheader:                        ; preds = %invertfor.body4, %invertfor.cond.cleanup3
; CHECK-NEXT:   %"'de.0" = phi double [ %"'de.2", %invertfor.cond.cleanup3 ], [ 0.000000e+00, %invertfor.body4 ]
; CHECK-NEXT:   %"mul9'de.0" = phi double [ %"mul9'de.2", %invertfor.cond.cleanup3 ], [ 0.000000e+00, %invertfor.body4 ]
; CHECK-NEXT:   %"sum.134'de.0" = phi double [ %"sum.134'de.2", %invertfor.cond.cleanup3 ], [ 0.000000e+00, %invertfor.body4 ]
; CHECK-NEXT:   %"add10'de.0" = phi double [ %"add10'de.2", %invertfor.cond.cleanup3 ], [ 0.000000e+00, %invertfor.body4 ]
; CHECK-NEXT:   %"sum.036'de.0" = phi double [ %"sum.1.lcssa'de.0", %invertfor.cond.cleanup3 ], [ %[[a23:.+]], %invertfor.body4 ]
; CHECK-NEXT:   %[[a11:.+]] = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %[[a11]], label %invertentry, label %incinvertfor.cond1.preheader

; CHECK: incinvertfor.cond1.preheader:                     ; preds = %invertfor.cond1.preheader
; CHECK-NEXT:   %[[a12:.+]] = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertfor.cond.cleanup3

; CHECK: invertfor.body4:                                  ; preds = %invertfor.cond.cleanup3.loopexit, %incinvertfor.body4
; CHECK-NEXT:   %"'de.1" = phi double [ %"'de.2", %invertfor.cond.cleanup3.loopexit ], [ 0.000000e+00, %incinvertfor.body4 ]
; CHECK-NEXT:   %"mul9'de.1" = phi double [ %"mul9'de.2", %invertfor.cond.cleanup3.loopexit ], [ 0.000000e+00, %incinvertfor.body4 ]
; CHECK-NEXT:   %"sum.134'de.1" = phi double [ %"sum.134'de.2", %invertfor.cond.cleanup3.loopexit ], [ 0.000000e+00, %incinvertfor.body4 ]
; CHECK-NEXT:   %"add10'de.1" = phi double [ %[[a27:.+]], %invertfor.cond.cleanup3.loopexit ], [ %[[a13:.+]], %incinvertfor.body4 ]
; CHECK-NEXT:   %"iv1'ac.1" = phi i64 [ %[[_unwrap20:.+]], %invertfor.cond.cleanup3.loopexit ], [ %[[a24:.+]], %incinvertfor.body4 ]
; CHECK-NEXT:   %_unwrap = mul nuw nsw i64 %"iv'ac.0", 10
; CHECK-NEXT:   %[[_unwrap3:.+]] = add nuw nsw i64 %"iv1'ac.1", %_unwrap
; CHECK-NEXT:   %"arrayidx'ipg_unwrap" = getelementptr inbounds double, double* %"a'", i64 %[[_unwrap3]]
; CHECK-NEXT:   store double 0.000000e+00, double* %"arrayidx'ipg_unwrap", align 8
; CHECK-NEXT:   %[[a13]] = fadd fast double %"sum.134'de.1", %"add10'de.1"
; CHECK-NEXT:   %[[a14:.+]] = fadd fast double %"mul9'de.1", %"add10'de.1"
; CHECK-NEXT:   %[[a16:.+]] = mul nuw nsw i64 %"iv'ac.0", %[[smax_unwrap:.+]]
; CHECK-NEXT:   %[[a17:.+]] = add nuw nsw i64 %"iv1'ac.1", %[[a16]]
; CHECK-NEXT:   %[[a18:.+]] = getelementptr inbounds double, double* %_malloccache, i64 %[[a17]]
; CHECK-NEXT:   %[[a19:.+]] = load double, double* %[[a18]], align 8, !tbaa !6, !invariant.group ![[g9]]
; CHECK-NEXT:   %m0diffe = fmul fast double %[[a14]], %[[a19]]
; CHECK-NEXT:   %[[a20:.+]] = fadd fast double %"'de.1", %m0diffe
; CHECK-NEXT:   %[[a21:.+]] = fadd fast double %[[a20]], %m0diffe
; CHECK-NEXT:   store double %[[a21]], double* %"arrayidx'ipg_unwrap", align 8
; CHECK-NEXT:   %[[a22:.+]] = icmp eq i64 %"iv1'ac.1", 0
; CHECK-NEXT:   br i1 %[[a22]], label %invertfor.cond1.preheader, label %incinvertfor.body4

; CHECK: incinvertfor.body4:                               ; preds = %invertfor.body4
; CHECK-NEXT:   %[[a24]] = add nsw i64 %"iv1'ac.1", -1
; CHECK-NEXT:   br label %invertfor.body4

; CHECK: invertfor.cond.cleanup3.loopexit:                 ; preds = %invertfor.cond.cleanup3
; CHECK-NEXT:   %[[a25:.+]] = getelementptr inbounds i32, i32* %[[malloccache12]], i64 %"iv'ac.0"
; CHECK-NEXT:   %[[a26:.+]] = load i32, i32* %[[a25]], align 4, !tbaa !2, !invariant.group !8
; CHECK-NEXT:   %[[_unwrap17:.+]] = sext i32 %[[a26]] to i64
; TODO-CHECK-NEXT:   %[[_unwrap14:.+]] = icmp sgt i64 %[[_unwrap17]], 1
; TODO-CHECK-NEXT:   %[[smax_unwrap19:.+]] = select i1 %[[_unwrap14]], i64 %[[_unwrap17]], i64 1
; CHECK:   %[[_unwrap20]] = add{{( nsw)?}} i64 %[[smax_unwrap19:.+]], -1
; CHECK-NEXT:   br label %invertfor.body4

; CHECK: invertfor.cond.cleanup3:                          ; preds = %for.cond.cleanup, %incinvertfor.cond1.preheader
; CHECK-NEXT:   %"'de.2" = phi double [ 0.000000e+00, %for.cond.cleanup ], [ %"'de.0", %incinvertfor.cond1.preheader ]
; CHECK-NEXT:   %"mul9'de.2" = phi double [ 0.000000e+00, %for.cond.cleanup ], [ %"mul9'de.0", %incinvertfor.cond1.preheader ]
; CHECK-NEXT:   %"sum.134'de.2" = phi double [ 0.000000e+00, %for.cond.cleanup ], [ %"sum.134'de.0", %incinvertfor.cond1.preheader ]
; CHECK-NEXT:   %"add10'de.2" = phi double [ 0.000000e+00, %for.cond.cleanup ], [ %"add10'de.0", %incinvertfor.cond1.preheader ]
; CHECK-NEXT:   %"sum.1.lcssa'de.0" = phi double [ %differeturn, %for.cond.cleanup ], [ %"sum.036'de.0", %incinvertfor.cond1.preheader ]
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ 9, %for.cond.cleanup ], [ %[[a12]], %incinvertfor.cond1.preheader ]
; CHECK-NEXT:   %[[a27]] = fadd fast double %"add10'de.2", %"sum.1.lcssa'de.0"
; CHECK-NEXT:   br i1 %cmp233, label %invertfor.cond.cleanup3.loopexit, label %invertfor.cond1.preheader
