; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -early-cse -simplifycfg -instsimplify -correlated-propagation -adce -S | FileCheck %s

; ModuleID = '../test/Integration/rwrloop.c'
source_filename = "../test/Integration/rwrloop.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.Gradients = type { double, double, double }

; Function Attrs: nounwind
declare %struct.Gradients @__enzyme_fwddiff(i8*, ...)

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

define void @main(double* %a, double* %da1, double* %da2, double* %da3, i32* %N) {
entry:
  %call = call %struct.Gradients (i8*, ...) @__enzyme_fwddiff(i8* bitcast (double (double*, i32*)* @alldiv to i8*), metadata !"enzyme_width", i64 3, double* nonnull %a, double* %da1, double* %da2, double* %da3, i32* nonnull %N)
  ret void
}

; Function Attrs: nounwind
declare i8* @llvm.stacksave() #3

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1) #1

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


; CHECK: define internal [3 x double] @fwddiffe3alldiv(double* noalias nocapture %a, [3 x double*] %"a'", i32* noalias nocapture %N)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = load i32, i32* %N, align 4, !tbaa !2
; CHECK-NEXT:   %cmp233 = icmp sgt i32 %0, 0
; CHECK-NEXT:   br label %for.cond1.preheader

; CHECK: for.cond1.preheader:                              ; preds = %for.cond.cleanup3, %entry
; CHECK-DAG:   %iv = phi i64 [ %iv.next, %for.cond.cleanup3 ], [ 0, %entry ]
; CHECK-DAG:   %[[sum036_0:.+]] = phi {{(fast )?}}double [ 0.000000e+00, %entry ], [ %[[sumlcssa_0:.+]], %for.cond.cleanup3 ]
; CHECK-DAG:   %[[sum036_1:.+]] = phi {{(fast )?}}double [ 0.000000e+00, %entry ], [ %[[sumlcssa_1:.+]], %for.cond.cleanup3 ]
; CHECK-DAG:   %[[sum036_2:.+]] = phi {{(fast )?}}double [ 0.000000e+00, %entry ], [ %[[sumlcssa_2:.+]], %for.cond.cleanup3 ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   br i1 %cmp233, label %for.body4.lr.ph, label %for.cond.cleanup3

; CHECK: for.body4.lr.ph:                                  ; preds = %for.cond1.preheader
; CHECK-NEXT:   %[[i1:.+]] = mul nuw nsw i64 %iv, 10
; CHECK-NEXT:   %[[i2:.+]] = load i32, i32* %N, align 4, !tbaa !2
; CHECK-NEXT:   %[[i3:.+]] = sext i32 %[[i2]] to i64
; CHECK-NEXT:   br label %for.body4

; CHECK: for.body4:                                        ; preds = %for.body4, %for.body4.lr.ph
; CHECK-DAG:   %iv1 = phi i64 [ %iv.next2, %for.body4 ], [ 0, %for.body4.lr.ph ]
; CHECK-DAG:   %[[sum134_0:.+]] = phi {{(fast )?}}double [ %[[sum036_0]], %for.body4.lr.ph ], [ %[[i26_0:.+]], %for.body4 ]
; CHECK-DAG:   %[[sum134_1:.+]] = phi {{(fast )?}}double [ %[[sum036_1]], %for.body4.lr.ph ], [ %[[i26_1:.+]], %for.body4 ]
; CHECK-DAG:   %[[sum134_2:.+]] = phi {{(fast )?}}double [ %[[sum036_2]], %for.body4.lr.ph ], [ %[[i26_2:.+]], %for.body4 ]
; CHECK-NEXT:   %iv.next2 = add nuw nsw i64 %iv1, 1
; CHECK-NEXT:   %[[i4:.+]] = add nuw nsw i64 %iv1, %[[i1]]
; CHECK-NEXT:   %[[i5:.+]] = extractvalue [3 x double*] %"a'", 0
; CHECK-NEXT:   %"arrayidx'ipg" = getelementptr inbounds double, double* %[[i5]], i64 %[[i4]]
; CHECK-NEXT:   %[[i6:.+]] = extractvalue [3 x double*] %"a'", 1
; CHECK-NEXT:   %"arrayidx'ipg1" = getelementptr inbounds double, double* %[[i6]], i64 %[[i4]]
; CHECK-NEXT:   %[[i7:.+]] = extractvalue [3 x double*] %"a'", 2
; CHECK-NEXT:   %"arrayidx'ipg2" = getelementptr inbounds double, double* %[[i7]], i64 %[[i4]]
; CHECK-NEXT:   %arrayidx = getelementptr inbounds double, double* %a, i64 %[[i4]]
; CHECK-NEXT:   %[[i9:.+]] = load double, double* %"arrayidx'ipg", align 8
; CHECK-NEXT:   %[[i10:.+]] = load double, double* %"arrayidx'ipg1", align 8
; CHECK-NEXT:   %[[i11:.+]] = load double, double* %"arrayidx'ipg2", align 8
; CHECK-NEXT:   %[[i8:.+]] = load double, double* %arrayidx, align 8, !tbaa !6
; CHECK-NEXT:   %[[i12:.+]] = fmul fast double %[[i9]], %[[i8]]
; CHECK-NEXT:   %[[i13:.+]] = fadd fast double %[[i12]], %[[i12]]
; CHECK-NEXT:   %[[i14:.+]] = fmul fast double %[[i10]], %[[i8]]
; CHECK-NEXT:   %[[i15:.+]] = fadd fast double %[[i14]], %[[i14]]
; CHECK-NEXT:   %[[i16:.+]] = fmul fast double %[[i11]], %[[i8]]
; CHECK-NEXT:   %[[i17:.+]] = fadd fast double %[[i16]], %[[i16]]
; CHECK-NEXT:   %[[i26_0:.+]] = fadd fast double %[[sum134_0]], %[[i13]]
; CHECK-NEXT:   %[[i26_1:.+]] = fadd fast double %[[sum134_1]], %[[i15]]
; CHECK-NEXT:   %[[i26_2:.+]] = fadd fast double %[[sum134_2]], %[[i17]]
; CHECK-NEXT:   store double 0.000000e+00, double* %arrayidx, align 8, !tbaa !6
; CHECK-NEXT:   store double 0.000000e+00, double* %"arrayidx'ipg", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"arrayidx'ipg1", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"arrayidx'ipg2", align 8
; CHECK-NEXT:   %cmp2 = icmp slt i64 %iv.next2, %[[i3]]
; CHECK-NEXT:   br i1 %cmp2, label %for.body4, label %for.cond.cleanup3

; CHECK: for.cond.cleanup3:                                ; preds = %for.body4, %for.cond1.preheader
; CHECK-NEXT:   %[[sumlcssa_0]] = phi {{(fast )?}}double [ %[[sum036_0]], %for.cond1.preheader ], [ %[[i26_0]], %for.body4 ]
; CHECK-NEXT:   %[[sumlcssa_1]] = phi {{(fast )?}}double [ %[[sum036_1]], %for.cond1.preheader ], [ %[[i26_1]], %for.body4 ]
; CHECK-NEXT:   %[[sumlcssa_2]] = phi {{(fast )?}}double [ %[[sum036_2]], %for.cond1.preheader ], [ %[[i26_2]], %for.body4 ]
; CHECK-NEXT:   %[[i27:.+]] = insertvalue [3 x double] undef, double %[[sumlcssa_0]], 0
; CHECK-NEXT:   %[[i28:.+]] = insertvalue [3 x double] %[[i27]], double %[[sumlcssa_1]], 1
; CHECK-NEXT:   %[[i29:.+]] = insertvalue [3 x double] %[[i28]], double %[[sumlcssa_2]], 2
; CHECK-NEXT:   %exitcond = icmp eq i64 %iv.next, 10
; CHECK-NEXT:   br i1 %exitcond, label %for.cond.cleanup, label %for.cond1.preheader

; CHECK: for.cond.cleanup:                                 ; preds = %for.cond.cleanup3
; CHECK-NEXT:   store i32 7, i32* %N, align 4, !tbaa !2
; CHECK-NEXT:   ret [3 x double] %[[i29]]
; CHECK-NEXT: }
