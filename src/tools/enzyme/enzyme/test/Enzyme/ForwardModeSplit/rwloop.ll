; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -early-cse -simplifycfg -instsimplify -correlated-propagation -adce -S | FileCheck %s

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
  %call = call double (i8*, ...) @__enzyme_fwdsplit(i8* bitcast (double (double*, i32*)* @alldiv to i8*), metadata !"enzyme_nofree", double* nonnull %a, double* nonnull %da, i32* nonnull %N, i8* null)
  ret void
}

; Function Attrs: nounwind
declare i8* @llvm.stacksave() #3

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1) #1

declare dso_local double @__enzyme_fwdsplit(i8*, ...) local_unnamed_addr #4

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


; CHECK: define internal double @fwddiffealldiv(double* noalias nocapture %a, double* nocapture %"a'", i32* noalias nocapture %N, i8* %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = bitcast i8* %tapeArg to { i1, i32*, double** }*
; CHECK-NEXT:   %truetape = load { i1, i32*, double** }, { i1, i32*, double** }* %0
; CHECK-DAG:   %[[i1:.+]] = extractvalue { i1, i32*, double** } %truetape, 1
; CHECK-DAG:   %[[i2:.+]] = extractvalue { i1, i32*, double** } %truetape, 2
; CHECK-DAG:   %cmp233 = extractvalue { i1, i32*, double** } %truetape, 0
; CHECK-NEXT:   br label %for.cond1.preheader

; CHECK: for.cond1.preheader:                              ; preds = %for.cond.cleanup3, %entry
; CHECK-DAG:   %iv = phi i64 [ %iv.next, %for.cond.cleanup3 ], [ 0, %entry ]
; CHECK-DAG:   %[[sum036:.+]] = phi {{(fast )?}}double [ 0.000000e+00, %entry ], [ %[[sumlcssa:.+]], %for.cond.cleanup3 ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   br i1 %cmp233, label %for.body4.lr.ph, label %for.cond.cleanup3

; CHECK: for.body4.lr.ph:                                  ; preds = %for.cond1.preheader
; CHECK-NEXT:   %[[i3:.+]] = mul nuw nsw i64 %iv, 10
; CHECK-NEXT:   %[[i4:.+]] = getelementptr inbounds i32, i32* %[[i1]], i64 %iv
; CHECK-NEXT:   %[[i5:.+]] = load i32, i32* %[[i4]], align 4, !invariant.group !
; CHECK-NEXT:   %[[i6:.+]] = sext i32 %[[i5]] to i64
; CHECK-NEXT:   br label %for.body4

; CHECK: for.body4:                                        ; preds = %for.body4, %for.body4.lr.ph
; CHECK-DAG:   %iv1 = phi i64 [ %iv.next2, %for.body4 ], [ 0, %for.body4.lr.ph ]
; CHECK-DAG:   %[[sum134:.+]] = phi {{(fast )?}}double [ %[[sum036]], %for.body4.lr.ph ], [ %[[i15:.+]], %for.body4 ]
; CHECK-NEXT:   %iv.next2 = add nuw nsw i64 %iv1, 1
; CHECK-NEXT:   %[[i7:.+]] = add nuw nsw i64 %iv1, %[[i3]]
; CHECK-NEXT:   %[[arrayidxipg:.+]] = getelementptr inbounds double, double* %"a'", i64 %[[i7]]
; CHECK-NEXT:   %[[i12:.+]] = load double, double* %[[arrayidxipg]], align 8
; CHECK-NEXT:   %[[i8:.+]] = getelementptr inbounds double*, double** %[[i2]], i64 %iv
; CHECK-NEXT:   %[[i9:.+]] = load double*, double** %[[i8]], align 8, !dereferenceable !{{[0-9]+}}, !invariant.group !
; CHECK-NEXT:   %[[i10:.+]] = getelementptr inbounds double, double* %[[i9]], i64 %iv1
; CHECK-NEXT:   %[[i11:.+]] = load double, double* %[[i10]], align 8, !invariant.group !
; CHECK-NEXT:   %[[i13:.+]] = fmul fast double %[[i12]], %[[i11]]
; CHECK-NEXT:   %[[i14:.+]] = fadd fast double %[[i13]], %[[i13]]
; CHECK-NEXT:   %[[i15]] = fadd fast double %[[sum134]], %[[i14]]
; CHECK-NEXT:   store double 0.000000e+00, double* %[[arrayidxipg]], align 8
; CHECK-NEXT:   %cmp2 = icmp slt i64 %iv.next2, %[[i6]]
; CHECK-NEXT:   br i1 %cmp2, label %for.body4, label %for.cond.cleanup3

; CHECK: for.cond.cleanup3:                                ; preds = %for.body4, %for.cond1.preheader
; CHECK-NEXT:   %[[sumlcssa]] = phi {{(fast )?}}double [ %[[sum036:.+]], %for.cond1.preheader ], [ %[[i15]], %for.body4 ]
; CHECK-NEXT:   %exitcond = icmp eq i64 %iv.next, 10
; CHECK-NEXT:   br i1 %exitcond, label %for.cond.cleanup, label %for.cond1.preheader

; CHECK: for.cond.cleanup:                                 ; preds = %for.cond.cleanup3
; CHECK-NEXT:   ret double %[[sumlcssa]]
; CHECK-NEXT: }
