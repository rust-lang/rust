; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -instsimplify -simplifycfg -S | FileCheck %s

; ModuleID = '/mnt/Data/git/Enzyme/enzyme/test/Integration/vecmax.c'
source_filename = "/mnt/Data/git/Enzyme/enzyme/test/Integration/vecmax.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }

@.str = private unnamed_addr constant [11 x i8] c"0 && \22bad\22\00", align 1
@.str.1 = private unnamed_addr constant [54 x i8] c"/mnt/Data/git/Enzyme/enzyme/test/Integration/vecmax.c\00", align 1
@__PRETTY_FUNCTION__.reduce_max = private unnamed_addr constant [33 x i8] c"double reduce_max(double *, int)\00", align 1
@.str.2 = private unnamed_addr constant [15 x i8] c"reduce_max=%f\0A\00", align 1
@.str.3 = private unnamed_addr constant [21 x i8] c"d_reduce_max(%i)=%f\0A\00", align 1
@.str.4 = private unnamed_addr constant [22 x i8] c"i=%d d_vec=%f ans=%f\0A\00", align 1
@stderr = external dso_local local_unnamed_addr global %struct._IO_FILE*, align 8
@.str.5 = private unnamed_addr constant [68 x i8] c"Assertion Failed: fabs( [%s = %g] - [%s = %g] ) > %g at %s:%d (%s)\0A\00", align 1
@.str.6 = private unnamed_addr constant [9 x i8] c"d_vec[i]\00", align 1
@.str.7 = private unnamed_addr constant [7 x i8] c"ans[i]\00", align 1
@__PRETTY_FUNCTION__.main = private unnamed_addr constant [11 x i8] c"int main()\00", align 1
@str = private unnamed_addr constant [5 x i8] c"done\00"

; Function Attrs: nounwind uwtable
define dso_local double @reduce_max(double* nocapture readonly %vec, i32 %size) #0 {
entry:
  %cmp15 = icmp sgt i32 %size, 0
  br i1 %cmp15, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %0 = sext i32 %size to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.inc, %entry
  %ret.0.lcssa = phi double [ 0xFFF0000000000000, %entry ], [ %ret.0., %for.inc ]
  ret double %ret.0.lcssa

for.body:                                         ; preds = %for.body.preheader, %for.inc
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.inc ]
  %ret.016 = phi double [ 0xFFF0000000000000, %for.body.preheader ], [ %ret.0., %for.inc ]
  %arrayidx = getelementptr inbounds double, double* %vec, i64 %indvars.iv
  %1 = load double, double* %arrayidx, align 8, !tbaa !2
  %cmp4 = icmp ugt i64 %indvars.iv, 5
  br i1 %cmp4, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  tail call void @__assert_fail(i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([54 x i8], [54 x i8]* @.str.1, i64 0, i64 0), i32 25, i8* getelementptr inbounds ([33 x i8], [33 x i8]* @__PRETTY_FUNCTION__.reduce_max, i64 0, i64 0)) #7
  unreachable

for.inc:                                          ; preds = %for.body
  %cmp1 = fcmp fast ogt double %ret.016, %1
  %ret.0. = select i1 %cmp1, double %ret.016, double %1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %cmp = icmp slt i64 %indvars.iv.next, %0
  br i1 %cmp, label %for.body, label %for.cond.cleanup
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

; Function Attrs: noreturn nounwind
declare dso_local void @__assert_fail(i8*, i8*, i32, i8*) local_unnamed_addr #2

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

; Function Attrs: nounwind uwtable
define dso_local i32 @main() local_unnamed_addr #0 {
entry:
  %vec = alloca [5 x double], align 16
  %d_vec = alloca [5 x double], align 16
  %ans = alloca [5 x double], align 16
  %0 = bitcast [5 x double]* %vec to i8*
  call void @llvm.lifetime.start.p0i8(i64 40, i8* nonnull %0) #6
  %1 = getelementptr inbounds [5 x double], [5 x double]* %vec, i64 0, i64 0
  store double -1.000000e+00, double* %1, align 16
  %2 = getelementptr inbounds [5 x double], [5 x double]* %vec, i64 0, i64 1
  store double 2.000000e+00, double* %2, align 8
  %3 = getelementptr inbounds [5 x double], [5 x double]* %vec, i64 0, i64 2
  store double -2.000000e-01, double* %3, align 16
  %4 = getelementptr inbounds [5 x double], [5 x double]* %vec, i64 0, i64 3
  store double 2.000000e+00, double* %4, align 8
  %5 = getelementptr inbounds [5 x double], [5 x double]* %vec, i64 0, i64 4
  store double 1.000000e+00, double* %5, align 16
  %6 = bitcast [5 x double]* %d_vec to i8*
  call void @llvm.lifetime.start.p0i8(i64 40, i8* nonnull %6) #6
  call void @llvm.memset.p0i8.i64(i8* nonnull align 16 %6, i8 0, i64 40, i1 false)
  %call = call fast double @reduce_max(double* nonnull %1, i32 5)
  %call1 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([15 x i8], [15 x i8]* @.str.2, i64 0, i64 0), double %call)
  %arraydecay3 = getelementptr inbounds [5 x double], [5 x double]* %d_vec, i64 0, i64 0
  call void @__enzyme_autodiff(i8* bitcast (double (double*, i32)* @reduce_max to i8*), double* nonnull %1, double* nonnull %arraydecay3, i32 5) #6
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  %7 = bitcast [5 x double]* %ans to i8*
  call void @llvm.lifetime.start.p0i8(i64 40, i8* nonnull %7) #6
  call void @llvm.memset.p0i8.i64(i8* nonnull align 16 %7, i8 0, i64 40, i1 false)
  %8 = getelementptr inbounds [5 x double], [5 x double]* %ans, i64 0, i64 3
  store double 1.000000e+00, double* %8, align 8
  br label %for.body9

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv49 = phi i64 [ 0, %entry ], [ %indvars.iv.next50, %for.body ]
  %arrayidx = getelementptr inbounds [5 x double], [5 x double]* %d_vec, i64 0, i64 %indvars.iv49
  %9 = load double, double* %arrayidx, align 8, !tbaa !2
  %10 = trunc i64 %indvars.iv49 to i32
  %call4 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([21 x i8], [21 x i8]* @.str.3, i64 0, i64 0), i32 %10, double %9)
  %indvars.iv.next50 = add nuw nsw i64 %indvars.iv49, 1
  %exitcond = icmp eq i64 %indvars.iv.next50, 5
  br i1 %exitcond, label %for.cond.cleanup, label %for.body

for.cond6:                                        ; preds = %for.body9
  %cmp7 = icmp ult i64 %indvars.iv.next, 5
  br i1 %cmp7, label %for.body9, label %for.cond.cleanup8

for.cond.cleanup8:                                ; preds = %for.cond6
  %puts = call i32 @puts(i8* getelementptr inbounds ([5 x i8], [5 x i8]* @str, i64 0, i64 0))
  call void @llvm.lifetime.end.p0i8(i64 40, i8* nonnull %7) #6
  call void @llvm.lifetime.end.p0i8(i64 40, i8* nonnull %6) #6
  call void @llvm.lifetime.end.p0i8(i64 40, i8* nonnull %0) #6
  ret i32 0

for.body9:                                        ; preds = %for.cond.cleanup, %for.cond6
  %indvars.iv = phi i64 [ 0, %for.cond.cleanup ], [ %indvars.iv.next, %for.cond6 ]
  %arrayidx11 = getelementptr inbounds [5 x double], [5 x double]* %d_vec, i64 0, i64 %indvars.iv
  %11 = load double, double* %arrayidx11, align 8, !tbaa !2
  %arrayidx13 = getelementptr inbounds [5 x double], [5 x double]* %ans, i64 0, i64 %indvars.iv
  %12 = load double, double* %arrayidx13, align 8, !tbaa !2
  %13 = trunc i64 %indvars.iv to i32
  %call14 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([22 x i8], [22 x i8]* @.str.4, i64 0, i64 0), i32 %13, double %11, double %12)
  %14 = load double, double* %arrayidx11, align 8, !tbaa !2
  %sub = fsub fast double %14, %12
  %15 = call fast double @llvm.fabs.f64(double %sub)
  %cmp19 = fcmp fast ogt double %15, 0x3E7AD7F29ABCAF48
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br i1 %cmp19, label %if.then, label %for.cond6

if.then:                                          ; preds = %for.body9
  %16 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !6
  %call24 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %16, i8* getelementptr inbounds ([68 x i8], [68 x i8]* @.str.5, i64 0, i64 0), i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str.6, i64 0, i64 0), double %14, i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.7, i64 0, i64 0), double %12, double 0x3E7AD7F29ABCAF48, i8* getelementptr inbounds ([54 x i8], [54 x i8]* @.str.1, i64 0, i64 0), i32 44, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @__PRETTY_FUNCTION__.main, i64 0, i64 0)) #8
  call void @abort() #7
  unreachable
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1) #1

; Function Attrs: nounwind
declare dso_local i32 @printf(i8* nocapture readonly, ...) local_unnamed_addr #3

declare dso_local void @__enzyme_autodiff(i8*, double*, double*, i32) local_unnamed_addr #4

; Function Attrs: nounwind readnone speculatable
declare double @llvm.fabs.f64(double) #5

; Function Attrs: nounwind
declare dso_local i32 @fprintf(%struct._IO_FILE* nocapture, i8* nocapture readonly, ...) local_unnamed_addr #3

; Function Attrs: noreturn nounwind
declare dso_local void @abort() local_unnamed_addr #2

; Function Attrs: nounwind
declare i32 @puts(i8* nocapture readonly) local_unnamed_addr #6

attributes #0 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { noreturn nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #3 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #4 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #5 = { nounwind readnone speculatable }
attributes #6 = { nounwind }
attributes #7 = { noreturn nounwind }
attributes #8 = { cold }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0}
!3 = !{!"double", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
!6 = !{!7, !7, i64 0}
!7 = !{!"any pointer", !4, i64 0}

; CHECK-NOT: call {{.*}} @realloc