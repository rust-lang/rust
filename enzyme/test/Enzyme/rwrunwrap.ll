; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -early-cse -simplifycfg -S | FileCheck %s

source_filename = "/mnt/Data/git/Enzyme/enzyme/test/Integration/readwriteread.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }

@.str = private unnamed_addr constant [20 x i8] c"dx is %f ret is %f\0A\00", align 1
@stderr = external dso_local local_unnamed_addr global %struct._IO_FILE*, align 8
@.str.1 = private unnamed_addr constant [68 x i8] c"Assertion Failed: fabs( [%s = %g] - [%s = %g] ) > %g at %s:%d (%s)\0A\00", align 1
@.str.2 = private unnamed_addr constant [4 x i8] c"*dx\00", align 1
@.str.3 = private unnamed_addr constant [10 x i8] c"3*2.0*2.0\00", align 1
@.str.4 = private unnamed_addr constant [61 x i8] c"/mnt/Data/git/Enzyme/enzyme/test/Integration/readwriteread.c\00", align 1
@__PRETTY_FUNCTION__.main = private unnamed_addr constant [23 x i8] c"int main(int, char **)\00", align 1

; Function Attrs: norecurse nounwind uwtable
define dso_local void @readwriteread(double* noalias nocapture %x, double* noalias nocapture %ret) #2 {
entry:
  %0 = load double, double* %x, align 8, !tbaa !2
  %mul.i.i = fmul double %0, %0
  %mul.i6.i = fmul double %0, %mul.i.i
  store double %mul.i6.i, double* %x, align 8, !tbaa !2
  store double %mul.i6.i, double* %ret, align 8, !tbaa !2
  ret void
}

; Function Attrs: nounwind uwtable
define dso_local i32 @main(i32 %argc, i8** nocapture readnone %argv) local_unnamed_addr #3 {
entry:
  %ret = alloca double, align 8
  %dret = alloca double, align 8
  %0 = bitcast double* %ret to i8*
  store double 0.000000e+00, double* %ret, align 8, !tbaa !2
  %1 = bitcast double* %dret to i8*
  store double 1.000000e+00, double* %dret, align 8, !tbaa !2
  %call = tail call noalias i8* @malloc(i64 8) #8
  %2 = bitcast i8* %call to double*
  %call1 = tail call noalias i8* @malloc(i64 8) #8
  %3 = bitcast i8* %call1 to double*
  store double 2.000000e+00, double* %2, align 8, !tbaa !2
  store double 0.000000e+00, double* %3, align 8, !tbaa !2
  %call2 = call double (i8*, ...) @__enzyme_autodiff(i8* bitcast (void (double*, double*)* @readwriteread to i8*), i8* %call, i8* %call1, double* nonnull %ret, double* nonnull %dret) #8
  %4 = load double, double* %3, align 8, !tbaa !2
  %5 = load double, double* %ret, align 8, !tbaa !2
  %call3 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([20 x i8], [20 x i8]* @.str, i64 0, i64 0), double %4, double %5)
  %6 = load double, double* %3, align 8, !tbaa !2
  %sub = fadd double %6, -1.200000e+01
  %7 = call double @llvm.fabs.f64(double %sub)
  %cmp = fcmp ogt double %7, 1.000000e-10
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %8 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !6
  %call4 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %8, i8* getelementptr inbounds ([68 x i8], [68 x i8]* @.str.1, i64 0, i64 0), i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.2, i64 0, i64 0), double %6, i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str.3, i64 0, i64 0), double 1.200000e+01, double 1.000000e-10, i8* getelementptr inbounds ([61 x i8], [61 x i8]* @.str.4, i64 0, i64 0), i32 55, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @__PRETTY_FUNCTION__.main, i64 0, i64 0)) #9
  call void @abort() #10
  unreachable

if.end:                                           ; preds = %entry
  ret i32 0
}

; Function Attrs: nounwind
declare dso_local noalias i8* @malloc(i64) local_unnamed_addr #4

declare dso_local double @__enzyme_autodiff(i8*, ...) local_unnamed_addr #5

; Function Attrs: nounwind
declare dso_local i32 @printf(i8* nocapture readonly, ...) local_unnamed_addr #4

; Function Attrs: nounwind readnone speculatable
declare double @llvm.fabs.f64(double) #6

; Function Attrs: nounwind
declare dso_local i32 @fprintf(%struct._IO_FILE* nocapture, i8* nocapture readonly, ...) local_unnamed_addr #4

; Function Attrs: noreturn nounwind
declare dso_local void @abort() local_unnamed_addr #7

attributes #0 = { norecurse nounwind readonly uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { nounwind readnone speculatable }
attributes #7 = { noreturn nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #8 = { nounwind }
attributes #9 = { cold }
attributes #10 = { noreturn nounwind }

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

; CHECK: define internal void @differeadwriteread(double* noalias nocapture %x, double* nocapture %"x'", double* noalias nocapture %ret, double* nocapture %"ret'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = load double, double* %x, align 8, !tbaa !2
; CHECK-NEXT:   %mul.i.i = fmul double %0, %0
; CHECK-NEXT:   %mul.i6.i = fmul double %0, %mul.i.i
; CHECK-NEXT:   store double %mul.i6.i, double* %x, align 8, !tbaa !2
; CHECK-NEXT:   store double %mul.i6.i, double* %ret, align 8, !tbaa !2
; CHECK-NEXT:   %1 = load double, double* %"ret'", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"ret'", align 8
; CHECK-NEXT:   %2 = load double, double* %"x'", align 8
; CHECK-NEXT:   %3 = fadd fast double %1, %2
; CHECK-NEXT:   %m0diffe = fmul fast double %3, %mul.i.i
; CHECK-NEXT:   %m1diffemul.i.i = fmul fast double %3, %0
; CHECK-NEXT:   %m0diffe1 = fmul fast double %m1diffemul.i.i, %0
; CHECK-NEXT:   %4 = fadd fast double %m0diffe, %m0diffe1
; CHECK-NEXT:   %5 = fadd fast double %4, %m0diffe1
; CHECK-NEXT:   store double %5, double* %"x'", align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
