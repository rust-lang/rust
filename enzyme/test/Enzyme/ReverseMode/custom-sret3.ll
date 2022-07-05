; RUN: if [ %llvmver -ge 12 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi

; ModuleID = '/host/myblas.c'
source_filename = "/host/myblas.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

%struct.complex = type { double, double }
%struct.TapeAndComplex = type { i8*, %struct.complex }

@.str = private unnamed_addr constant [8 x i8] c"byref_6\00", align 1
@__enzyme_register_gradient1 = dso_local local_unnamed_addr global [4 x i8*] [i8* bitcast ({ double, double } (%struct.complex*, %struct.complex*, i32)* @myblas_cdot to i8*), i8* bitcast (void (%struct.TapeAndComplex*, %struct.complex*, %struct.complex*, %struct.complex*, %struct.complex*, i32, i32)* @myblas_cdot_fwd to i8*), i8* bitcast (void (%struct.complex*, %struct.complex*, %struct.complex*, %struct.complex*, i32, i32, %struct.complex*, i8*)* @myblas_cdot_rev to i8*), i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str, i32 0, i32 0)], align 16
@__enzyme_register_gradient2 = dso_local local_unnamed_addr global [3 x i8*] [i8* bitcast (double (double, double)* @myblas_cabs to i8*), i8* bitcast ({ i8*, double } (double, double)* @myblas_cabs_fwd to i8*), i8* bitcast ({ double, double } (double, double, double, i8*)* @myblas_cabs_rev to i8*)], align 16
@enzyme_byref = dso_local local_unnamed_addr global i32 0, align 4

; Function Attrs: norecurse nounwind readonly uwtable
define dso_local { double, double } @myblas_cdot(%struct.complex* nocapture readonly %0, %struct.complex* nocapture readonly %1, i32 %2) #0 {
  %4 = icmp sgt i32 %2, 0
  br i1 %4, label %5, label %7

5:                                                ; preds = %3
  %6 = zext i32 %2 to i64
  br label %12

7:                                                ; preds = %12, %3
  %8 = phi double [ 0.000000e+00, %3 ], [ %31, %12 ]
  %9 = phi double [ 0.000000e+00, %3 ], [ %27, %12 ]
  %10 = insertvalue { double, double } undef, double %9, 0
  %11 = insertvalue { double, double } %10, double %8, 1
  ret { double, double } %11

12:                                               ; preds = %5, %12
  %13 = phi i64 [ 0, %5 ], [ %32, %12 ]
  %14 = phi double [ 0.000000e+00, %5 ], [ %27, %12 ]
  %15 = phi double [ 0.000000e+00, %5 ], [ %31, %12 ]
  %16 = getelementptr inbounds %struct.complex, %struct.complex* %0, i64 %13, i32 0
  %17 = load double, double* %16, align 8, !tbaa !2
  %18 = getelementptr inbounds %struct.complex, %struct.complex* %1, i64 %13, i32 0
  %19 = load double, double* %18, align 8, !tbaa !2
  %20 = fmul fast double %19, %17
  %21 = getelementptr inbounds %struct.complex, %struct.complex* %0, i64 %13, i32 1
  %22 = load double, double* %21, align 8, !tbaa !7
  %23 = getelementptr inbounds %struct.complex, %struct.complex* %1, i64 %13, i32 1
  %24 = load double, double* %23, align 8, !tbaa !7
  %25 = fadd fast double %20, %14
  %26 = fmul fast double %24, %22
  %27 = fsub fast double %25, %26
  %28 = fmul fast double %24, %17
  %29 = fmul fast double %22, %19
  %30 = fadd fast double %29, %15
  %31 = fadd fast double %30, %28
  %32 = add nuw nsw i64 %13, 1
  %33 = icmp eq i64 %32, %6
  br i1 %33, label %7, label %12, !llvm.loop !8
}

; Function Attrs: nounwind uwtable
define dso_local void @myblas_cdot_fwd(%struct.TapeAndComplex* noalias sret(%struct.TapeAndComplex) align 8 %0, %struct.complex* %1, %struct.complex* %2, %struct.complex* %3, %struct.complex* %4, i32 %5, i32 %6) #1 {
  tail call void (%struct.TapeAndComplex*, i8*, ...) @__enzyme_augmentfwd(%struct.TapeAndComplex* sret(%struct.TapeAndComplex) align 8 %0, i8* bitcast ({ double, double } (%struct.complex*, %struct.complex*, i32)* @myblas_cdot to i8*), %struct.complex* %1, %struct.complex* %2, %struct.complex* %3, %struct.complex* %4, i32 %5) #4
  ret void
}

; Function Attrs: nounwind uwtable
define dso_local void @myblas_cdot_rev(%struct.complex* %0, %struct.complex* %1, %struct.complex* %2, %struct.complex* %3, i32 %4, i32 %5, %struct.complex* %6, i8* %7) #1 {
  %9 = load i32, i32* @enzyme_byref, align 4, !tbaa !10
  tail call void (i8*, ...) @__enzyme_reverse(i8* bitcast ({ double, double } (%struct.complex*, %struct.complex*, i32)* @myblas_cdot to i8*), %struct.complex* %0, %struct.complex* %1, %struct.complex* %2, %struct.complex* %3, metadata !"enzyme_dup", i32 %4, i32 %5, i32 %9, i64 16, %struct.complex* %6, i8* %7) #4
  ret void
}

; Function Attrs: norecurse nounwind readnone uwtable willreturn
define dso_local double @myblas_cabs(double %0, double %1) #2 {
  %3 = fmul fast double %0, %0
  %4 = fmul fast double %1, %1
  %5 = fadd fast double %4, %3
  ret double %5
}

; Function Attrs: nounwind uwtable
define dso_local { i8*, double } @myblas_cabs_fwd(double %0, double %1) #1 {
  %3 = tail call { i8*, double } (i8*, ...) @__enzyme_augmentfwd2(i8* bitcast (double (double, double)* @myblas_cabs to i8*), double %0, double %1) #4
  ret { i8*, double } %3
}

; Function Attrs: nounwind uwtable
define dso_local { double, double } @myblas_cabs_rev(double %0, double %1, double %2, i8* %3) #1 {
  %5 = tail call { double, double } (i8*, ...) @__enzyme_reverse2(i8* bitcast (double (double, double)* @myblas_cabs to i8*), double %0, double %1, double %2, i8* %3) #4
  ret { double, double } %5
}

declare dso_local void @__enzyme_augmentfwd(%struct.TapeAndComplex* sret(%struct.TapeAndComplex) align 8, i8*, ...) local_unnamed_addr #3

declare dso_local void @__enzyme_reverse(i8*, ...) local_unnamed_addr #3

declare dso_local { i8*, double } @__enzyme_augmentfwd2(i8*, ...) local_unnamed_addr #3

declare dso_local { double, double } @__enzyme_reverse2(i8*, ...) local_unnamed_addr #3

attributes #0 = { norecurse nounwind readonly uwtable "denormal-fp-math"="preserve-sign,preserve-sign" "denormal-fp-math-f32"="ieee,ieee" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { nounwind uwtable "denormal-fp-math"="preserve-sign,preserve-sign" "denormal-fp-math-f32"="ieee,ieee" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #2 = { norecurse nounwind readnone uwtable willreturn "denormal-fp-math"="preserve-sign,preserve-sign" "denormal-fp-math-f32"="ieee,ieee" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #3 = { "denormal-fp-math"="preserve-sign,preserve-sign" "denormal-fp-math-f32"="ieee,ieee" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #4 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"Ubuntu clang version 12.0.1-++20211029101322+fed41342a82f-1~exp1~20211029221816.4"}
!2 = !{!3, !4, i64 0}
!3 = !{!"complex", !4, i64 0, !4, i64 8}
!4 = !{!"double", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C/C++ TBAA"}
!7 = !{!3, !4, i64 8}
!8 = distinct !{!8, !9}
!9 = !{!"llvm.loop.mustprogress"}
!10 = !{!11, !11, i64 0}
!11 = !{!"int", !5, i64 0}

; CHECK: define internal void @fixbyval_myblas_cdot_rev(%struct.complex* %arg0, %struct.complex* %arg1, %struct.complex* %arg2, %struct.complex* %arg3, i32 %arg4, i32 %arg5, %struct.complex %arg6, i8* %arg7)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = alloca %struct.complex
; CHECK-NEXT:   store %struct.complex %arg6, %struct.complex* %0
; CHECK-NEXT:   call void @myblas_cdot_rev(%struct.complex* %arg0, %struct.complex* %arg1, %struct.complex* %arg2, %struct.complex* %arg3, i32 %arg4, i32 %arg5, %struct.complex* %0, i8* %arg7)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
