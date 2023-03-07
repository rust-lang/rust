; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -correlated-propagation -adce -instsimplify -early-cse-memssa -simplifycfg -correlated-propagation -adce -jump-threading -instsimplify -early-cse -simplifycfg -S | FileCheck %s

; ModuleID = 'inp.ll'
source_filename = "../benchmarks/hand/hand.cpp"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.Triangle = type { [3 x i32] }
%struct.Matrix = type { i64, i64, double* }

@enzyme_dup = external dso_local local_unnamed_addr global i32, align 4
@enzyme_const = external dso_local local_unnamed_addr global i32, align 4
@enzyme_dupnoneed = external dso_local local_unnamed_addr global i32, align 4

; Function Attrs: nounwind
declare dso_local noalias i8* @malloc(i64) local_unnamed_addr #0

declare dso_local void @__enzyme_autodiff(...)

; Function Attrs: nounwind uwtable
define dso_local void @dhand_objective(i1 %cmp, i64* %parents, double* %M, double* %Mp) local_unnamed_addr #1 {
entry:
  tail call void (...) @__enzyme_autodiff(void (i1, i64*, double*)* nonnull @hand_objective, i1 %cmp, i64* %parents, metadata !"enzyme_dup", double* %M, double* %Mp) #2
  ret void
}

; Function Attrs: nounwind uwtable
define dso_local void @hand_objective(i1 %cmp, i64* %parents, double* %M) #1 {
entry:
  %call.i = tail call noalias i8* @malloc(i64 8) #2
  %tmp = bitcast i8* %call.i to i64*
  
  tail call void @relatives_to_absolutes(i1 %cmp, i64* %tmp, double* %M)
  ret void
}

declare double @llvm.sin.f64(double)

; Function Attrs: nounwind uwtable
define internal void @relatives_to_absolutes(i1 %cmp, i64* %r, double* %tmp9) local_unnamed_addr #1 {
entry:
  br i1 %cmp, label %cond, label %mid

cond:                         ; preds = %entry
  %.pre.i = load i64, i64* %r, align 8, !tbaa !15, !alias.scope !16, !noalias !17
  br label %mid

mid:                                         ; preds = %if.then.i, %cond
  %lim = phi i64 [ %.pre.i, %cond ], [ 3, %entry ]
  store i64 %lim, i64* %r, align 8, !tbaa !15, !alias.scope !9, !noalias !12
  br label %loop1

loop1:                            ; preds = %for.inc28.i, %for.cond8.preheader.lr.ph.i
  %i = phi i64 [ 0, %mid ], [ %ni, %inc1 ]
  br label %loop2

loop2:                                     ; preds = %for.body14.i, %for.body14.lr.ph.i
  %j = phi i64 [ 0, %loop1 ], [ %indvars.iv.next.i, %loop2 ]
  %tmp15 = load double, double* %tmp9, align 8, !tbaa !22, !noalias !20
  %addj = call double @llvm.sin.f64(double %tmp15)
  store double %addj, double* %tmp9, align 8, !tbaa !22, !noalias !20
  %indvars.iv.next.i = add nuw nsw i64 %j, 1
  %exitcond.i = icmp eq i64 %indvars.iv.next.i, 22
  br i1 %exitcond.i, label %inc1, label %loop2

inc1:                                      ; preds = %for.end.i
  %ni = add nuw nsw i64 %i, 1
  %exitcond65.i = icmp eq i64 %ni, %lim
  br i1 %exitcond65.i, label %exit, label %loop1

exit:                                    ; preds = %for.inc28.i, %if.end.i
  ret void
}

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #2 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0}
!3 = !{!"int", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C++ TBAA"}
!6 = !{!7, !3, i64 4}
!7 = !{!"_ZTS6Matrix", !3, i64 0, !3, i64 4, !8, i64 8}
!8 = !{!"any pointer", !4, i64 0}
!9 = !{!10}
!10 = distinct !{!10, !11, !"mat_mult: %out"}
!11 = distinct !{!11, !"mat_mult"}
!12 = !{!13, !14}
!13 = distinct !{!13, !11, !"mat_mult: %lhs"}
!14 = distinct !{!14, !11, !"mat_mult: %rhs"}
!15 = !{!7, !3, i64 0}
!16 = !{!13}
!17 = !{!14, !10}
!18 = !{!14}
!19 = !{!13, !10}
!20 = !{!13, !14, !10}
!21 = !{!7, !8, i64 8}
!22 = !{!23, !23, i64 0}
!23 = !{!"double", !4, i64 0}

; CHECK: define internal void @differelatives_to_absolutes(i1 %cmp, i64* %r, i64* %"r'", double* %tmp9, double* %"tmp9'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   br i1 %cmp, label %cond, label %mid

; CHECK: cond:                                             ; preds = %entry
; CHECK-NEXT:   %.pre.i = load i64, i64* %r, align 8
; CHECK-NEXT:   br label %mid

; CHECK: mid:                                              ; preds = %cond, %entry
; CHECK-NEXT:   %lim = phi i64 [ %.pre.i, %cond ], [ 3, %entry ]
; CHECK-NEXT:   store i64 %lim, i64* %"r'", align 8
; CHECK-NEXT:   store i64 %lim, i64* %r, align 8
; CHECK-NEXT:   %0 = add i64 %lim, -1
; CHECK-NEXT:   %1 = mul nuw nsw i64 22, %lim
; CHECK-NEXT:   %mallocsize = mul nuw nsw i64 %1, 8
; CHECK-NEXT:   %malloccall = tail call noalias nonnull i8* @malloc(i64 %mallocsize)
; CHECK-NEXT:   %tmp15_malloccache = bitcast i8* %malloccall to double*
; CHECK-NEXT:   br label %loop1

; CHECK: loop1:                                            ; preds = %inc1, %mid
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %inc1 ], [ 0, %mid ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   br label %loop2

; CHECK: loop2:                                            ; preds = %loop2, %loop1
; CHECK-NEXT:   %iv1 = phi i64 [ %iv.next2, %loop2 ], [ 0, %loop1 ]
; CHECK-NEXT:   %iv.next2 = add nuw nsw i64 %iv1, 1
; CHECK-NEXT:   %tmp15 = load double, double* %tmp9, align 8
; CHECK-NEXT:   %addj = call double @llvm.sin.f64(double %tmp15)
; CHECK-NEXT:   store double %addj, double* %tmp9, align 8
; CHECK-NEXT:   %2 = mul nuw nsw i64 %iv, 22
; CHECK-NEXT:   %3 = add nuw nsw i64 %iv1, %2
; CHECK-NEXT:   %4 = getelementptr inbounds double, double* %tmp15_malloccache, i64 %3
; CHECK-NEXT:   store double %tmp15, double* %4, align 8
; CHECK-NEXT:   %exitcond.i = icmp eq i64 %iv.next2, 22
; CHECK-NEXT:   br i1 %exitcond.i, label %inc1, label %loop2

; CHECK: inc1:                                             ; preds = %loop2
; CHECK-NEXT:   %exitcond65.i = icmp eq i64 %iv.next, %lim
; CHECK-NEXT:   br i1 %exitcond65.i, label %invertinc1, label %loop1

; CHECK: invertmid:                                        ; preds = %invertloop1
; CHECK-NEXT:   tail call void @free(i8* nonnull %malloccall)
; CHECK-NEXT:   ret void

; CHECK: invertloop1:                                      ; preds = %invertloop2
; CHECK-NEXT:   %5 = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %5, label %invertmid, label %incinvertloop1

; CHECK: incinvertloop1:                                   ; preds = %invertloop1
; CHECK-NEXT:   %6 = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertinc1

; CHECK: invertloop2:                                      ; preds = %invertinc1, %incinvertloop2
; CHECK-NEXT:   %"iv1'ac.0" = phi i64 [ 21, %invertinc1 ], [ %15, %incinvertloop2 ]
; CHECK-NEXT:   %7 = load double, double* %"tmp9'", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"tmp9'", align 8
; CHECK-NEXT:   %8 = mul nuw nsw i64 %"iv'ac.0", 22
; CHECK-NEXT:   %9 = add nuw nsw i64 %"iv1'ac.0", %8
; CHECK-NEXT:   %10 = getelementptr inbounds double, double* %tmp15_malloccache, i64 %9
; CHECK-NEXT:   %11 = load double, double* %10, align 8
; CHECK-NEXT:   %12 = call fast double @llvm.cos.f64(double %11)
; CHECK-NEXT:   %13 = fmul fast double %7, %12
; CHECK-NEXT:   store double %13, double* %"tmp9'", align 8
; CHECK-NEXT:   %14 = icmp eq i64 %"iv1'ac.0", 0
; CHECK-NEXT:   br i1 %14, label %invertloop1, label %incinvertloop2

; CHECK: incinvertloop2:                                   ; preds = %invertloop2
; CHECK-NEXT:   %15 = add nsw i64 %"iv1'ac.0", -1
; CHECK-NEXT:   br label %invertloop2

; CHECK: invertinc1:                                       ; preds = %inc1, %incinvertloop1
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %6, %incinvertloop1 ], [ %0, %inc1 ]
; CHECK-NEXT:   br label %invertloop2
; CHECK-NEXT: }