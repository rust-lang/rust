; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -early-cse -simplifycfg -correlated-propagation -adce -S | FileCheck %s
source_filename = "llvm-link"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare i64 @use()

declare dso_local void @_Z17__enzyme_autodiffPvS_S_S_S_(i8*, i8*, i8*, i8*, i8*) local_unnamed_addr #1

; Function Attrs: inlinehint nofree nounwind uwtable mustprogress
define internal void @_ZL16LagrangeLeapFrogR6DomainPd(double* noalias %pbvc, double* noalias %out) #22 {
entry:
  br label %for.body

for.body:                                         ; preds = %for.cond.cleanup15, %entry
  %i = phi i64 [ %nexti, %for.cond.cleanup15 ], [ 0, %entry ]
  %i5 = call i64 @use()
  store double 1.000000e+00, double* %pbvc, align 8
  br label %for.body16

for.body16:                                       ; preds = %cdce.end, %for.body16.preheader
  %k = phi i64 [ 0, %for.body ], [ %nextk, %for.body16 ]
  %arrayidx18 = getelementptr inbounds double, double* %pbvc, i64 %k
  %i10 = load double, double* %arrayidx18, align 8
  %mul = fmul double %i10, %i10
  store double %mul, double* %out
  %nextk = add nuw nsw i64 %k, 1
  %exitcond66.not = icmp eq i64 %nextk, %i5
  br i1 %exitcond66.not, label %for.cond.cleanup15, label %for.body16

for.cond.cleanup15:                               ; preds = %cdce.end, %for.cond13.preheader, %for.body
  %nexti = add nuw nsw i64 %i, 1
  %cmp = icmp ne i64 %nexti, 10
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup15, %entry
  ret void
}

define dso_local i64 @meta(i8* %call30, i8* %call35, i8* %i49, i8* %i50) {
entry:
  call void @_Z17__enzyme_autodiffPvS_S_S_S_(i8* bitcast (void (double*, double*)* @_ZL16LagrangeLeapFrogR6DomainPd to i8*), i8* %call30, i8* %call35, i8* nonnull %i49, i8* nonnull %i50)
  ret i64 0
}

!llvm.ident = !{!0, !0, !0, !0, !0}
!llvm.module.flags = !{!1, !2, !3, !4}

!0 = !{!"clang version 13.0.0 (git@github.com:llvm/llvm-project 619bfe8bd23f76b22f0a53fedafbfc8c97a15f12)"}
!1 = !{i64 1, !"wchar_size", i64 4}
!2 = !{i64 7, !"uwtable", i64 1}
!3 = !{i64 1, !"ThinLTO", i64 0}
!4 = !{i64 1, !"EnableSplitLTOUnit", i64 1}
!5 = !{!6, !6, i64 0}
!6 = !{!"int", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C++ TBAA"}

; CHECK: define internal void @diffe_ZL16LagrangeLeapFrogR6DomainPd(double* noalias %pbvc, double* %"pbvc'", double* noalias %out, double* %"out'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %malloccall = tail call noalias nonnull dereferenceable(80) dereferenceable_or_null(80) i8* @malloc(i64 80)
; CHECK-NEXT:   %i10_malloccache = bitcast i8* %malloccall to double**
; CHECK-NEXT:   %malloccall7 = tail call noalias nonnull dereferenceable(80) dereferenceable_or_null(80) i8* @malloc(i64 80)
; CHECK-NEXT:   %_malloccache = bitcast i8* %malloccall7 to i64*
; CHECK-NEXT:   br label %for.body

; CHECK: for.body:                                         ; preds = %for.cond.cleanup15, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %for.cond.cleanup15 ], [ 0, %entry ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %i5 = call i64 @use()
; CHECK-NEXT:   store double 1.000000e+00, double* %pbvc, align 8
; CHECK-NEXT:   %0 = add i64 %i5, -1
; CHECK-NEXT:   %1 = getelementptr inbounds double*, double** %i10_malloccache, i64 %iv
; CHECK-NEXT:   %mallocsize = mul nuw nsw i64 %i5, 8
; CHECK-NEXT:   %malloccall3 = tail call noalias nonnull i8* @malloc(i64 %mallocsize)
; CHECK-NEXT:   %i10_malloccache4 = bitcast i8* %malloccall3 to double*
; CHECK-NEXT:   store double* %i10_malloccache4, double** %1, align 8, !invariant.group !5
; CHECK-NEXT:   %2 = getelementptr inbounds i64, i64* %_malloccache, i64 %iv
; CHECK-NEXT:   store i64 %0, i64* %2, align 8, !invariant.group !6
; CHECK-NEXT:   %3 = load double*, double** %1, align 8, !dereferenceable !7, !invariant.group !5
; CHECK-NEXT:   %4 = bitcast double* %3 to i8*
; CHECK-NEXT:   %5 = bitcast double* %pbvc to i8*
; CHECK-NEXT:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 %4, i8* nonnull align 8 %5, i64 %mallocsize, i1 false)
; CHECK-NEXT:   br label %for.body16

; CHECK: for.body16:                                       ; preds = %for.body16, %for.body
; CHECK-NEXT:   %iv1 = phi i64 [ %iv.next2, %for.body16 ], [ 0, %for.body ]
; CHECK-NEXT:   %iv.next2 = add nuw nsw i64 %iv1, 1
; CHECK-NEXT:   %arrayidx18 = getelementptr inbounds double, double* %pbvc, i64 %iv1
; CHECK-NEXT:   %i10 = load double, double* %arrayidx18, align 8
; CHECK-NEXT:   %mul = fmul double %i10, %i10
; CHECK-NEXT:   store double %mul, double* %out
; CHECK-NEXT:   %exitcond66.not = icmp eq i64 %iv.next2, %i5
; CHECK-NEXT:   br i1 %exitcond66.not, label %for.cond.cleanup15, label %for.body16

; CHECK: for.cond.cleanup15:                               ; preds = %for.body16
; CHECK-NEXT:   %cmp = icmp ne i64 %iv.next, 10
; CHECK-NEXT:   br i1 %cmp, label %for.body, label %invertfor.cond.cleanup15

; CHECK: invertentry:                                      ; preds = %invertfor.body
; CHECK-NEXT:   tail call void @free(i8* nonnull %malloccall)
; CHECK-NEXT:   tail call void @free(i8* nonnull %malloccall7)
; CHECK-NEXT:   ret void

; CHECK: invertfor.body:                                   ; preds = %invertfor.body16
; CHECK-NEXT:   store double 0.000000e+00, double* %"pbvc'", align 8
; CHECK-NEXT:   %6 = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   %forfree6 = load double*, double** %10, align 8, !dereferenceable !7, !invariant.group !5
; CHECK-NEXT:   %7 = bitcast double* %forfree6 to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %7)
; CHECK-NEXT:   br i1 %6, label %invertentry, label %incinvertfor.body

; CHECK: incinvertfor.body:                                ; preds = %invertfor.body
; CHECK-NEXT:   %8 = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertfor.cond.cleanup15

; CHECK: invertfor.body16:                                 ; preds = %invertfor.cond.cleanup15, %incinvertfor.body16
; CHECK-NEXT:   %"iv1'ac.0" = phi i64 [ %20, %invertfor.cond.cleanup15 ], [ %18, %incinvertfor.body16 ]
; CHECK-NEXT:   %9 = load double, double* %"out'"
; CHECK-NEXT:   store double 0.000000e+00, double* %"out'"
; CHECK-NEXT:   %10 = getelementptr inbounds double*, double** %i10_malloccache, i64 %"iv'ac.0"
; CHECK-NEXT:   %11 = load double*, double** %10, align 8, !dereferenceable !7, !invariant.group !5
; CHECK-NEXT:   %12 = getelementptr inbounds double, double* %11, i64 %"iv1'ac.0"
; CHECK-NEXT:   %13 = load double, double* %12, align 8, !invariant.group !8
; CHECK-NEXT:   %m0diffei10 = fmul fast double %9, %13
; CHECK-NEXT:   %14 = fadd fast double %m0diffei10, %m0diffei10
; CHECK-NEXT:   %"arrayidx18'ipg_unwrap" = getelementptr inbounds double, double* %"pbvc'", i64 %"iv1'ac.0"
; CHECK-NEXT:   %15 = load double, double* %"arrayidx18'ipg_unwrap", align 8
; CHECK-NEXT:   %16 = fadd fast double %15, %14
; CHECK-NEXT:   store double %16, double* %"arrayidx18'ipg_unwrap", align 8
; CHECK-NEXT:   %17 = icmp eq i64 %"iv1'ac.0", 0
; CHECK-NEXT:   br i1 %17, label %invertfor.body, label %incinvertfor.body16

; CHECK: incinvertfor.body16:                              ; preds = %invertfor.body16
; CHECK-NEXT:   %18 = add nsw i64 %"iv1'ac.0", -1
; CHECK-NEXT:   br label %invertfor.body16

; CHECK: invertfor.cond.cleanup15:                         ; preds = %for.cond.cleanup15, %incinvertfor.body
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %8, %incinvertfor.body ], [ 9, %for.cond.cleanup15 ]
; CHECK-NEXT:   %19 = getelementptr inbounds i64, i64* %_malloccache, i64 %"iv'ac.0"
; CHECK-NEXT:   %20 = load i64, i64* %19, align 8, !invariant.group !6
; CHECK-NEXT:   br label %invertfor.body16
; CHECK-NEXT: }
