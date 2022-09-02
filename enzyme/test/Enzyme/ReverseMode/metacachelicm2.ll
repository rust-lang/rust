; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

; Function Attrs: nounwind uwtable
define dso_local void @compute(double* noalias nocapture %data, i64* noalias nocapture readonly %array, double* noalias nocapture %out) #0 {
entry:
  br label %for.body5.preheader

for.cond.cleanup:                                 ; preds = %for.cond.cleanup4
  store double 0.000000e+00, double* %data, align 8, !tbaa !2
  ret void

for.body5.preheader:                              ; preds = %entry, %for.cond.cleanup4
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.cond.cleanup4 ]
  %arrayidx = getelementptr inbounds i64, i64* %array, i64 %indvars.iv
  %len = load i64, i64* %arrayidx, align 8, !tbaa !6
  br label %for.body5

for.cond.cleanup4:                                ; preds = %for.body5
  %arrayidx9 = getelementptr inbounds double, double* %out, i64 %indvars.iv
  store double %add, double* %arrayidx9, align 8, !tbaa !2
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond32 = icmp eq i64 %indvars.iv.next, 10
  br i1 %exitcond32, label %for.cond.cleanup, label %for.body5.preheader

for.body5:                                        ; preds = %for.body5, %for.body5.preheader
  %j = phi i64 [ %inc, %for.body5 ], [ 0, %for.body5.preheader ]
  %res.029 = phi double [ %add, %for.body5 ], [ 0.000000e+00, %for.body5.preheader ]
  %arrayidx6 = getelementptr inbounds double, double* %data, i64 %j
  %ld = load double, double* %arrayidx6, align 8, !tbaa !2
  %mul = fmul double %ld, %ld
  %add = fadd double %res.029, %mul
  %inc = add nuw i64 %j, 1
  %exitcond = icmp eq i64 %inc, %len
  br i1 %exitcond, label %for.cond.cleanup4, label %for.body5
}

; Function Attrs: nounwind
declare void @llvm.assume(i1) #1

; Function Attrs: nounwind uwtable
define dso_local void @call(double* %data, double* %d_data, i64* %array, double* %out, double* %d_out) local_unnamed_addr #0 {
entry:
  tail call void (i8*, ...) @__enzyme_autodiff(i8* bitcast (void (double*, i64*, double*)* @compute to i8*), double* %data, double* %d_data, i64* %array, double* %out, double* %d_out) #1
  ret void
}

declare dso_local void @__enzyme_autodiff(i8*, ...) local_unnamed_addr #2

attributes #0 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0}
!3 = !{!"double", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
!6 = !{!7, !7, i64 0}
!7 = !{!"long", !4, i64 0}

; CHECK: define internal void @diffecompute(double* noalias nocapture %data, double* nocapture %"data'", i64* noalias nocapture readonly %array, double* noalias nocapture %out, double* nocapture %"out'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %malloccall = tail call noalias nonnull dereferenceable(80) dereferenceable_or_null(80) i8* @malloc(i64 80)
; CHECK-NEXT:   %ld_malloccache = bitcast i8* %malloccall to double**
; CHECK-NEXT:   br label %for.body5.preheader

; CHECK: for.cond.cleanup:                                 ; preds = %for.cond.cleanup4
; CHECK-NEXT:   store double 0.000000e+00, double* %data, align 8, !tbaa !2
; CHECK-NEXT:   store double 0.000000e+00, double* %"data'", align 8
; CHECK-NEXT:   br label %invertfor.cond.cleanup4

; CHECK: for.body5.preheader:                              ; preds = %for.cond.cleanup4, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %for.cond.cleanup4 ], [ 0, %entry ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %arrayidx = getelementptr inbounds i64, i64* %array, i64 %iv
; CHECK-NEXT:   %len = load i64, i64* %arrayidx, align 8, !tbaa !6, !invariant.group ![[g8:[0-9]+]]
; CHECK-NEXT:   %0 = getelementptr inbounds double*, double** %ld_malloccache, i64 %iv
; CHECK-NEXT:   %mallocsize = mul nuw nsw i64 %len, 8
; CHECK-NEXT:   %[[malloccall3:.+]] = tail call noalias nonnull i8* @malloc(i64 %mallocsize)
; CHECK-NEXT:   %[[ld_malloccache4:.+]] = bitcast i8* %[[malloccall3]] to double*
; CHECK-NEXT:   store double* %[[ld_malloccache4]], double** %0, align 8, !invariant.group ![[g9:[0-9]+]]
; CHECK-NEXT:   %1 = getelementptr inbounds double*, double** %ld_malloccache, i64 %iv
; CHECK-NEXT:   %2 = load double*, double** %1, align 8, !dereferenceable !{{[0-9]+}}, !invariant.group ![[g9]]
; CHECK-NEXT:   %3 = bitcast double* %2 to i8*
; CHECK-NEXT:   %4 = bitcast double* %data to i8*
; CHECK-NEXT:   %5 = mul nuw nsw i64 8, %len
; CHECK-NEXT:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 %3, i8* nonnull align 8 %4, i64 %5, i1 false)
; CHECK-NEXT:   br label %for.body5

; CHECK: for.cond.cleanup4:                                ; preds = %for.body5
; CHECK-NEXT:   %arrayidx9 = getelementptr inbounds double, double* %out, i64 %iv
; CHECK-NEXT:   store double %add, double* %arrayidx9, align 8, !tbaa !2
; CHECK-NEXT:   %exitcond32 = icmp eq i64 %iv.next, 10
; CHECK-NEXT:   br i1 %exitcond32, label %for.cond.cleanup, label %for.body5.preheader

; CHECK: for.body5:                                        ; preds = %for.body5, %for.body5.preheader
; CHECK-NEXT:   %iv1 = phi i64 [ %iv.next2, %for.body5 ], [ 0, %for.body5.preheader ]
; CHECK-NEXT:   %res.029 = phi double [ %add, %for.body5 ], [ 0.000000e+00, %for.body5.preheader ]
; CHECK-NEXT:   %iv.next2 = add nuw nsw i64 %iv1, 1
; CHECK-NEXT:   %arrayidx6 = getelementptr inbounds double, double* %data, i64 %iv1
; CHECK-NEXT:   %ld = load double, double* %arrayidx6, align 8, !tbaa !2
; CHECK-NEXT:   %mul = fmul double %ld, %ld
; CHECK-NEXT:   %add = fadd double %res.029, %mul
; CHECK-NEXT:   %exitcond = icmp eq i64 %iv.next2, %len
; CHECK-NEXT:   br i1 %exitcond, label %for.cond.cleanup4, label %for.body5

; CHECK: invertentry:                                      ; preds = %invertfor.body5.preheader
; CHECK-NEXT:   tail call void @free(i8* nonnull %malloccall)
; CHECK-NEXT:   ret void

; CHECK: invertfor.body5.preheader:                        ; preds = %invertfor.body5
; CHECK-NEXT:   %6 = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   %[[_unwrap6:.+]] = getelementptr inbounds double*, double** %ld_malloccache, i64 %"iv'ac.0"
; CHECK-NEXT:   %[[forfree7:.+]] = load double*, double** %[[_unwrap6]], align 8, !dereferenceable !{{[0-9]+}}, !invariant.group ![[g9]]
; CHECK-NEXT:   %7 = bitcast double* %[[forfree7]] to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %7)
; CHECK-NEXT:   br i1 %6, label %invertentry, label %incinvertfor.body5.preheader

; CHECK: incinvertfor.body5.preheader:                     ; preds = %invertfor.body5.preheader
; CHECK-NEXT:   %8 = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertfor.cond.cleanup4

; CHECK: invertfor.cond.cleanup4:                          ; preds = %incinvertfor.body5.preheader, %for.cond.cleanup
; CHECK-NEXT:   %"add'de.0" = phi double [ 0.000000e+00, %for.cond.cleanup ], [ %19, %incinvertfor.body5.preheader ]
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ 9, %for.cond.cleanup ], [ %8, %incinvertfor.body5.preheader ]
; CHECK-NEXT:   %"arrayidx9'ipg_unwrap" = getelementptr inbounds double, double* %"out'", i64 %"iv'ac.0"
; CHECK-NEXT:   %9 = load double, double* %"arrayidx9'ipg_unwrap", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"arrayidx9'ipg_unwrap", align 8
; CHECK-NEXT:   %10 = fadd fast double %"add'de.0", %9
; CHECK-NEXT:   %arrayidx_unwrap = getelementptr inbounds i64, i64* %array, i64 %"iv'ac.0"
; CHECK-NEXT:   %len_unwrap = load i64, i64* %arrayidx_unwrap, align 8, !tbaa !6, !invariant.group ![[g8]]
; CHECK-NEXT:   %_unwrap = add i64 %len_unwrap, -1
; CHECK-NEXT:   br label %invertfor.body5

; CHECK: invertfor.body5:                                  ; preds = %incinvertfor.body5, %invertfor.cond.cleanup4
; CHECK-NEXT:   %"add'de.1" = phi double [ %10, %invertfor.cond.cleanup4 ], [ %19, %incinvertfor.body5 ]
; CHECK-NEXT:   %"iv1'ac.0" = phi i64 [ %_unwrap, %invertfor.cond.cleanup4 ], [ %20, %incinvertfor.body5 ]
; CHECK-NEXT:   %11 = getelementptr inbounds double*, double** %ld_malloccache, i64 %"iv'ac.0"
; CHECK-NEXT:   %12 = load double*, double** %11, align 8, !dereferenceable !{{[0-9]+}}, !invariant.group ![[g9]]
; CHECK-NEXT:   %13 = getelementptr inbounds double, double* %12, i64 %"iv1'ac.0"
; CHECK-NEXT:   %14 = load double, double* %13, align 8, !invariant.group !
; CHECK-NEXT:   %m0diffeld = fmul fast double %"add'de.1", %14
; CHECK-NEXT:   %m1diffeld = fmul fast double %"add'de.1", %14
; CHECK-NEXT:   %15 = fadd fast double %m0diffeld, %m1diffeld
; CHECK-NEXT:   %"arrayidx6'ipg_unwrap" = getelementptr inbounds double, double* %"data'", i64 %"iv1'ac.0"
; CHECK-NEXT:   %16 = load double, double* %"arrayidx6'ipg_unwrap", align 8
; CHECK-NEXT:   %17 = fadd fast double %16, %15
; CHECK-NEXT:   store double %17, double* %"arrayidx6'ipg_unwrap", align 8
; CHECK-NEXT:   %18 = icmp eq i64 %"iv1'ac.0", 0
; CHECK-NEXT:   %19 = select{{( fast)?}} i1 %18, double 0.000000e+00, double %"add'de.1"
; CHECK-NEXT:   br i1 %18, label %invertfor.body5.preheader, label %incinvertfor.body5

; CHECK: incinvertfor.body5:                               ; preds = %invertfor.body5
; CHECK-NEXT:   %20 = add nsw i64 %"iv1'ac.0", -1
; CHECK-NEXT:   br label %invertfor.body5
; CHECK-NEXT: }
