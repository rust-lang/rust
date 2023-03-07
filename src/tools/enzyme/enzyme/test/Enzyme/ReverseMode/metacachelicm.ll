; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

; Function Attrs: nounwind uwtable
define dso_local void @compute(double* noalias nocapture %data, i64* noalias nocapture readnone %array, double* noalias nocapture %out) #0 {
entry:
  br label %for.body5.preheader

for.cond.cleanup:                                 ; preds = %for.cond.cleanup4
  store double 0.000000e+00, double* %data, align 8, !tbaa !2
  ret void

for.body5.preheader:                              ; preds = %entry, %for.cond.cleanup4
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.cond.cleanup4 ]
  %call = tail call i64 @getSize() #2
  br label %for.body5

for.cond.cleanup4:                                ; preds = %for.body5
  %arrayidx7 = getelementptr inbounds double, double* %out, i64 %indvars.iv
  store double %add, double* %arrayidx7, align 8, !tbaa !2
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond29 = icmp eq i64 %indvars.iv.next, 10
  br i1 %exitcond29, label %for.cond.cleanup, label %for.body5.preheader

for.body5:                                        ; preds = %for.body5, %for.body5.preheader
  %j.027 = phi i64 [ %inc, %for.body5 ], [ 0, %for.body5.preheader ]
  %res.026 = phi double [ %add, %for.body5 ], [ 0.000000e+00, %for.body5.preheader ]
  %arrayidx = getelementptr inbounds double, double* %data, i64 %j.027
  %i0 = load double, double* %arrayidx, align 8, !tbaa !2
  %mul = fmul double %i0, %i0
  %add = fadd double %res.026, %mul
  %inc = add nuw i64 %j.027, 1
  %exitcond = icmp eq i64 %inc, %call
  br i1 %exitcond, label %for.cond.cleanup4, label %for.body5
}

declare dso_local i64 @getSize() local_unnamed_addr #1

; Function Attrs: nounwind
declare void @llvm.assume(i1) #2

; Function Attrs: nounwind uwtable
define dso_local void @call(double* %data, double* %d_data, i64* %array, double* %out, double* %d_out) local_unnamed_addr #0 {
entry:
  tail call void (i8*, ...) @__enzyme_autodiff(i8* bitcast (void (double*, i64*, double*)* @compute to i8*), double* %data, double* %d_data, i64* %array, double* %out, double* %d_out) #2
  ret void
}

declare dso_local void @__enzyme_autodiff(i8*, ...) local_unnamed_addr #1

attributes #0 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0}
!3 = !{!"double", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}

; CHECK: define internal void @diffecompute(double* noalias nocapture %data, double* nocapture %"data'", i64* noalias nocapture readnone %array, double* noalias nocapture %out, double* nocapture %"out'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %malloccall = tail call noalias nonnull dereferenceable(80) dereferenceable_or_null(80) i8* @malloc(i64 80)
; CHECK-NEXT:   %call_malloccache = bitcast i8* %malloccall to i64*
; CHECK-NEXT:   %[[malloccall3:.+]] = tail call noalias nonnull dereferenceable(80) dereferenceable_or_null(80) i8* @malloc(i64 80)
; CHECK-NEXT:   %i0_malloccache = bitcast i8* %[[malloccall3]] to double**
; CHECK-NEXT:   br label %for.body5.preheader

; CHECK: for.cond.cleanup:                                 ; preds = %for.cond.cleanup4
; CHECK-NEXT:   store double 0.000000e+00, double* %data, align 8, !tbaa !2
; CHECK-NEXT:   store double 0.000000e+00, double* %"data'", align 8
; CHECK-NEXT:   br label %invertfor.cond.cleanup4

; CHECK: for.body5.preheader:                              ; preds = %for.cond.cleanup4, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %for.cond.cleanup4 ], [ 0, %entry ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %call = tail call i64 @getSize()
; CHECK-NEXT:   %[[a1:.+]] = getelementptr inbounds i64, i64* %call_malloccache, i64 %iv
; CHECK-NEXT:   store i64 %call, i64* %[[a1]], align 8, !invariant.group ![[g6:[0-9]+]]
; CHECK-NEXT:   %[[a2:.+]] = getelementptr inbounds double*, double** %i0_malloccache, i64 %iv
; CHECK-NEXT:   %mallocsize = mul nuw nsw i64 %call, 8
; CHECK-NEXT:   %[[malloccall5:.+]] = tail call noalias nonnull i8* @malloc(i64 %mallocsize)
; CHECK-NEXT:   %[[i0_malloccache6:.+]] = bitcast i8* %[[malloccall5]] to double*
; CHECK-NEXT:   store double* %[[i0_malloccache6]], double** %[[a2]], align 8, !invariant.group ![[g7:[0-9]+]]
; CHECK-NEXT:   %[[a3:.+]] = getelementptr inbounds double*, double** %i0_malloccache, i64 %iv
; CHECK-NEXT:   %[[a4:.+]] = load double*, double** %[[a3]], align 8, !dereferenceable !{{[0-9]+}}, !invariant.group ![[g7]]
; CHECK-NEXT:   %[[a5:.+]] = bitcast double* %[[a4]] to i8*
; CHECK-NEXT:   %[[a6:.+]] = bitcast double* %data to i8*
; CHECK-NEXT:   %[[a7:.+]] = mul nuw nsw i64 8, %call
; CHECK-NEXT:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 %[[a5]], i8* nonnull align 8 %[[a6]], i64 %[[a7]], i1 false)
; CHECK-NEXT:   br label %for.body5

; CHECK: for.cond.cleanup4:                                ; preds = %for.body5
; CHECK-NEXT:   %arrayidx7 = getelementptr inbounds double, double* %out, i64 %iv
; CHECK-NEXT:   store double %add, double* %arrayidx7, align 8, !tbaa !2
; CHECK-NEXT:   %exitcond29 = icmp eq i64 %iv.next, 10
; CHECK-NEXT:   br i1 %exitcond29, label %for.cond.cleanup, label %for.body5.preheader

; CHECK: for.body5:                                        ; preds = %for.body5, %for.body5.preheader
; CHECK-NEXT:   %iv1 = phi i64 [ %iv.next2, %for.body5 ], [ 0, %for.body5.preheader ]
; CHECK-NEXT:   %res.026 = phi double [ %add, %for.body5 ], [ 0.000000e+00, %for.body5.preheader ]
; CHECK-NEXT:   %iv.next2 = add nuw nsw i64 %iv1, 1
; CHECK-NEXT:   %arrayidx = getelementptr inbounds double, double* %data, i64 %iv1
; CHECK-NEXT:   %i0 = load double, double* %arrayidx, align 8, !tbaa !2
; CHECK-NEXT:   %mul = fmul double %i0, %i0
; CHECK-NEXT:   %add = fadd double %res.026, %mul
; CHECK-NEXT:   %exitcond = icmp eq i64 %iv.next2, %call
; CHECK-NEXT:   br i1 %exitcond, label %for.cond.cleanup4, label %for.body5

; CHECK: invertentry:                                      ; preds = %invertfor.body5.preheader
; CHECK-NEXT:   tail call void @free(i8* nonnull %malloccall)
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[malloccall3]])
; CHECK-NEXT:   ret void

; CHECK: invertfor.body5.preheader:                        ; preds = %invertfor.body5
; CHECK-NEXT:   %[[a8:.+]] = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   %[[_unwrap7:.+]] = getelementptr inbounds double*, double** %i0_malloccache, i64 %"iv'ac.0"
; CHECK-NEXT:   %[[forfree8:.+]] = load double*, double** %[[_unwrap7]], align 8, !dereferenceable !{{[0-9]+}}, !invariant.group ![[g7]]
; CHECK-NEXT:   %[[a9:.+]] = bitcast double* %[[forfree8]] to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[a9]])
; CHECK-NEXT:   br i1 %[[a8]], label %invertentry, label %incinvertfor.body5.preheader

; CHECK: incinvertfor.body5.preheader:                     ; preds = %invertfor.body5.preheader
; CHECK-NEXT:   %[[a10:.+]] = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertfor.cond.cleanup4

; CHECK: invertfor.cond.cleanup4:                          ; preds = %incinvertfor.body5.preheader, %for.cond.cleanup
; CHECK-NEXT:   %"add'de.0" = phi double [ 0.000000e+00, %for.cond.cleanup ], [ %[[a23:.+]], %incinvertfor.body5.preheader ]
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ 9, %for.cond.cleanup ], [ %[[a10]], %incinvertfor.body5.preheader ]
; CHECK-NEXT:   %"arrayidx7'ipg_unwrap" = getelementptr inbounds double, double* %"out'", i64 %"iv'ac.0"
; CHECK-NEXT:   %[[a11:.+]] = load double, double* %"arrayidx7'ipg_unwrap", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"arrayidx7'ipg_unwrap", align 8
; CHECK-NEXT:   %[[a12:.+]] = fadd fast double %"add'de.0", %[[a11]]
; CHECK-NEXT:   %[[a13:.+]] = getelementptr inbounds i64, i64* %call_malloccache, i64 %"iv'ac.0"
; CHECK-NEXT:   %[[cload:.+]] = load i64, i64* %[[a13]], align 8, !invariant.group ![[g6]]
; CHECK-NEXT:   %[[a14:.+]] = add i64 %[[cload]], -1
; CHECK-NEXT:   br label %invertfor.body5

; CHECK: invertfor.body5:                                  ; preds = %incinvertfor.body5, %invertfor.cond.cleanup4
; CHECK-NEXT:   %"add'de.1" = phi double [ %[[a12]], %invertfor.cond.cleanup4 ], [ %[[a23]], %incinvertfor.body5 ]
; CHECK-NEXT:   %"iv1'ac.0" = phi i64 [ %[[a14]], %invertfor.cond.cleanup4 ], [ %[[a24:.+]], %incinvertfor.body5 ]
; CHECK-NEXT:   %[[a15:.+]] = getelementptr inbounds double*, double** %i0_malloccache, i64 %"iv'ac.0"
; CHECK-NEXT:   %[[a16:.+]] = load double*, double** %[[a15]], align 8, !dereferenceable !{{[0-9]+}}, !invariant.group ![[g7]]
; CHECK-NEXT:   %[[a17:.+]] = getelementptr inbounds double, double* %[[a16:.+]], i64 %"iv1'ac.0"
; CHECK-NEXT:   %[[a18:.+]] = load double, double* %[[a17]], align 8, !invariant.group !
; CHECK-NEXT:   %m0diffei0 = fmul fast double %"add'de.1", %[[a18]]
; CHECK-NEXT:   %m1diffei0 = fmul fast double %"add'de.1", %[[a18]]
; CHECK-NEXT:   %[[a19:.+]] = fadd fast double %m0diffei0, %m1diffei0
; CHECK-NEXT:   %"arrayidx'ipg_unwrap" = getelementptr inbounds double, double* %"data'", i64 %"iv1'ac.0"
; CHECK-NEXT:   %[[a20:.+]] = load double, double* %"arrayidx'ipg_unwrap", align 8
; CHECK-NEXT:   %[[a21:.+]] = fadd fast double %[[a20]], %[[a19]]
; CHECK-NEXT:   store double %[[a21:.+]], double* %"arrayidx'ipg_unwrap", align 8
; CHECK-NEXT:   %[[a22:.+]] = icmp eq i64 %"iv1'ac.0", 0
; CHECK-NEXT:   %[[a23]] = select{{( fast)?}} i1 %[[a22]], double 0.000000e+00, double %"add'de.1"
; CHECK-NEXT:   br i1 %[[a22]], label %invertfor.body5.preheader, label %incinvertfor.body5

; CHECK: incinvertfor.body5:                               ; preds = %invertfor.body5
; CHECK-NEXT:   %[[a24]] = add nsw i64 %"iv1'ac.0", -1
; CHECK-NEXT:   br label %invertfor.body5
; CHECK-NEXT: }
