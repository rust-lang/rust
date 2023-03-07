; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -loop-deletion -correlated-propagation -adce -simplifycfg -S | FileCheck %s

; This requires the additional optimization to create memcpy's

source_filename = "mem.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@.str = private unnamed_addr constant [24 x i8] c"dat[%d]=%f ddat[%d]=%f\0A\00", align 1

; Function Attrs: nounwind readnone uwtable
define dso_local zeroext i1 @approx_fp_equality_float(float %f1, float %f2, double %threshold) local_unnamed_addr #0 {
entry:
  %sub = fsub fast float %f1, %f2
  %0 = tail call fast float @llvm.fabs.f32(float %sub)
  %1 = fpext float %0 to double
  %cmp = fcmp fast ule double %1, %threshold
  ret i1 %cmp
}

; Function Attrs: noinline norecurse nounwind uwtable
define dso_local double @compute_loops(double* nocapture %a) #1 {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  store double 0.000000e+00, double* %a, align 8, !tbaa !2
  ret double %add

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %sumsq.012 = phi double [ 0.000000e+00, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds double, double* %a, i64 %indvars.iv
  %0 = load double, double* %arrayidx, align 8, !tbaa !2
  %mul = fmul fast double %0, %0
  %add = fadd fast double %mul, %sumsq.012
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 100
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #2

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #2

; Function Attrs: nounwind uwtable
define dso_local i32 @main(i32 %argc, i8** nocapture readnone %argv) local_unnamed_addr #3 {
entry:
  %dat = alloca [100 x double], align 16
  %ddat = alloca [100 x double], align 16
  %0 = bitcast [100 x double]* %ddat to i8*
  %1 = bitcast [100 x double]* %dat to i8*
  call void @llvm.lifetime.start.p0i8(i64 800, i8* nonnull %1) #7
  call void @llvm.lifetime.start.p0i8(i64 800, i8* nonnull %0) #7
  call void @llvm.memset.p0i8.i64(i8* nonnull align 16 %0, i8 0, i64 800, i1 false)
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  %arraydecay = getelementptr inbounds [100 x double], [100 x double]* %dat, i64 0, i64 0
  %arraydecay3 = getelementptr inbounds [100 x double], [100 x double]* %ddat, i64 0, i64 0
  %call = call fast double @__enzyme_autodiff(i8* bitcast (double (double*)* @compute_loops to i8*), double* nonnull %arraydecay, double* nonnull %arraydecay3) #7
  br label %for.body8

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv29 = phi i64 [ 0, %entry ], [ %indvars.iv.next30, %for.body ]
  %arrayidx = getelementptr inbounds [100 x double], [100 x double]* %dat, i64 0, i64 %indvars.iv29
  store double 2.000000e+00, double* %arrayidx, align 8, !tbaa !2
  %indvars.iv.next30 = add nuw nsw i64 %indvars.iv29, 1
  %exitcond31 = icmp eq i64 %indvars.iv.next30, 100
  br i1 %exitcond31, label %for.cond.cleanup, label %for.body

for.cond.cleanup7:                                ; preds = %for.body8
  call void @llvm.lifetime.end.p0i8(i64 800, i8* nonnull %0) #7
  call void @llvm.lifetime.end.p0i8(i64 800, i8* nonnull %1) #7
  ret i32 0

for.body8:                                        ; preds = %for.body8, %for.cond.cleanup
  %indvars.iv = phi i64 [ 0, %for.cond.cleanup ], [ %indvars.iv.next, %for.body8 ]
  %arrayidx10 = getelementptr inbounds [100 x double], [100 x double]* %dat, i64 0, i64 %indvars.iv
  %2 = load double, double* %arrayidx10, align 8, !tbaa !2
  %arrayidx12 = getelementptr inbounds [100 x double], [100 x double]* %ddat, i64 0, i64 %indvars.iv
  %3 = load double, double* %arrayidx12, align 8, !tbaa !2
  %4 = trunc i64 %indvars.iv to i32
  %call13 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([24 x i8], [24 x i8]* @.str, i64 0, i64 0), i32 %4, double %2, i32 %4, double %3)
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 100
  br i1 %exitcond, label %for.cond.cleanup7, label %for.body8
}

declare dso_local double @__enzyme_autodiff(i8*, double*, double*) local_unnamed_addr #4

; Function Attrs: nounwind
declare dso_local i32 @printf(i8* nocapture readonly, ...) local_unnamed_addr #5

; Function Attrs: nounwind readnone speculatable
declare float @llvm.fabs.f32(float) #6

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1) #2

attributes #0 = { nounwind readnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { noinline norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #2 = { argmemonly nounwind }
attributes #3 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #4 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #5 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #6 = { nounwind readnone speculatable }
attributes #7 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0}
!3 = !{!"double", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}

; CHECK: define internal void @diffecompute_loops(double* nocapture %a, double* nocapture %"a'", double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %malloccall = tail call noalias nonnull dereferenceable(800) dereferenceable_or_null(800) i8* @malloc(i64 800)
; CHECK-NEXT:   %_malloccache = bitcast i8* %malloccall to double*
; CHECK-NEXT:   %0 = bitcast double* %a to i8*
; CHECK-NEXT:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 %malloccall, i8* nonnull align 8 %0, i64 800, i1 false)
; CHECK-NEXT:   store double 0.000000e+00, double* %a, align 8, !tbaa !2
; CHECK-NEXT:   store double 0.000000e+00, double* %"a'", align 8
; CHECK-NEXT:   br label %invertfor.body

; CHECK: invertentry:                                      ; preds = %invertfor.body
; CHECK-NEXT:   tail call void @free(i8* nonnull %malloccall)
; CHECK-NEXT:   ret void

; CHECK: invertfor.body:                                   ; preds = %incinvertfor.body, %entry
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ 99, %entry ], [ %[[inc:.+]], %incinvertfor.body ]
; CHECK-NEXT:   %[[gep:.+]] = getelementptr inbounds double, double* %_malloccache, i64 %"iv'ac.0"
; CHECK-NEXT:   %[[ld:.+]] = load double, double* %[[gep]], align 8, !invariant.group !
; CHECK-NEXT:   %m0diffe = fmul fast double %differeturn, %[[ld]]
; CHECK-NEXT:   %m1diffe = fmul fast double %differeturn, %[[ld]]
; CHECK-NEXT:   %[[add:.+]] = fadd fast double %m0diffe, %m1diffe
; CHECK-NEXT:   %"arrayidx'ipg_unwrap" = getelementptr inbounds double, double* %"a'", i64 %"iv'ac.0"
; CHECK-NEXT:   %[[pre:.+]] = load double, double* %"arrayidx'ipg_unwrap", align 8
; CHECK-NEXT:   %[[post:.+]] = fadd fast double %[[pre]], %[[add]]
; CHECK-NEXT:   store double %[[post]], double* %"arrayidx'ipg_unwrap", align 8
; CHECK-NEXT:   %[[cmp:.+]] = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %[[cmp]], label %invertentry, label %incinvertfor.body

; CHECK: incinvertfor.body:                                ; preds = %invertfor.body
; CHECK-NEXT:   %[[inc]] = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertfor.body
; CHECK-NEXT: }
