; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -simplifycfg -instcombine -gvn -adce -S | FileCheck %s
source_filename = "rm.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind readonly uwtable
define dso_local double @foo(double* nocapture readonly %x) local_unnamed_addr #0 {
entry:
  %tmp = alloca [20 x double], align 16
  %0 = bitcast [20 x double]* %tmp to i8*
  call void @llvm.lifetime.start.p0i8(i64 160, i8* nonnull %0) #4
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv27 = phi i64 [ 0, %entry ], [ %indvars.iv.next28, %for.body ]
  %arrayidx = getelementptr inbounds double, double* %x, i64 %indvars.iv27
  %1 = load double, double* %arrayidx, align 8, !tbaa !2
  %2 = trunc i64 %indvars.iv27 to i32
  %conv = sitofp i32 %2 to double
  %mul = fmul double %1, %conv
  %arrayidx2 = getelementptr inbounds [20 x double], [20 x double]* %tmp, i64 0, i64 %indvars.iv27
  store double %mul, double* %arrayidx2, align 8, !tbaa !2
  %indvars.iv.next28 = add nuw nsw i64 %indvars.iv27, 1
  %exitcond29 = icmp eq i64 %indvars.iv.next28, 20
  br i1 %exitcond29, label %for.body8, label %for.body

for.cond.cleanup7:                                ; preds = %for.body8
  call void @llvm.lifetime.end.p0i8(i64 160, i8* nonnull %0) #4
  ret double %add

for.body8:                                        ; preds = %for.body, %for.body8
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body8 ], [ 0, %for.body ]
  %sum.024 = phi double [ %add, %for.body8 ], [ 0.000000e+00, %for.body ]
  %arrayidx10 = getelementptr inbounds [20 x double], [20 x double]* %tmp, i64 0, i64 %indvars.iv
  %3 = load double, double* %arrayidx10, align 8, !tbaa !2
  %add = fadd double %sum.024, %3
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 20
  br i1 %exitcond, label %for.cond.cleanup7, label %for.body8
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

; Function Attrs: nounwind uwtable
define dso_local double @square(double* nocapture %x) #2 {
entry:
  %call = tail call double @foo(double* %x)
  store double 0.000000e+00, double* %x, align 8, !tbaa !2
  ret double %call
}

; Function Attrs: nounwind uwtable
define dso_local double @dsquare(double %x) local_unnamed_addr #2 {
entry:
  %x.addr = alloca double, align 8
  store double %x, double* %x.addr, align 8, !tbaa !2
  %call = call double (i8*, ...) @__enzyme_autodiff(i8* bitcast (double (double*)* @square to i8*), double* nonnull %x.addr, double* nonnull %x.addr) #4
  ret double %call
}

declare dso_local double @__enzyme_autodiff(i8*, ...) local_unnamed_addr #3

attributes #0 = { noinline nounwind readonly uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0}
!3 = !{!"double", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}

; CHECK: define internal void @diffefoo(double* nocapture readonly %x, double* nocapture %"x'", double %differeturn)
; CHECK-NEXT: entry:
; TODO fix realignment
; CHECK-NEXT:   %"tmp'ai" = alloca [20 x double], align 16
; CHECK-NEXT:   %0 = bitcast [20 x double]* %"tmp'ai" to i8*
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* {{(noundef )?}}nonnull align 16 dereferenceable(160) dereferenceable_or_null(160) %0, i8 0, i64 160, i1 false)
; CHECK-NEXT:   br label %for.body

; CHECK: for.body:                                         ; preds = %for.body, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %exitcond29 = icmp eq i64 %iv.next, 20
; CHECK-NEXT:   br i1 %exitcond29, label %for.body8, label %for.body

; CHECK: for.body8:                                        ; preds = %for.body, %for.body8
; CHECK-NEXT:   %iv1 = phi i64 [ %iv.next2, %for.body8 ], [ 0, %for.body ]
; CHECK-NEXT:   %iv.next2 = add nuw nsw i64 %iv1, 1
; CHECK-NEXT:   %exitcond = icmp eq i64 %iv.next2, 20
; CHECK-NEXT:   br i1 %exitcond, label %invertfor.body8, label %for.body8

; CHECK: invertentry:                                      ; preds = %invertfor.body
; CHECK-NEXT:   ret void

; CHECK: invertfor.body:                                   ; preds = %invertfor.body8, %incinvertfor.body
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %[[i4:.+]], %incinvertfor.body ], [ 19, %invertfor.body8 ]
; CHECK-NEXT:   %"arrayidx2'ipg_unwrap" = getelementptr inbounds [20 x double], [20 x double]* %"tmp'ai", i64 0, i64 %"iv'ac.0"
; CHECK-NEXT:   %[[i0:.+]] = load double, double* %"arrayidx2'ipg_unwrap", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"arrayidx2'ipg_unwrap", align 8
; CHECK-NEXT:   %_unwrap = trunc i64 %"iv'ac.0" to i32
; CHECK-NEXT:   %conv_unwrap = sitofp i32 %_unwrap to double
; CHECK-NEXT:   %m0diffe = fmul fast double %[[i0]], %conv_unwrap
; CHECK-NEXT:   %"arrayidx'ipg_unwrap" = getelementptr inbounds double, double* %"x'", i64 %"iv'ac.0"
; CHECK-NEXT:   %[[i1:.+]] = load double, double* %"arrayidx'ipg_unwrap", align 8
; CHECK-NEXT:   %[[i2:.+]] = fadd fast double %[[i1]], %m0diffe
; CHECK-NEXT:   store double %[[i2]], double* %"arrayidx'ipg_unwrap", align 8
; CHECK-NEXT:   %[[i3:.+]] = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %[[i3:.+]], label %invertentry, label %incinvertfor.body

; CHECK: incinvertfor.body:                                ; preds = %invertfor.body
; CHECK-NEXT:   %[[i4]] = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertfor.body

; CHECK: invertfor.body8:                                  ; preds = %for.body8, %incinvertfor.body8
; CHECK-NEXT:   %"iv1'ac.0" = phi i64 [ %[[i8:.+]], %incinvertfor.body8 ], [ 19, %for.body8 ]
; CHECK-NEXT:   %"arrayidx10'ipg_unwrap" = getelementptr inbounds [20 x double], [20 x double]* %"tmp'ai", i64 0, i64 %"iv1'ac.0"
; CHECK-NEXT:   %[[i5:.+]] = load double, double* %"arrayidx10'ipg_unwrap", align 8
; CHECK-NEXT:   %[[i6:.+]] = fadd fast double %[[i5]], %differeturn
; CHECK-NEXT:   store double %[[i6]], double* %"arrayidx10'ipg_unwrap", align 8
; CHECK-NEXT:   %[[i7:.+]] = icmp eq i64 %"iv1'ac.0", 0
; CHECK-NEXT:   br i1 %[[i7]], label %invertfor.body, label %incinvertfor.body8

; CHECK: incinvertfor.body8:                               ; preds = %invertfor.body8
; CHECK-NEXT:   %[[i8]] = add nsw i64 %"iv1'ac.0", -1
; CHECK-NEXT:   br label %invertfor.body8
; CHECK-NEXT: }

