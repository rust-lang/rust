; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -simplifycfg -instcombine -gvn -adce -S | FileCheck %s
source_filename = "rm.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind readonly uwtable
define dso_local double @foo(double* nocapture readonly %x) local_unnamed_addr #0 {
entry:
  %tmp = alloca [20 x i32], align 16
  %0 = bitcast [20 x i32]* %tmp to i8*
  call void @llvm.lifetime.start.p0i8(i64 80, i8* nonnull %0) #4
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv28 = phi i64 [ 0, %entry ], [ %indvars.iv.next29, %for.body ]
  %1 = trunc i64 %indvars.iv28 to i32
  %mul = mul nsw i32 %1, %1
  %arrayidx = getelementptr inbounds [20 x i32], [20 x i32]* %tmp, i64 0, i64 %indvars.iv28
  store i32 %mul, i32* %arrayidx, align 4, !tbaa !2
  %indvars.iv.next29 = add nuw nsw i64 %indvars.iv28, 1
  %exitcond30 = icmp eq i64 %indvars.iv.next29, 20
  br i1 %exitcond30, label %for.body5, label %for.body

for.cond.cleanup4:                                ; preds = %for.body5
  call void @llvm.lifetime.end.p0i8(i64 80, i8* nonnull %0) #4
  ret double %add

for.body5:                                        ; preds = %for.body, %for.body5
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body5 ], [ 0, %for.body ]
  %sum.025 = phi double [ %add, %for.body5 ], [ 0.000000e+00, %for.body ]
  %arrayidx7 = getelementptr inbounds double, double* %x, i64 %indvars.iv
  %2 = load double, double* %arrayidx7, align 8, !tbaa !6
  %arrayidx9 = getelementptr inbounds [20 x i32], [20 x i32]* %tmp, i64 0, i64 %indvars.iv
  %3 = load i32, i32* %arrayidx9, align 4, !tbaa !2
  %conv = sitofp i32 %3 to double
  %mul10 = fmul double %2, %conv
  %add = fadd double %sum.025, %mul10
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 20
  br i1 %exitcond, label %for.cond.cleanup4, label %for.body5
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

; Function Attrs: nounwind uwtable
define dso_local double @square(double* nocapture %x) #2 {
entry:
  %call = tail call double @foo(double* %x)
  store double 0.000000e+00, double* %x, align 8, !tbaa !6
  ret double %call
}

; Function Attrs: nounwind uwtable
define dso_local double @dsquare(double %x) local_unnamed_addr #2 {
entry:
  %x.addr = alloca double, align 8
  store double %x, double* %x.addr, align 8, !tbaa !6
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
!3 = !{!"int", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
!6 = !{!7, !7, i64 0}
!7 = !{!"double", !4, i64 0}


; CHECK: define internal void @diffefoo(double* nocapture readonly %x, double* nocapture %"x'", double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %tmp = alloca [20 x i32], align 1
; CHECK-NEXT:   br label %for.body

; CHECK: for.body:                                         ; preds = %for.body, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %[[a1:.+]] = trunc i64 %iv to i32
; CHECK-NEXT:   %mul = mul nsw i32 %[[a1]], %[[a1]]
; CHECK-NEXT:   %arrayidx = getelementptr inbounds [20 x i32], [20 x i32]* %tmp, i64 0, i64 %iv
; CHECK-NEXT:   store i32 %mul, i32* %arrayidx, align 4, !tbaa ![[itbaa:[0-9]+]]
; CHECK-NEXT:   %exitcond30 = icmp eq i64 %iv.next, 20
; CHECK-NEXT:   br i1 %exitcond30, label %for.body5, label %for.body

; CHECK: for.body5:                                        ; preds = %for.body, %for.body5
; CHECK-NEXT:   %iv1 = phi i64 [ %iv.next2, %for.body5 ], [ 0, %for.body ]
; CHECK-NEXT:   %iv.next2 = add nuw nsw i64 %iv1, 1
; CHECK-NEXT:   %exitcond = icmp eq i64 %iv.next2, 20
; CHECK-NEXT:   br i1 %exitcond, label %invertfor.body5, label %for.body5

; CHECK: invertentry:                                      ; preds = %invertfor.body
; CHECK-NEXT:   ret void

; CHECK: invertfor.body:                                   ; preds = %invertfor.body5, %incinvertfor.body
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %[[a3:.+]], %incinvertfor.body ], [ 19, %invertfor.body5 ]
; CHECK-NEXT:   %[[a2:.+]] = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %[[a2]], label %invertentry, label %incinvertfor.body

; CHECK: incinvertfor.body:                                ; preds = %invertfor.body
; CHECK-NEXT:   %[[a3]] = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertfor.body

; CHECK: invertfor.body5:                                  ; preds = %for.body5, %incinvertfor.body5
; CHECK-NEXT:   %"iv1'ac.0" = phi i64 [ %[[a7:.+]], %incinvertfor.body5 ], [ 19, %for.body5 ]
; CHECK-NEXT:   %arrayidx9_unwrap = getelementptr inbounds [20 x i32], [20 x i32]* %tmp, i64 0, i64 %"iv1'ac.0"
; CHECK-NEXT:   %_unwrap = load i32, i32* %arrayidx9_unwrap, align 4, !tbaa ![[itbaa]], !invariant.group !
; CHECK-NEXT:   %conv_unwrap = sitofp i32 %_unwrap to double
; CHECK-NEXT:   %m0diffe = fmul fast double %conv_unwrap, %differeturn
; CHECK-NEXT:   %"arrayidx7'ipg_unwrap" = getelementptr inbounds double, double* %"x'", i64 %"iv1'ac.0"
; CHECK-NEXT:   %[[a4:.+]] = load double, double* %"arrayidx7'ipg_unwrap", align 8
; CHECK-NEXT:   %[[a5:.+]] = fadd fast double %[[a4]], %m0diffe
; CHECK-NEXT:   store double %[[a5]], double* %"arrayidx7'ipg_unwrap", align 8
; CHECK-NEXT:   %[[a6:.+]] = icmp eq i64 %"iv1'ac.0", 0
; CHECK-NEXT:   br i1 %[[a6]], label %invertfor.body, label %incinvertfor.body5

; CHECK: incinvertfor.body5:                               ; preds = %invertfor.body5
; CHECK-NEXT:   %[[a7]] = add nsw i64 %"iv1'ac.0", -1
; CHECK-NEXT:   br label %invertfor.body5
; CHECK-NEXT: }
