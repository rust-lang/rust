; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -gvn -simplifycfg -loop-deletion -simplifycfg -instsimplify -adce -S | FileCheck %s

source_filename = "mem.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@.str = private unnamed_addr constant [4 x i8] c"%f\0A\00", align 1

; Function Attrs: nounwind uwtable
define double @infLoop(double %rho0, i64 %numReg) {
entry:
  %cmp3 = icmp ult i64 0, %numReg
  br i1 %cmp3, label %for.body.lr.ph, label %for.end8

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.end
  %r.04 = phi i64 [ 0, %for.body.lr.ph ], [ %inc7, %for.end ]
  %call = call noalias align 16 i8* @calloc(i64 8, i64 1000000) #3
  %i4 = bitcast i8* %call to double*
  store double 1.000000e+00, double* %i4, align 8
  br label %for.body3

for.body3:                                        ; preds = %for.body, %for.body3
  %i.01 = phi i64 [ 1, %for.body ], [ %inc, %for.body3 ]
  %sub = sub i64 %i.01, 1
  %arrayidx4 = getelementptr inbounds double, double* %i4, i64 %sub
  %i10 = load double, double* %arrayidx4, align 8
  %mul = fmul double %i10, %rho0
  %arrayidx5 = getelementptr inbounds double, double* %i4, i64 %i.01
  store double %mul, double* %arrayidx5, align 8
  %inc = add i64 %i.01, 1
  %cmp2 = icmp ult i64 %inc, 1000000
  br i1 %cmp2, label %for.body3, label %for.end, !llvm.loop !4

for.end:                                          ; preds = %for.body3
  call void @free(i8* %call) #3
  %inc7 = add i64 %r.04, 1
  %cmp = icmp ult i64 %inc7, %numReg
  br i1 %cmp, label %for.body, label %for.cond.for.end8_crit_edge, !llvm.loop !6

for.cond.for.end8_crit_edge:                      ; preds = %for.end
  br label %for.end8

for.end8:                                         ; preds = %for.cond.for.end8_crit_edge, %entry
  ret double %rho0
}

; Function Attrs: nounwind
declare dso_local noalias i8* @calloc(i64, i64) #1

; Function Attrs: nounwind
declare dso_local void @free(i8*) #1

; Function Attrs: nounwind uwtable
define dso_local i32 @main() #0 {
entry:
  %call = call double @__enzyme_autodiff(i8* bitcast (double (double, i64)* @infLoop to i8*), double 2.000000e+00, i64 10000000)
  %call1 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str, i64 0, i64 0), double %call)
  ret i32 0
}

declare dso_local i32 @printf(i8*, ...) #2

declare dso_local double @__enzyme_autodiff(i8*, double, i64) #2

attributes #0 = { nounwind uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { nounwind }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"uwtable", i32 1}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{!"clang version 14.0.0 (git@github.com:jdoerfert/llvm-project b5b6dc5cda07dc505cc24f6960980780f3d58f3a)"}
!4 = distinct !{!4, !5}
!5 = !{!"llvm.loop.mustprogress"}
!6 = distinct !{!6, !5}

; CHECK: define internal { double } @diffeinfLoop(double %rho0, i64 %numReg, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %cmp3 = icmp ult i64 0, %numReg
; CHECK-NEXT:   br i1 %cmp3, label %for.body.lr.ph, label %invertentry

; CHECK: for.body.lr.ph:                                   ; preds = %entry
; CHECK-NEXT:   br label %for.body

; CHECK: for.body:                                         ; preds = %for.end, %for.body.lr.ph
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %for.end ], [ 0, %for.body.lr.ph ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %call = call noalias align 16 i8* @calloc(i64 8, i64 1000000) 
; CHECK-NEXT:   %i4 = bitcast i8* %call to double*
; CHECK-NEXT:   store double 1.000000e+00, double* %i4, align 8
; CHECK-NEXT:   br label %for.body3

; CHECK: for.body3:                                        ; preds = %for.body3, %for.body
; CHECK-NEXT:   %i10 = phi double [ %mul, %for.body3 ], [ 1.000000e+00, %for.body ]
; CHECK-NEXT:   %iv1 = phi i64 [ %iv.next2, %for.body3 ], [ 0, %for.body ]
; CHECK-NEXT:   %iv.next2 = add nuw nsw i64 %iv1, 1
; CHECK-NEXT:   %mul = fmul double %i10, %rho0
; CHECK-NEXT:   %arrayidx5 = getelementptr inbounds double, double* %i4, i64 %iv.next2
; CHECK-NEXT:   store double %mul, double* %arrayidx5, align 8
; CHECK-NEXT:   %inc = add i64 %iv.next2, 1
; CHECK-NEXT:   %cmp2 = icmp ult i64 %inc, 1000000
; CHECK-NEXT:   br i1 %cmp2, label %for.body3, label %for.end, !llvm.loop !4

; CHECK: for.end:                                          ; preds = %for.body3
; CHECK-NEXT:   call void @free(i8* %call)
; CHECK-NEXT:   %cmp = icmp ne i64 %iv.next, %numReg
; CHECK-NEXT:   br i1 %cmp, label %for.body, label %for.end8, !llvm.loop !6

; CHECK: for.end8:                                         ; preds = %for.end
; CHECK-NEXT:   br i1 %cmp3, label %invertfor.cond.for.end8_crit_edge, label %invertentry

; CHECK: invertentry:                                      ; preds = %invertfor.body, %entry, %for.end8
; CHECK-NEXT:   %"rho0'de.0" = phi double [ %differeturn, %for.end8 ], [ %differeturn, %entry ], [ %4, %invertfor.body ]
; CHECK-NEXT:   %0 = insertvalue { double } undef, double %"rho0'de.0", 0
; CHECK-NEXT:   ret { double } %0

; CHECK: invertfor.body:                                   ; preds = %invertfor.body3
; CHECK-NEXT:   store double 0.000000e+00, double* %"i4'ipc_unwrap2.phi.trans.insert", align 8
; CHECK-NEXT:   tail call void @free(i8* nonnull %"call'mi")
; CHECK-NEXT:   tail call void @free(i8* %remat_call)
; CHECK-NEXT:   %1 = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %1, label %invertentry, label %incinvertfor.body

; CHECK: incinvertfor.body:                                ; preds = %invertfor.body
; CHECK-NEXT:   %2 = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %remat_enter

; CHECK: invertfor.body3:                                  ; preds = %remat_for.body_for.end, %incinvertfor.body3
; CHECK-NEXT:   %3 = phi double [ 0.000000e+00, %remat_for.body_for.end ], [ %6, %incinvertfor.body3 ]
; CHECK-NEXT:   %"rho0'de.1" = phi double [ %"rho0'de.2", %remat_for.body_for.end ], [ %4, %incinvertfor.body3 ]
; CHECK-NEXT:   %"iv1'ac.0" = phi i64 [ 999998, %remat_for.body_for.end ], [ %8, %incinvertfor.body3 ]
; CHECK-NEXT:   %iv.next2_unwrap = add nuw nsw i64 %"iv1'ac.0", 1
; CHECK-NEXT:   %"arrayidx5'ipg_unwrap" = getelementptr inbounds double, double* %"i4'ipc_unwrap2.phi.trans.insert", i64 %iv.next2_unwrap
; CHECK-NEXT:   store double 0.000000e+00, double* %"arrayidx5'ipg_unwrap", align 8
; CHECK-NEXT:   %m0diffei10 = fmul fast double %3, %rho0
; CHECK-NEXT:   %arrayidx4_unwrap5 = getelementptr inbounds double, double* %i4_unwrap, i64 %"iv1'ac.0"
; CHECK-NEXT:   %i10_unwrap6 = load double, double* %arrayidx4_unwrap5, align 8, !invariant.group !7
; CHECK-NEXT:   %m1differho0 = fmul fast double %3, %i10_unwrap6
; CHECK-NEXT:   %4 = fadd fast double %"rho0'de.1", %m1differho0
; CHECK-NEXT:   %"arrayidx4'ipg_unwrap" = getelementptr inbounds double, double* %"i4'ipc_unwrap2.phi.trans.insert", i64 %"iv1'ac.0"
; CHECK-NEXT:   %5 = load double, double* %"arrayidx4'ipg_unwrap", align 8
; CHECK-NEXT:   %6 = fadd fast double %5, %m0diffei10
; CHECK-NEXT:   store double %6, double* %"arrayidx4'ipg_unwrap", align 8
; CHECK-NEXT:   %7 = icmp eq i64 %"iv1'ac.0", 0
; CHECK-NEXT:   br i1 %7, label %invertfor.body, label %incinvertfor.body3

; CHECK: incinvertfor.body3:                               ; preds = %invertfor.body3
; CHECK-NEXT:   %8 = add nsw i64 %"iv1'ac.0", -1
; CHECK-NEXT:   br label %invertfor.body3

; CHECK: invertfor.cond.for.end8_crit_edge:                ; preds = %for.end8
; CHECK-NEXT:   %_unwrap = add i64 %numReg, -1
; CHECK-NEXT:   br label %remat_enter

; CHECK: remat_enter:                                      ; preds = %invertfor.cond.for.end8_crit_edge, %incinvertfor.body
; CHECK-NEXT:   %"rho0'de.2" = phi double [ %differeturn, %invertfor.cond.for.end8_crit_edge ], [ %4, %incinvertfor.body ]
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %_unwrap, %invertfor.cond.for.end8_crit_edge ], [ %2, %incinvertfor.body ]
; CHECK-NEXT:   %remat_call = call noalias align 16 i8* @calloc(i64 8, i64 1000000)
; CHECK-NEXT:   %"call'mi" = call noalias nonnull align 16 i8* @calloc(i64 8, i64 1000000)
; CHECK-NEXT:   %i4_unwrap = bitcast i8* %remat_call to double*
; CHECK-NEXT:   store double 1.000000e+00, double* %i4_unwrap, align 8
; CHECK-NEXT:   br label %remat_for.body_for.body3

; CHECK: remat_for.body_for.body3:                         ; preds = %remat_for.body_for.body3, %remat_enter
; CHECK-NEXT:   %i10_unwrap = phi double [ %mul_unwrap, %remat_for.body_for.body3 ], [ 1.000000e+00, %remat_enter ]
; CHECK-NEXT:   %fiv = phi i64 [ %9, %remat_for.body_for.body3 ], [ 0, %remat_enter ]
; CHECK-NEXT:   %9 = add i64 %fiv, 1
; CHECK-DAG:    %arrayidx5_unwrap = getelementptr inbounds double, double* %i4_unwrap, i64 %9
; CHECK-DAG:    %mul_unwrap = fmul double %i10_unwrap, %rho0
; CHECK-NEXT:   store double %mul_unwrap, double* %arrayidx5_unwrap, align 8
; CHECK-NEXT:   %inc_unwrap = add i64 %9, 1
; CHECK-NEXT:   %cmp2_unwrap = icmp ult i64 %inc_unwrap, 1000000
; CHECK-NEXT:   br i1 %cmp2_unwrap, label %remat_for.body_for.body3, label %remat_for.body_for.end

; CHECK: remat_for.body_for.end:                           ; preds = %remat_for.body_for.body3
; CHECK-NEXT:   %"i4'ipc_unwrap2.phi.trans.insert" = bitcast i8* %"call'mi" to double*
; CHECK-NEXT:   br label %invertfor.body3
