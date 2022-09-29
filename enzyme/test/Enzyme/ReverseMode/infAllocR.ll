; RUN: if [ %llvmver -lt 15 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -gvn -simplifycfg -loop-deletion -simplifycfg -instsimplify -correlated-propagation -early-cse-memssa  -adce -S | FileCheck %s -check-prefixes LL14,CHECK; fi
; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -gvn -simplifycfg -loop-deletion -simplifycfg -instsimplify -correlated-propagation -early-cse-memssa  -adce -S | FileCheck %s -check-prefixes LL15,CHECK; fi

source_filename = "mem.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@.str = private unnamed_addr constant [4 x i8] c"%f\0A\00", align 1

; Function Attrs: nounwind uwtable
define double @infLoop(double %rho0, i64 %numReg) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.end
  %r.04 = phi i64 [ 0, %entry ], [ %inc7, %for.end ]
  %val = phi double [ %rho0, %entry ], [ %fadd, %for.end ]
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
  %lgep = getelementptr inbounds double, double* %i4, i64 999998
  %ld = load double, double* %lgep, align 8
  call void @free(i8* %call) #3
  %fadd = fadd double %ld, %val
  %inc7 = add i64 %r.04, 1
  %cmp = icmp ult i64 %inc7, %numReg
  br i1 %cmp, label %for.body, label %for.cond.for.end8_crit_edge, !llvm.loop !6

for.cond.for.end8_crit_edge:                      ; preds = %for.end
  br label %for.end8

for.end8:                                         ; preds = %for.cond.for.end8_crit_edge, %entry
  ret double %fadd
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

declare dso_local i32 @printf(i8*, ...)

declare dso_local double @__enzyme_autodiff(i8*, double, i64)

attributes #0 = { nounwind uwtable }
attributes #1 = { nounwind }
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
; CHECK-NEXT:   %0 = add i64 %numReg, -1
; CHECK-NEXT:   br label %for.body

; CHECK: for.body:  
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %for.end ], [ 0, %entry ]
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
; CHECK-NEXT:   %inc = add {{(nuw nsw )?}}i64 %iv.next2, 1
; CHECK-NEXT:   %cmp2 = icmp ult i64 %inc, 1000000
; CHECK-NEXT:   br i1 %cmp2, label %for.body3, label %for.end, !llvm.loop !4

; CHECK: for.end:                                          ; preds = %for.body3
; CHECK-NEXT:   call void @free(i8* nonnull %call)
; CHECK-NEXT:   %cmp = icmp ne i64 %iv.next, %numReg
; CHECK-NEXT:   br i1 %cmp, label %for.body, label %remat_enter, !llvm.loop !6

; CHECK: invertentry:
; CHECK-NEXT:   %[[p1:.+]] = insertvalue { double } undef, double %[[i4:.+]], 0
; CHECK-NEXT:   ret { double } %[[p1]]

; CHECK: invertfor.body:                                   ; preds = %invertfor.body3
; CHECK-NEXT:   store double 0.000000e+00, double* %"i4'ipc_unwrap8", align 8
; CHECK-NEXT:   tail call void @free(i8* nonnull %"call'mi")
; CHECK-NEXT:   tail call void @free(i8* nonnull %remat_call)
; CHECK-NEXT:   %[[a1:.+]] = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   %[[i3:.+]] = fadd fast double %8, %differeturn
; CHECK-NEXT:   %[[i4]] = select {{(fast )?}}i1 %[[a1]], double %[[i3]], double %8
; CHECK-NEXT:   br i1 %[[a1]], label %invertentry, label %incinvertfor.body

; CHECK: incinvertfor.body:                                ; preds = %invertfor.body
; CHECK-NEXT:   %[[a2:.+]] = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %remat_enter

; CHECK: invertfor.body3:                                  ; preds = %remat_for.body_for.end, %incinvertfor.body3
; LL14-NEXT:   %[[i6:.+]] = phi double [ 0.000000e+00, %remat_for.body_for.end ], [ %[[i9:.+]], %incinvertfor.body3 ]
; LL15-NEXT:   %[[i6:.+]] = phi double [ %[[pre11:.+]], %remat_for.body_for.end ], [ %[[i9:.+]], %incinvertfor.body3 ]
; LL14-NEXT:   %[[i7:.+]] = phi double [ %differeturn, %remat_for.body_for.end ], [ %[[pre:.+]], %incinvertfor.body3 ]
; LL15-NEXT:   %[[i7:.+]] = phi double [ %[[a14:.+]], %remat_for.body_for.end ], [ %[[pre:.+]], %incinvertfor.body3 ]
; CHECK-NEXT:   %"rho0'de.0" = phi double [ %"rho0'de.1", %remat_for.body_for.end ], [ %[[i8:.+]], %incinvertfor.body3 ]
; CHECK-NEXT:   %"iv1'ac.0" = phi i64 [ 999998, %remat_for.body_for.end ], [ %[[i11:.+]], %incinvertfor.body3 ]
; CHECK-NEXT:   %iv.next2_unwrap = add nuw nsw i64 %"iv1'ac.0", 1
; CHECK-NEXT:   %"arrayidx5'ipg_unwrap" = getelementptr inbounds double, double* %"i4'ipc_unwrap8", i64 %iv.next2_unwrap
; CHECK-NEXT:   store double 0.000000e+00, double* %"arrayidx5'ipg_unwrap", align 8
; CHECK-NEXT:   %m0diffei10 = fmul fast double %[[i6]], %rho0
; CHECK-NEXT:   %[[arrayidx4_unwrap5:.+]] = getelementptr inbounds double, double* %i4_unwrap, i64 %"iv1'ac.0"
; CHECK-NEXT:   %[[i10_unwrap6:.+]] = load double, double* %[[arrayidx4_unwrap5]], align 8, !invariant.group !
; CHECK-NEXT:   %m1differho0 = fmul fast double %[[i6]], %[[i10_unwrap6]]
; CHECK-NEXT:   %[[i8]] = fadd fast double %"rho0'de.0", %m1differho0
; CHECK-NEXT:   %"arrayidx4'ipg_unwrap" = getelementptr inbounds double, double* %"i4'ipc_unwrap8", i64 %"iv1'ac.0"
; CHECK-NEXT:   %[[i9]] = fadd fast double %[[i7]], %m0diffei10
; CHECK-NEXT:   store double %[[i9]], double* %"arrayidx4'ipg_unwrap", align 8
; CHECK-NEXT:   %[[i10:.+]] = icmp eq i64 %"iv1'ac.0", 0
; CHECK-NEXT:   br i1 %[[i10]], label %invertfor.body, label %incinvertfor.body3

; CHECK: incinvertfor.body3:                               ; preds = %invertfor.body3
; CHECK-NEXT:   %[[i11]] = add nsw i64 %"iv1'ac.0", -1
; CHECK-NEXT:   %"arrayidx4'ipg_unwrap.phi.trans.insert" = getelementptr inbounds double, double* %"i4'ipc_unwrap8", i64 %[[i11]]
; CHECK-NEXT:   %[[pre]] = load double, double* %"arrayidx4'ipg_unwrap.phi.trans.insert", align 8
; CHECK-NEXT:   br label %invertfor.body3

; CHECK: remat_enter:  
; CHECK-NEXT:   %"rho0'de.1" = phi double [ %[[i8]], %incinvertfor.body ], [ 0.000000e+00, %for.end ]
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %[[a2]], %incinvertfor.body ], [ %0, %for.end ]
; CHECK-NEXT:   %remat_call = call noalias align 16 i8* @calloc(i64 8, i64 1000000)
; CHECK-NEXT:   %"call'mi" = call noalias nonnull align 16 i8* @calloc(i64 8, i64 1000000)
; CHECK-NEXT:   %i4_unwrap = bitcast i8* %remat_call to double*
; CHECK-NEXT:   store double 1.000000e+00, double* %i4_unwrap, align 8
; CHECK-NEXT:   br label %remat_for.body_for.body3

; CHECK: remat_for.body_for.body3:                         ; preds = %remat_for.body_for.body3, %remat_enter
; CHECK-NEXT:   %i10_unwrap = phi double [ %mul_unwrap, %remat_for.body_for.body3 ], [ 1.000000e+00, %remat_enter ]
; CHECK-NEXT:   %fiv = phi i64 [ %[[p9:.+]], %remat_for.body_for.body3 ], [ 0, %remat_enter ]
; CHECK-NEXT:   %[[p9:.+]] = add {{(nsw )?}}i64 %fiv, 1
; CHECK-DAG:    %arrayidx5_unwrap = getelementptr inbounds double, double* %i4_unwrap, i64 %[[p9]]
; CHECK-DAG:    %mul_unwrap = fmul double %i10_unwrap, %rho0
; CHECK-NEXT:   store double %mul_unwrap, double* %arrayidx5_unwrap, align 8
; CHECK-NEXT:   %inc_unwrap = add {{(nuw nsw )?}}i64 %[[p9]], 1
; CHECK-NEXT:   %cmp2_unwrap = icmp ult i64 %inc_unwrap, 1000000
; CHECK-NEXT:   br i1 %cmp2_unwrap, label %remat_for.body_for.body3, label %remat_for.body_for.end

; CHECK: remat_for.body_for.end:                           ; preds = %remat_for.body_for.body3
; CHECK-NEXT:   %"i4'ipc_unwrap8" = bitcast i8* %"call'mi" to double*
; CHECK-NEXT:   %"lgep'ipg_unwrap" = getelementptr inbounds double, double* %"i4'ipc_unwrap8", i64 999998
; LL14-NEXT:   store double %differeturn, double* %"lgep'ipg_unwrap", align 8

; LL15-NEXT:  %[[a13:.+]] = load double, double* %"lgep'ipg_unwrap", align 8, !alias.scope !10, !noalias !7
; LL15-NEXT:  %[[a14]] = fadd fast double %[[a13]], %differeturn
; LL15-NEXT:  store double %[[a14]], double* %"lgep'ipg_unwrap", align 8

; LL15-NEXT:  %"arrayidx5'ipg_unwrap.phi.trans.insert" = getelementptr inbounds double, double* %"i4'ipc_unwrap8", i64 999999
; LL15-NEXT:  %[[pre11]] = load double, double* %"arrayidx5'ipg_unwrap.phi.trans.insert", align 8

; CHECK-NEXT:   br label %invertfor.body3
