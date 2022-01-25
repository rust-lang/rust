; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -gvn -simplifycfg -correlated-propagation -instsimplify -early-cse -adce -S | FileCheck %s

; this test checks that no malloc needs to be made for the code

; ModuleID = '/mnt/pci4/wmdata/Enzyme/enzyme/test/Integration/fwdsolve.c'
source_filename = "/mnt/pci4/wmdata/Enzyme/enzyme/test/Integration/fwdsolve.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nofree noinline norecurse nounwind uwtable
define void @forward_sub(i64 %N, double* noalias nocapture readonly %L, double* noalias nocapture readonly %b, double* noalias nocapture %out) {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %if.end
  %indvars.iv = phi i64 [ %indvars.iv.next, %if.end ], [ -1, %entry ]
  %i.037 = phi i64 [ %inc16, %if.end ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds double, double* %b, i64 %i.037
  %0 = load double, double* %arrayidx, align 8, !tbaa !2
  %cmp2 = icmp ugt i64 %i.037, 1
  br i1 %cmp2, label %for.body8.lr.ph, label %if.end

for.body8.lr.ph:                                  ; preds = %for.body
  %mul = mul i64 %i.037, %N
  br label %for.body8

for.body8:                                        ; preds = %for.body8, %for.body8.lr.ph
  %j.035 = phi i64 [ 0, %for.body8.lr.ph ], [ %inc, %for.body8 ]
  %tmp.034 = phi double [ %0, %for.body8.lr.ph ], [ %sub13, %for.body8 ]
  %add = add i64 %j.035, %mul
  %arrayidx10 = getelementptr inbounds double, double* %L, i64 %add
  %1 = load double, double* %arrayidx10, align 8, !tbaa !2
  %arrayidx11 = getelementptr inbounds double, double* %out, i64 %j.035
  %2 = load double, double* %arrayidx11, align 8, !tbaa !2
  %mul12 = fmul double %1, %2
  %sub13 = fsub double %tmp.034, %mul12
  %inc = add nuw i64 %j.035, 1
  %exitcond = icmp eq i64 %inc, %indvars.iv
  br i1 %exitcond, label %if.end, label %for.body8

if.end:                                           ; preds = %for.body8, %for.body
  %tmp.1 = phi double [ %0, %for.body ], [ %sub13, %for.body8 ]
  %arrayidx14 = getelementptr inbounds double, double* %out, i64 %i.037
  store double %tmp.1, double* %arrayidx14, align 8, !tbaa !2
  %inc16 = add nuw i64 %i.037, 1
  %cmp = icmp ult i64 %inc16, %N
  %indvars.iv.next = add i64 %indvars.iv, 1
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:
  ret void
}

; Function Attrs: nounwind uwtable
define void @caller(i64 %N, double* nonnull %L, double* nonnull %dL, double* nonnull %b, double* nonnull %dB, double* nonnull %out, double* nonnull %dout) {
entry:
  call void (i8*, ...) @__enzyme_autodiff(i8* bitcast (void (i64, double*, double*, double*)* @forward_sub to i8*), i64 %N, double* nonnull %L, double* nonnull %dL, double* nonnull %b, double* nonnull %dB, double* nonnull %out, double* nonnull %dout) #7
  ret void
}

declare dso_local void @__enzyme_autodiff(i8*, ...)

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 10.0.1 (git@github.com:llvm/llvm-project ef32c611aa214dea855364efd7ba451ec5ec3f74)"}
!2 = !{!3, !3, i64 0}
!3 = !{!"double", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
!6 = !{!7, !7, i64 0}
!7 = !{!"any pointer", !4, i64 0}

; CHECK: define internal void @diffeforward_sub(i64 %N, double* noalias nocapture readonly %L, double* nocapture %"L'", double* noalias nocapture readonly %b, double* nocapture %"b'", double* noalias nocapture %out, double* nocapture %"out'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = add i64 %N, -1
; CHECK-NEXT:   br label %for.body

; CHECK: for.body:                                         ; preds = %if.end, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %if.end ], [ 0, %entry ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %[[i1:.+]] = add i64 %iv, -1

; CHECK-NEXT:   %arrayidx = getelementptr inbounds double, double* %b, i64 %iv
; CHECK-NEXT:   %2 = load double, double* %arrayidx, align 8, !tbaa !2
; CHECK-NEXT:   %cmp2 = icmp ugt i64 %iv, 1
; CHECK-NEXT:   br i1 %cmp2, label %for.body8.lr.ph, label %if.end

; CHECK: for.body8.lr.ph:                                  ; preds = %for.body
; CHECK-NEXT:   %mul = mul i64 %iv, %N
; CHECK-NEXT:   br label %for.body8

; CHECK: for.body8:                                        ; preds = %for.body8, %for.body8.lr.ph
; CHECK-NEXT:   %iv1 = phi i64 [ %iv.next2, %for.body8 ], [ 0, %for.body8.lr.ph ]
; CHECK-NEXT:   %tmp.034 = phi double [ %2, %for.body8.lr.ph ], [ %sub13, %for.body8 ]
; CHECK-NEXT:   %iv.next2 = add nuw nsw i64 %iv1, 1
; CHECK-NEXT:   %add = add i64 %iv1, %mul
; CHECK-NEXT:   %arrayidx10 = getelementptr inbounds double, double* %L, i64 %add
; CHECK-NEXT:   %3 = load double, double* %arrayidx10, align 8, !tbaa !2, !invariant.group !6
; CHECK-NEXT:   %arrayidx11 = getelementptr inbounds double, double* %out, i64 %iv1
; CHECK-NEXT:   %4 = load double, double* %arrayidx11, align 8, !tbaa !2, !invariant.group !7
; CHECK-NEXT:   %mul12 = fmul double %3, %4
; CHECK-NEXT:   %sub13 = fsub double %tmp.034, %mul12
; CHECK-NEXT:   %exitcond = icmp eq i64 %iv.next2, %[[i1]]
; CHECK-NEXT:   br i1 %exitcond, label %if.end, label %for.body8

; CHECK: if.end:                                           ; preds = %for.body8, %for.body
; CHECK-NEXT:   %tmp.1 = phi double [ %2, %for.body ], [ %sub13, %for.body8 ]
; CHECK-NEXT:   %arrayidx14 = getelementptr inbounds double, double* %out, i64 %iv
; CHECK-NEXT:   store double %tmp.1, double* %arrayidx14, align 8, !tbaa !2
; CHECK-NEXT:   %cmp = icmp ne i64 %iv.next, %N
; CHECK-NEXT:   br i1 %cmp, label %for.body, label %invertif.end

; CHECK: invertentry:                                      ; preds = %invertfor.body
; CHECK-NEXT:   ret void

; CHECK: invertfor.body:                                   ; preds = %invertfor.body8, %invertif.end
; CHECK-NEXT:   %"'de3.0" = phi double [ %"'de3.2", %invertif.end ], [ 0.000000e+00, %invertfor.body8 ]
; CHECK-NEXT:   %"'de2.0" = phi double [ %"'de2.2", %invertif.end ], [ 0.000000e+00, %invertfor.body8 ]
; CHECK-NEXT:   %"mul12'de.0" = phi double [ %"mul12'de.2", %invertif.end ], [ 0.000000e+00, %invertfor.body8 ]
; CHECK-NEXT:   %"tmp.034'de.0" = phi double [ %"tmp.034'de.2", %invertif.end ], [ 0.000000e+00, %invertfor.body8 ]
; CHECK-NEXT:   %"sub13'de.0" = phi double [ %"sub13'de.2", %invertif.end ], [ 0.000000e+00, %invertfor.body8 ]
; CHECK-NEXT:   %"'de.0" = phi double [ %[[i20:.+]], %invertif.end ], [ %[[i10:.+]], %invertfor.body8 ]
; CHECK-NEXT:   %"arrayidx'ipg_unwrap" = getelementptr inbounds double, double* %"b'", i64 %"iv'ac.0"
; CHECK-NEXT:   %5 = load double, double* %"arrayidx'ipg_unwrap", align 8
; CHECK-NEXT:   %6 = fadd fast double %5, %"'de.0"
; CHECK-NEXT:   store double %6, double* %"arrayidx'ipg_unwrap", align 8
; CHECK-NEXT:   %7 = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %7, label %invertentry, label %incinvertfor.body

; CHECK: incinvertfor.body:                                ; preds = %invertfor.body
; CHECK-NEXT:   %8 = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertif.end

; CHECK: invertfor.body8:                                  ; preds = %invertif.end.loopexit, %incinvertfor.body8
; CHECK-NEXT:   %"'de3.1" = phi double [ %"'de3.2", %invertif.end.loopexit ], [ 0.000000e+00, %incinvertfor.body8 ]
; CHECK-NEXT:   %"'de2.1" = phi double [ %"'de2.2", %invertif.end.loopexit ], [ 0.000000e+00, %incinvertfor.body8 ]
; CHECK-NEXT:   %"mul12'de.1" = phi double [ %"mul12'de.2", %invertif.end.loopexit ], [ 0.000000e+00, %incinvertfor.body8 ]
; CHECK-NEXT:   %"tmp.034'de.1" = phi double [ %"tmp.034'de.2", %invertif.end.loopexit ], [ 0.000000e+00, %incinvertfor.body8 ]
; CHECK-NEXT:   %"sub13'de.1" = phi double [ %[[i21:.+]], %invertif.end.loopexit ], [ %[[i10:.+]], %incinvertfor.body8 ]
; CHECK-NEXT:   %"iv1'ac.0" = phi i64 [ %[[_unwrap7:.+]], %invertif.end.loopexit ], [ %[[i19:.+]], %incinvertfor.body8 ]
; :   %[[i9:.+]] = fneg fast double %"sub13'de.1"
; CHECK-DAG:   %[[i10]] = fadd fast double %"tmp.034'de.1", %"sub13'de.1"
; CHECK-NEXT:   %[[i11:.+]] = {{(fadd|fsub)}} fast double %"mul12'de.1"
; , %[[i9]]
; CHECK-NEXT:   %arrayidx11_unwrap = getelementptr inbounds double, double* %out, i64 %"iv1'ac.0"
; CHECK-NEXT:   %_unwrap = load double, double* %arrayidx11_unwrap, align 8, !tbaa !2, !invariant.group !7
; CHECK-NEXT:   %m0diffe = fmul fast double %[[i11]], %_unwrap
; CHECK-NEXT:   %mul_unwrap = mul i64 %"iv'ac.0", %N
; CHECK-NEXT:   %add_unwrap = add i64 %"iv1'ac.0", %mul_unwrap
; CHECK-NEXT:   %arrayidx10_unwrap = getelementptr inbounds double, double* %L, i64 %add_unwrap
; CHECK-NEXT:   %[[_unwrap3:.+]] = load double, double* %arrayidx10_unwrap, align 8, !tbaa !2, !invariant.group !6
; CHECK-NEXT:   %m1diffe = fmul fast double %[[i11]], %[[_unwrap3]]
; CHECK-NEXT:   %[[i12:.+]] = fadd fast double %"'de2.1", %m0diffe
; CHECK-NEXT:   %[[i13:.+]] = fadd fast double %"'de3.1", %m1diffe
; CHECK-NEXT:   %"arrayidx11'ipg_unwrap" = getelementptr inbounds double, double* %"out'", i64 %"iv1'ac.0"
; CHECK-NEXT:   %[[i14:.+]] = load double, double* %"arrayidx11'ipg_unwrap", align 8
; CHECK-NEXT:   %[[i15:.+]] = fadd fast double %[[i14]], %[[i13]]
; CHECK-NEXT:   store double %[[i15]], double* %"arrayidx11'ipg_unwrap", align 8
; CHECK-NEXT:   %"arrayidx10'ipg_unwrap" = getelementptr inbounds double, double* %"L'", i64 %add_unwrap
; CHECK-NEXT:   %[[i16:.+]] = load double, double* %"arrayidx10'ipg_unwrap", align 8
; CHECK-NEXT:   %[[i17:.+]] = fadd fast double %[[i16]], %[[i12]]
; CHECK-NEXT:   store double %[[i17]], double* %"arrayidx10'ipg_unwrap", align 8
; CHECK-NEXT:   %[[i18:.+]] = icmp eq i64 %"iv1'ac.0", 0
; CHECK-NEXT:   br i1 %[[i18]], label %invertfor.body, label %incinvertfor.body8

; CHECK: incinvertfor.body8:                               ; preds = %invertfor.body8
; CHECK-NEXT:   %[[i19]] = add nsw i64 %"iv1'ac.0", -1
; CHECK-NEXT:   br label %invertfor.body8

; CHECK: invertif.end.loopexit:                            ; preds = %invertif.end
; CHECK-NEXT:   %[[_unwrap7]] = add i64 %"iv'ac.0", -2
; CHECK-NEXT:   br label %invertfor.body8

; CHECK: invertif.end:                                     ; preds = %if.end, %incinvertfor.body
; CHECK-NEXT:   %"'de3.2" = phi double [ %"'de3.0", %incinvertfor.body ], [ 0.000000e+00, %if.end ]
; CHECK-NEXT:   %"'de2.2" = phi double [ %"'de2.0", %incinvertfor.body ], [ 0.000000e+00, %if.end ]
; CHECK-NEXT:   %"mul12'de.2" = phi double [ %"mul12'de.0", %incinvertfor.body ], [ 0.000000e+00, %if.end ]
; CHECK-NEXT:   %"tmp.034'de.2" = phi double [ %"tmp.034'de.0", %incinvertfor.body ], [ 0.000000e+00, %if.end ]
; CHECK-NEXT:   %"sub13'de.2" = phi double [ %"sub13'de.0", %incinvertfor.body ], [ 0.000000e+00, %if.end ]
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %8, %incinvertfor.body ], [ %0, %if.end ]
; CHECK-NEXT:   %"arrayidx14'ipg_unwrap" = getelementptr inbounds double, double* %"out'", i64 %"iv'ac.0"
; CHECK-NEXT:   %[[i20]] = load double, double* %"arrayidx14'ipg_unwrap", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"arrayidx14'ipg_unwrap", align 8
; CHECK-NEXT:   %cmp2_unwrap = icmp ugt i64 %"iv'ac.0", 1
; CHECK-NEXT:   %[[i21]] = fadd fast double %"sub13'de.2", %[[i20]]
; CHECK-NEXT:   br i1 %cmp2_unwrap, label %invertif.end.loopexit, label %invertfor.body
; CHECK-NEXT: }
