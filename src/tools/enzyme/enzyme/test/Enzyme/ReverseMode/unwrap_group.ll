; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

define dso_local void @square(double* %arg, i32* noalias nocapture readonly %arg1) {
  %tmp = icmp eq double* %arg, null
  br i1 %tmp, label %bb12, label %bb2

bb2:                                              ; preds = %bb
  %tmp3 = load i32, i32* %arg1, align 4
  %tmp4 = sitofp i32 %tmp3 to double
  br label %bb5

bb5:                                              ; preds = %bb5, %bb2
  %tmp6 = phi i64 [ 0, %bb2 ], [ %tmp10, %bb5 ]
  %tmp7 = getelementptr inbounds double, double* %arg, i64 %tmp6
  %tmp8 = load double, double* %tmp7, align 8
  %tmp9 = fmul double %tmp8, %tmp4
  store double %tmp9, double* %tmp7, align 8
  %tmp10 = add nuw nsw i64 %tmp6, 1
  %tmp11 = icmp eq i64 %tmp10, 200
  br i1 %tmp11, label %bb12, label %bb5

bb12:                                             ; preds = %bb5, %bb
  store double 0.000000e+00, double* %arg, align 8
  ret void
}

define dso_local void @dsquare(double* %arg, double* %arg1, i32* %arg2) {
  tail call void @__enzyme_autodiff(i8* bitcast (void (double*, i32*)* @square to i8*), double* %arg, double* %arg1, i32* %arg2)
  ret void
}

declare dso_local void @__enzyme_autodiff(i8*, double*, double*, i32*) local_unnamed_addr

; CHECK: define internal void @diffesquare(double* %arg, double* %"arg'", i32* noalias nocapture readonly %arg1) 
; CHECK-NEXT:   %tmp = icmp eq double* %arg, null
; CHECK-NEXT:   br i1 %tmp, label %invert.critedge, label %bb2

; CHECK: bb2:                                              ; preds = %0
; CHECK-NEXT:   %tmp3 = load i32, i32* %arg1, align 4, !invariant.group ![[INVG:[0-9]+]]
; CHECK-NEXT:   %tmp4 = sitofp i32 %tmp3 to double
; CHECK-NEXT:   br label %bb5

; CHECK: bb5:                                              ; preds = %bb5, %bb2
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %bb5 ], [ 0, %bb2 ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %tmp7 = getelementptr inbounds double, double* %arg, i64 %iv
; CHECK-NEXT:   %tmp8 = load double, double* %tmp7, align 8
; CHECK-NEXT:   %tmp9 = fmul double %tmp8, %tmp4
; CHECK-NEXT:   store double %tmp9, double* %tmp7, align 8, !alias.scope !1, !noalias !4
; CHECK-NEXT:   %tmp11 = icmp eq i64 %iv.next, 200
; CHECK-NEXT:   br i1 %tmp11, label %bb12, label %bb5

; CHECK: bb12:                                             ; preds = %bb5
; CHECK-NEXT:   store double 0.000000e+00, double* %arg, align 8, !alias.scope !6, !noalias !9
; CHECK-NEXT:   store double 0.000000e+00, double* %"arg'", align 8, !alias.scope !9, !noalias !6
; CHECK-NEXT:   br i1 %tmp, label %invert, label %invertbb5

; CHECK: invert.critedge:                                  ; preds = %0
; CHECK-NEXT:   store double 0.000000e+00, double* %arg, align 8, !alias.scope !6, !noalias !9
; CHECK-NEXT:   store double 0.000000e+00, double* %"arg'", align 8, !alias.scope !9, !noalias !6
; CHECK-NEXT:   br label %invert

; CHECK: invert:                                           ; preds = %invert.critedge, %invertbb5, %bb12
; CHECK-NEXT:   ret void

; CHECK: invertbb5:                                        ; preds = %bb12, %incinvertbb5
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %5, %incinvertbb5 ], [ 199, %bb12 ]
; CHECK-NEXT:   %"tmp7'ipg_unwrap" = getelementptr inbounds double, double* %"arg'", i64 %"iv'ac.0"
; CHECK-NEXT:   %1 = load double, double* %"tmp7'ipg_unwrap", align 8, !noalias !11
; CHECK-NEXT:   store double 0.000000e+00, double* %"tmp7'ipg_unwrap", align 8, !alias.scope !4, !noalias !1
; CHECK-NEXT:   %tmp3_unwrap = load i32, i32* %arg1, align 4, !invariant.group ![[INVG]]
; CHECK-NEXT:   %tmp4_unwrap = sitofp i32 %tmp3_unwrap to double
; CHECK-NEXT:   %m0diffetmp8 = fmul fast double %1, %tmp4_unwrap
; CHECK-NEXT:   %2 = load double, double* %"tmp7'ipg_unwrap", align 8, !alias.scope !4, !noalias !1
; CHECK-NEXT:   %3 = fadd fast double %2, %m0diffetmp8
; CHECK-NEXT:   store double %3, double* %"tmp7'ipg_unwrap", align 8, !alias.scope !4, !noalias !1
; CHECK-NEXT:   %4 = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %4, label %invert, label %incinvertbb5

; CHECK: incinvertbb5:                                     ; preds = %invertbb5
; CHECK-NEXT:   %5 = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertbb5
; CHECK-NEXT: }
