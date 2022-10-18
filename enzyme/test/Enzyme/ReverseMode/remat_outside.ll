; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -adce -loop-deletion -correlated-propagation -simplifycfg -S | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@enzyme_const = dso_local local_unnamed_addr global i32 0, align 4

declare nonnull i8* @malloc(i64) 

declare void @free(i8*) 

define double @_Z15integrate_imagedi(double %arg, i32 %arg1) {
bb:
  %i7 = zext i32 %arg1 to i64
  %i4 = mul i64 %i7, 8
  br label %bb8

bb8:                                              ; preds = %bb19, %bb
  %i9 = phi double [ 0.000000e+00, %bb ], [ %i13, %bb12 ]
  %i10 = tail call noalias nonnull i8* @malloc(i64 %i4)
  %i11 = bitcast i8* %i10 to double*
  br label %bb14

bb14:                                             ; preds = %bb14, %bb8
  %i15 = phi i64 [ %i17, %bb14 ], [ 0, %bb8 ]
  %i16 = getelementptr inbounds double, double* %i11, i64 %i15
  store double %arg, double* %i16, align 8
  %i17 = add nuw nsw i64 %i15, 1
  %i18 = icmp eq i64 %i17, %i7
  br i1 %i18, label %bb12, label %bb14

bb12:                                             ; preds = %bb14
  %i13 = load double, double* %i11, align 8
  tail call void @free(i8* nonnull %i10) 
  %i21 = fsub double %i13, %i9
  %i22 = fcmp ogt double %i21, 1.000000e-04
  br i1 %i22, label %bb8, label %bb23

bb23:                                             ; preds = %bb19
  ret double %i13
}

define dso_local double @_Z3dondd(double %arg, double %arg1) {
bb:
  %i = load i32, i32* @enzyme_const, align 4
  %i2 = tail call double (double (double, i32)*, ...) @_Z17__enzyme_autodiffPFddiEz(double (double, i32)* nonnull @_Z15integrate_imagedi, i32 %i, double %arg, i32 %i, i32 10)
  ret double %i2
}

declare dso_local double @_Z17__enzyme_autodiffPFddiEz(double (double, i32)*, ...)

; CHECK: define internal void @diffe_Z15integrate_imagedi(double %arg, i32 %arg1, double %differeturn)
; CHECK-NEXT: bb:
; CHECK-NEXT:   %i7 = zext i32 %arg1 to i64
; CHECK-NEXT:   %i4 = mul {{(nuw nsw )?}}i64 %i7, 8
; CHECK-NEXT:   %0 = add {{(nsw )?}}i64 %i7, -1
; CHECK-NEXT:   br label %bb8

; CHECK: bb8:                                              ; preds = %bb12, %bb
; CHECK-NEXT:   %i9 = phi double [ 0.000000e+00, %bb ], [ %i13, %bb12 ]
; CHECK-NEXT:   %i10 = tail call noalias nonnull i8* @malloc(i64 %i4)
; CHECK-NEXT:   %i11 = bitcast i8* %i10 to double*
; CHECK-NEXT:   br label %bb14

; CHECK: bb14:                                             ; preds = %bb14, %bb8
; CHECK-NEXT:   %iv1 = phi i64 [ %iv.next2, %bb14 ], [ 0, %bb8 ]
; CHECK-NEXT:   %iv.next2 = add nuw nsw i64 %iv1, 1
; CHECK-NEXT:   %i16 = getelementptr inbounds double, double* %i11, i64 %iv1
; CHECK-NEXT:   store double %arg, double* %i16, align 8
; CHECK-NEXT:   %i18 = icmp eq i64 %iv.next2, %i7
; CHECK-NEXT:   br i1 %i18, label %bb12, label %bb14

; CHECK: bb12:                                             ; preds = %bb14
; CHECK-NEXT:   %i13 = load double, double* %i11, align 8
; CHECK-NEXT:   tail call void @free(i8* nonnull %i10)
; CHECK-NEXT:   %i21 = fsub double %i13, %i9
; CHECK-NEXT:   %i22 = fcmp ogt double %i21, 1.000000e-04
; CHECK-NEXT:   br i1 %i22, label %bb8, label %[[remat_bb8_bb8:.+]]

; CHECK: [[remat_bb8_bb8]]:                                    ; preds = %bb12
; CHECK:   %remat_i10 = tail call noalias nonnull i8* @malloc(i64 %i4)
; CHECK-NEXT:   br label %remat_bb8_bb14

; CHECK: remat_bb8_bb14: 
; CHECK-NEXT:   %fiv = phi i64 [ %[[i1:.+]], %remat_bb8_bb14 ], [ 0, %[[remat_bb8_bb8]] ]
; CHECK-NEXT:   %[[i1]] = add i64 %fiv, 1
; CHECK-NEXT:   %i11_unwrap = bitcast i8* %remat_i10 to double*
; CHECK-NEXT:   %i16_unwrap = getelementptr inbounds double, double* %i11_unwrap, i64 %fiv
; CHECK-NEXT:   store double %arg, double* %i16_unwrap, align 8
; CHECK-NEXT:   %i18_unwrap = icmp eq i64 %[[i1]], %i7
; CHECK-NEXT:   br i1 %i18_unwrap, label %remat_bb8_bb12_phimerge, label %remat_bb8_bb14

; CHECK: remat_bb8_bb12_phimerge:                          ; preds = %remat_bb8_bb14
; CHECK:   tail call void @free(i8* nonnull %remat_i10)

