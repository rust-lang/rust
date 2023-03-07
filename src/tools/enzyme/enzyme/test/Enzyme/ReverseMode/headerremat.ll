; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S -gvn -dse -dse | FileCheck %s

source_filename = "<source>"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@.str = private unnamed_addr constant [11 x i8] c"dtheta=%d\0A\00", align 1
@.str.1 = private unnamed_addr constant [23 x i8] c"dout[%d]=%f answer=%d\0A\00", align 1

define i32 @_Z18evaluate_integrandii(i32 %arg, i32 %arg1) {
bb:
  %i = mul nsw i32 %arg1, %arg
  ret i32 %i
}

define dso_local double @_Z15integrate_imagedPd(double %arg, double* nocapture %arg1) {
bb:
  br label %bb5

bb5:                                              ; preds = %bb5, %bb
  %i6 = phi i64 [ 0, %bb ], [ %i19, %bb5 ]
  %i7 = phi double [ %arg, %bb ], [ %i17, %bb5 ]
  %i8 = phi double [ 1.000000e+00, %bb ], [ %i18, %bb5 ]
  %i9 = fptosi double %i7 to i32
  %i10 = fptosi double %i8 to i32
  %i11 = mul nsw i32 %i9, %i10
  %i12 = sitofp i32 %i11 to double
  %i13 = getelementptr inbounds double, double* %arg1, i64 %i6
  %i14 = load double, double* %i13, align 8
  %i15 = fmul double %i14, %i12
  store double %i15, double* %i13, align 8
  %i17 = fdiv double %i7, 8.000000e-01
  %i18 = fmul double %i17, 2.500000e-01
  %i19 = add nuw nsw i64 %i6, 1
  %i20 = icmp eq i64 %i19, 10
  br i1 %i20, label %bb2, label %bb5

bb2:                                              ; preds = %bb5
  %i = tail call i32 @_Z18evaluate_integrandii(i32 %i9, i32 %i10)
  %i3 = sitofp i32 %i to double
  %i4 = fmul double %i7, %i3
  ret double %i4
}

; Function Attrs: nofree nounwind
declare dso_local i32 @printf(i8* nocapture readonly, ...)

; Function Attrs: norecurse uwtable mustprogress
define dso_local i32 @main() {
bb:
  %i = alloca [10 x double], align 16
  %i1 = alloca [10 x double], align 16
  %i2 = bitcast [10 x double]* %i to i8*
  %i3 = bitcast [10 x double]* %i1 to i8*
  %i4 = getelementptr inbounds [10 x double], [10 x double]* %i1, i64 0, i64 0
  store double 1.000000e+00, double* %i4, align 16
  %i5 = getelementptr inbounds [10 x double], [10 x double]* %i1, i64 0, i64 1
  store double 1.000000e+00, double* %i5, align 8
  %i6 = getelementptr inbounds [10 x double], [10 x double]* %i1, i64 0, i64 2
  store double 1.000000e+00, double* %i6, align 16
  %i7 = getelementptr inbounds [10 x double], [10 x double]* %i1, i64 0, i64 3
  store double 1.000000e+00, double* %i7, align 8
  %i8 = getelementptr inbounds [10 x double], [10 x double]* %i1, i64 0, i64 4
  store double 1.000000e+00, double* %i8, align 16
  %i9 = getelementptr inbounds [10 x double], [10 x double]* %i1, i64 0, i64 5
  store double 1.000000e+00, double* %i9, align 8
  %i10 = getelementptr inbounds [10 x double], [10 x double]* %i1, i64 0, i64 6
  store double 1.000000e+00, double* %i10, align 16
  %i11 = getelementptr inbounds [10 x double], [10 x double]* %i1, i64 0, i64 7
  store double 1.000000e+00, double* %i11, align 8
  %i12 = getelementptr inbounds [10 x double], [10 x double]* %i1, i64 0, i64 8
  store double 1.000000e+00, double* %i12, align 16
  %i13 = getelementptr inbounds [10 x double], [10 x double]* %i1, i64 0, i64 9
  store double 1.000000e+00, double* %i13, align 8
  %i14 = getelementptr inbounds [10 x double], [10 x double]* %i, i64 0, i64 0
  call void (double (double, double*)*, ...) @_Z17__enzyme_autodiffPFddPdEz(double (double, double*)* nonnull @_Z15integrate_imagedPd, double 2.000000e+02, double* nonnull %i14, double* nonnull %i4)
  %i15 = load double, double* %i4, align 16
  %i16 = call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([23 x i8], [23 x i8]* @.str.1, i64 0, i64 0), i32 0, double %i15, i32 200)
  %i17 = load double, double* %i5, align 8
  %i18 = call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([23 x i8], [23 x i8]* @.str.1, i64 0, i64 0), i32 1, double %i17, i32 15500)
  %i19 = load double, double* %i6, align 16
  %i20 = call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([23 x i8], [23 x i8]* @.str.1, i64 0, i64 0), i32 2, double %i19, i32 24336)
  %i21 = load double, double* %i7, align 8
  %i22 = call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([23 x i8], [23 x i8]* @.str.1, i64 0, i64 0), i32 3, double %i21, i32 37830)
  %i23 = load double, double* %i8, align 16
  %i24 = call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([23 x i8], [23 x i8]* @.str.1, i64 0, i64 0), i32 4, double %i23, i32 59536)
  %i25 = load double, double* %i9, align 8
  %i26 = call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([23 x i8], [23 x i8]* @.str.1, i64 0, i64 0), i32 5, double %i25, i32 92720)
  %i27 = load double, double* %i10, align 16
  %i28 = call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([23 x i8], [23 x i8]* @.str.1, i64 0, i64 0), i32 6, double %i27, i32 144780)
  %i29 = load double, double* %i11, align 8
  %i30 = call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([23 x i8], [23 x i8]* @.str.1, i64 0, i64 0), i32 7, double %i29, i32 226814)
  %i31 = load double, double* %i12, align 16
  %i32 = call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([23 x i8], [23 x i8]* @.str.1, i64 0, i64 0), i32 8, double %i31, i32 355216)
  %i33 = load double, double* %i13, align 8
  %i34 = call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([23 x i8], [23 x i8]* @.str.1, i64 0, i64 0), i32 9, double %i33, i32 554280)
  ret i32 0
}

declare void @_Z17__enzyme_autodiffPFddPdEz(double (double, double*)*, ...)

; CHECK: define internal { double } @diffe_Z15integrate_imagedPd(double %arg, double* nocapture %arg1, double* nocapture %"arg1'", double %differeturn)
; CHECK-NEXT: bb:
; CHECK-NEXT:   %malloccall = tail call noalias nonnull dereferenceable(80) dereferenceable_or_null(80) i8* @malloc(i64 80)
; CHECK-NEXT:   %i7_malloccache = bitcast i8* %malloccall to double*
; CHECK-NEXT:   br label %bb5

; CHECK: bb5:                                              ; preds = %bb5, %bb
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %bb5 ], [ 0, %bb ]
; CHECK-NEXT:   %i7 = phi double [ %arg, %bb ], [ %i17, %bb5 ]
; CHECK-NEXT:   %i8 = phi double [ 1.000000e+00, %bb ], [ %i18, %bb5 ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %i9 = fptosi double %i7 to i32
; CHECK-NEXT:   %i10 = fptosi double %i8 to i32
; CHECK-NEXT:   %i11 = mul nsw i32 %i9, %i10
; CHECK-NEXT:   %i12 = sitofp i32 %i11 to double
; CHECK-NEXT:   %i13 = getelementptr inbounds double, double* %arg1, i64 %iv
; CHECK-NEXT:   %i14 = load double, double* %i13, align 8
; CHECK-NEXT:   %i15 = fmul double %i14, %i12
; CHECK-NEXT:   store double %i15, double* %i13, align 8
; CHECK-NEXT:   %0 = getelementptr inbounds double, double* %i7_malloccache, i64 %iv
; CHECK-NEXT:   store double %i7, double* %0, align 8, !invariant.group ![[g0:[0-9]+]]
; CHECK-NEXT:   %i17 = fdiv double %i7, 8.000000e-01
; CHECK-NEXT:   %i18 = fmul double %i17, 2.500000e-01
; CHECK-NEXT:   %i20 = icmp eq i64 %iv.next, 10
; CHECK-NEXT:   br i1 %i20, label %bb2, label %bb5

; CHECK: bb2:                                              ; preds = %bb5
; CHECK-NEXT:   %i = tail call i32 @_Z18evaluate_integrandii(i32 %i9, i32 %i10)
; CHECK-NEXT:   %i3 = sitofp i32 %i to double
; CHECK-NEXT:   %m0diffei7 = fmul fast double %differeturn, %i3
; CHECK-NEXT:   br label %invertbb5

; CHECK: invertbb:                                         ; preds = %invertbb5_phimerge
; CHECK-NEXT:   %1 = insertvalue { double } undef, double %14, 0
; CHECK-NEXT:   tail call void @free(i8* nonnull %malloccall)
; CHECK-NEXT:   ret { double } %1

; CHECK: invertbb5:                                        ; preds = %bb2, %incinvertbb5
; CHECK-NEXT:   %"i7'de.0" = phi double [ %m0diffei7, %bb2 ], [ 0.000000e+00, %incinvertbb5 ]
; CHECK-NEXT:   %"i17'de.0" = phi double [ 0.000000e+00, %bb2 ], [ %12, %incinvertbb5 ]
; CHECK-NEXT:   %"arg'de.0" = phi double [ 0.000000e+00, %bb2 ], [ %14, %incinvertbb5 ]
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ 9, %bb2 ], [ %15, %incinvertbb5 ]
; CHECK-NEXT:   %d0diffei7 = fdiv fast double %"i17'de.0", 8.000000e-01
; CHECK-NEXT:   %2 = fadd fast double %"i7'de.0", %d0diffei7
; CHECK-NEXT:   %"i13'ipg_unwrap" = getelementptr inbounds double, double* %"arg1'", i64 %"iv'ac.0"
; CHECK-NEXT:   %3 = load double, double* %"i13'ipg_unwrap", align 8
; DCE-NEXT:   store double 0.000000e+00, double* %"i13'ipg_unwrap", align 8
; CHECK:   %4 = getelementptr inbounds double, double* %i7_malloccache, i64 %"iv'ac.0"
; CHECK-NEXT:   %5 = load double, double* %4, align 8, !invariant.group ![[g0]]
; CHECK-NEXT:   %i9_unwrap = fptosi double %5 to i32
; CHECK-NEXT:   %6 = icmp ne i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %6, label %invertbb5_phirc, label %invertbb5_phimerge

; CHECK: invertbb5_phirc:                                  ; preds = %invertbb5
; CHECK-NEXT:   %7 = sub nuw i64 %"iv'ac.0", 1
; CHECK-NEXT:   %8 = getelementptr inbounds double, double* %i7_malloccache, i64 %7
; CHECK-NEXT:   %9 = load double, double* %8, align 8, !invariant.group ![[g0]]
; CHECK-NEXT:   %i17_unwrap = fdiv double %9, 8.000000e-01
; CHECK-NEXT:   %i18_unwrap = fmul double %i17_unwrap, 2.500000e-01
; CHECK-NEXT:   br label %invertbb5_phimerge

; CHECK: invertbb5_phimerge:                               ; preds = %invertbb5, %invertbb5_phirc
; CHECK-NEXT:   %10 = phi {{(fast )?}}double [ %i18_unwrap, %invertbb5_phirc ], [ 1.000000e+00, %invertbb5 ]
; CHECK-NEXT:   %i10_unwrap = fptosi double %10 to i32
; CHECK-NEXT:   %i11_unwrap = mul nsw i32 %i9_unwrap, %i10_unwrap
; CHECK-NEXT:   %i12_unwrap = sitofp i32 %i11_unwrap to double
; CHECK-NEXT:   %m0diffei14 = fmul fast double %3, %i12_unwrap
; CHECK-NEXT:   store double %m0diffei14, double* %"i13'ipg_unwrap", align 8
; CHECK-NEXT:   %11 = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   %12 = select {{(fast )?}}i1 %11, double 0.000000e+00, double %2
; CHECK-NEXT:   %13 = fadd fast double %"arg'de.0", %2
; CHECK-NEXT:   %14 = select {{(fast )?}}i1 %11, double %13, double %"arg'de.0"
; CHECK-NEXT:   br i1 %11, label %invertbb, label %incinvertbb5

; CHECK: incinvertbb5:                                     ; preds = %invertbb5_phimerge
; CHECK-NEXT:   %15 = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertbb5
; CHECK-NEXT: }
