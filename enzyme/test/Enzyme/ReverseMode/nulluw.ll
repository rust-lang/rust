; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -adce -correlated-propagation -simplifycfg -adce -S | FileCheck %s

source_filename = "<source>"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define dso_local double @integrate_image(double %arg) {
bb:
  br label %bb2


bb2:                                              ; preds = %bb7, %bb
  %maini = phi double [ 1.000000e+00, %bb ], [ %i9, %bb7 ]
  %i3 = phi i32 [ 0, %bb ], [ %i10, %bb7 ]
  %i4 = tail call double @llvm.ceil.f64(double %maini)
  %i5 = fptosi double %i4 to i32
  %i6 = icmp sgt i32 %i5, 0
  br i1 %i6, label %bb12, label %bb7

bb7:                                              ; preds = %bb12, %bb2
  %i8 = phi double [ 0.000000e+00, %bb2 ], [ %i18, %bb12 ]
  %i9 = fmul double %maini, 8.000000e-01
  %i10 = add nuw nsw i32 %i3, 1
  %i11 = icmp eq i32 %i10, 200
  br i1 %i11, label %bb1, label %bb2

bb12:                                             ; preds = %bb12, %bb2
  %i13 = phi i32 [ %i19, %bb12 ], [ 0, %bb2 ]
  %i14 = phi double [ %i18, %bb12 ], [ 0.000000e+00, %bb2 ]
  %i15 = tail call noalias dereferenceable_or_null(8) i8* @malloc(i64 8)
  %i16 = bitcast i8* %i15 to double*
  tail call fastcc void @evaluate_integrand(double* %i16)
  %i17 = load double, double* %i16, align 8
  %i18 = fadd double %i14, %i17
  tail call void @free(i8* %i15)
  %i19 = add nuw nsw i32 %i13, 1
  %i20 = icmp eq i32 %i19, %i5
  br i1 %i20, label %bb7, label %bb12

bb1:                                              ; preds = %bb7
  ret double %i8
}

declare double @llvm.ceil.f64(double) 

declare dso_local noalias i8* @malloc(i64)

define internal fastcc void @evaluate_integrand(double* nocapture %a0) {
bb:
  store double 0.000000e+00, double* %a0, align 8, !tbaa !2
  ret void
}

declare dso_local void @free(i8*)

define dso_local double @caller(double %a) {
bb:
  %b = tail call double (i8*, ...) @__enzyme_autodiff(i8* bitcast (double (double)* @integrate_image to i8*), double %a)
  ret double %b
}

declare dso_local double @__enzyme_autodiff(i8*, ...)

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 12.0.1 (https://github.com/llvm/llvm-project.git fed41342a82f5a3a9201819a82bf7a48313e296b)"}
!2 = !{!3, !3, i64 0}
!3 = !{!"double", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}

; CHECK: define internal { double } @diffeintegrate_image(double %arg, double %differeturn)
; CHECK-NEXT: bb:
; CHECK-NEXT:   %malloccall = tail call noalias nonnull dereferenceable(800) dereferenceable_or_null(800) i8* @malloc(i64 800)
; CHECK-NEXT:   %i5_malloccache = bitcast i8* %malloccall to i32*
; CHECK-NEXT:   %[[malloccall3:.+]] = tail call noalias nonnull dereferenceable(1600) dereferenceable_or_null(1600) i8* @malloc(i64 1600)
; CHECK-NEXT:   %i15_malloccache = bitcast i8* %[[malloccall3]] to i8***
; CHECK-NEXT:   br label %bb2

; CHECK: bb2:                                              ; preds = %bb7, %bb
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %bb7 ], [ 0, %bb ]
; CHECK-NEXT:   %maini = phi double [ 1.000000e+00, %bb ], [ %i9, %bb7 ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %0 = trunc i64 %iv to i32
; CHECK-NEXT:   %i4 = tail call double @llvm.ceil.f64(double %maini)
; CHECK-NEXT:   %i5 = fptosi double %i4 to i32
; CHECK-NEXT:   %1 = getelementptr inbounds i32, i32* %i5_malloccache, i64 %iv
; CHECK-NEXT:   store i32 %i5, i32* %1, align 4, !invariant.group !6
; CHECK-NEXT:   %i6 = icmp sgt i32 %i5, 0
; CHECK-NEXT:   br i1 %i6, label %bb12.preheader, label %bb7

; CHECK: bb12.preheader:                                   ; preds = %bb2
; CHECK-NEXT:   %2 = add {{(nsw )?}}i32 %i5, -1
; CHECK-NEXT:   %3 = zext i32 %2 to i64
; CHECK-NEXT:   %4 = add nuw {{(nsw )?}}i64 %3, 1
; CHECK-NEXT:   %5 = getelementptr inbounds i8**, i8*** %i15_malloccache, i64 %iv
; CHECK-NEXT:   %mallocsize = mul nuw nsw i64 %4, 8
; CHECK-NEXT:   %[[malloccall5:.+]] = tail call noalias nonnull i8* @malloc(i64 %mallocsize)
; CHECK-NEXT:   %[[i15_malloccache6:.+]] = bitcast i8* %[[malloccall5]] to i8**
; CHECK-NEXT:   store i8** %[[i15_malloccache6]], i8*** %5, align 8, !invariant.group !7
; CHECK-NEXT:   br label %bb12

; CHECK: bb7:                                              ; preds = %bb12, %bb2
; CHECK-NEXT:   %i9 = fmul double %maini, 8.000000e-01
; CHECK-NEXT:   %i10 = add nuw nsw i32 %0, 1
; CHECK-NEXT:   %i11 = icmp eq i32 %i10, 200
; CHECK-NEXT:   br i1 %i11, label %invertbb7, label %bb2

; CHECK: bb12:                                             ; preds = %bb12, %bb12.preheader
; CHECK-NEXT:   %iv1 = phi i64 [ %iv.next2, %bb12 ], [ 0, %bb12.preheader ]
; CHECK-NEXT:   %iv.next2 = add nuw nsw i64 %iv1, 1
; CHECK-NEXT:   %6 = trunc i64 %iv1 to i32
; CHECK-NEXT:   %i15 = tail call noalias dereferenceable_or_null(8) i8* @malloc(i64 8)
; CHECK-NEXT:   %7 = getelementptr inbounds i8**, i8*** %i15_malloccache, i64 %iv
; CHECK-NEXT:   %8 = load i8**, i8*** %7, align 8, !dereferenceable !8, !invariant.group !7
; CHECK-NEXT:   %9 = getelementptr inbounds i8*, i8** %8, i64 %iv1
; CHECK-NEXT:   store i8* %i15, i8** %9, align 8, !invariant.group !9
; CHECK-NEXT:   %i19 = add nuw nsw i32 %6, 1
; CHECK-NEXT:   %i20 = icmp eq i32 %i19, %i5
; CHECK-NEXT:   br i1 %i20, label %bb7, label %bb12

; CHECK: invertbb:                                         ; preds = %invertbb2
; CHECK-NEXT:   tail call void @free(i8* nonnull %malloccall)
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[malloccall3]])
; CHECK-NEXT:   ret { double } zeroinitializer

; CHECK: invertbb2:                                        ; preds = %invertbb7, %invertbb12.preheader
; CHECK-NEXT:   %"i18'de.0" = phi double [ 0.000000e+00, %invertbb12.preheader ], [ %"i18'de.1", %invertbb7 ]
; CHECK-NEXT:   %"i14'de.0" = phi double [ 0.000000e+00, %invertbb12.preheader ], [ %"i14'de.1", %invertbb7 ]
; CHECK-NEXT:   %"i17'de.0" = phi double [ 0.000000e+00, %invertbb12.preheader ], [ %"i17'de.1", %invertbb7 ]
; CHECK-NEXT:   %10 = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %10, label %invertbb, label %incinvertbb2

; CHECK: incinvertbb2:                                     ; preds = %invertbb2
; CHECK-NEXT:   %11 = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertbb7

; CHECK: invertbb12.preheader:                             ; preds = %remat_enter
; CHECK-NEXT:   %[[_unwrap8:.+]] = getelementptr inbounds i8**, i8*** %i15_malloccache, i64 %"iv'ac.0"
; CHECK-NEXT:   %[[forfree9:.+]] = load i8**, i8*** %[[_unwrap8]], align 8, !dereferenceable !8, !invariant.group !7
; CHECK-NEXT:   %12 = bitcast i8** %[[forfree9]] to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %12)
; CHECK-NEXT:   br label %invertbb2

; CHECK: invertbb7.loopexit:                               ; preds = %invertbb7
; CHECK-NEXT:   %13 = getelementptr inbounds i32, i32* %i5_malloccache, i64 %"iv'ac.0"
; CHECK-NEXT:   %14 = load i32, i32* %13, align 4, !invariant.group !6
; CHECK-NEXT:   %_unwrap1 = add i32 %14, -1
; CHECK-NEXT:   %_unwrap2 = zext i32 %_unwrap1 to i64
; CHECK-NEXT:   br label %remat_enter

; CHECK: invertbb7:                                        ; preds = %bb7, %incinvertbb2
; CHECK-NEXT:   %"i18'de.1" = phi double [ %"i18'de.0", %incinvertbb2 ], [ 0.000000e+00, %bb7 ]
; CHECK-NEXT:   %"i14'de.1" = phi double [ %"i14'de.0", %incinvertbb2 ], [ 0.000000e+00, %bb7 ]
; CHECK-NEXT:   %"i17'de.1" = phi double [ %"i17'de.0", %incinvertbb2 ], [ 0.000000e+00, %bb7 ]
; CHECK-NEXT:   %"i8'de.0" = phi double [ 0.000000e+00, %incinvertbb2 ], [ %differeturn, %bb7 ]
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %11, %incinvertbb2 ], [ 199, %bb7 ]
; CHECK-NEXT:   %15 = getelementptr inbounds i32, i32* %i5_malloccache, i64 %"iv'ac.0"
; CHECK-NEXT:   %16 = load i32, i32* %15, align 4, !invariant.group !6
; CHECK-NEXT:   %i6_unwrap = icmp sgt i32 %16, 0
; CHECK-NEXT:   %17 = fadd fast double %"i18'de.1", %"i8'de.0"
; CHECK-NEXT:   br i1 %i6_unwrap, label %invertbb7.loopexit, label %invertbb2

; CHECK: incinvertbb12:                                    ; preds = %remat_enter
; CHECK-NEXT:   %18 = add nsw i64 %"iv1'ac.0", -1
; CHECK-NEXT:   br label %remat_enter

; CHECK: remat_enter:                                      ; preds = %incinvertbb12, %invertbb7.loopexit
; CHECK-NEXT:   %"i18'de.2" = phi double [ %17, %invertbb7.loopexit ], [ %19, %incinvertbb12 ]
; CHECK-NEXT:   %"i14'de.2" = phi double [ %"i14'de.1", %invertbb7.loopexit ], [ 0.000000e+00, %incinvertbb12 ]
; CHECK-NEXT:   %"i17'de.2" = phi double [ %"i17'de.1", %invertbb7.loopexit ], [ 0.000000e+00, %incinvertbb12 ]
; CHECK-NEXT:   %"iv1'ac.0" = phi i64 [ %_unwrap2, %invertbb7.loopexit ], [ %18, %incinvertbb12 ]
; CHECK-NEXT:   %"i15'mi" = tail call noalias nonnull dereferenceable_or_null(8) i8* @malloc(i64 8)
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull dereferenceable(8) dereferenceable_or_null(8) %"i15'mi", i8 0, i64 8, i1 false)
; CHECK-NEXT:   %19 = fadd fast double %"i14'de.2", %"i18'de.2"
; CHECK-NEXT:   %20 = fadd fast double %"i17'de.2", %"i18'de.2"
; CHECK-NEXT:   %"i16'ipc_unwrap" = bitcast i8* %"i15'mi" to double*
; CHECK-NEXT:   %21 = load double, double* %"i16'ipc_unwrap", align 8
; CHECK-NEXT:   %22 = fadd fast double %21, %20
; CHECK-NEXT:   store double %22, double* %"i16'ipc_unwrap", align 8
; CHECK-NEXT:   %23 = getelementptr inbounds i8**, i8*** %i15_malloccache, i64 %"iv'ac.0"
; CHECK-NEXT:   %24 = load i8**, i8*** %23, align 8, !dereferenceable !8, !invariant.group !7
; CHECK-NEXT:   %25 = getelementptr inbounds i8*, i8** %24, i64 %"iv1'ac.0"
; CHECK-NEXT:   %26 = load i8*, i8** %25, align 8, !invariant.group !9
; CHECK-NEXT:   %i16_unwrap = bitcast i8* %26 to double*
; CHECK-NEXT:   call fastcc void @diffeevaluate_integrand(double* %i16_unwrap, double* nonnull %"i16'ipc_unwrap")
; CHECK-NEXT:   tail call void @free(i8* nonnull %"i15'mi")
; CHECK-NEXT:   tail call void @free(i8* %26)
; CHECK-NEXT:   %27 = icmp eq i64 %"iv1'ac.0", 0
; CHECK-NEXT:   br i1 %27, label %invertbb12.preheader, label %incinvertbb12
; CHECK-NEXT: }

