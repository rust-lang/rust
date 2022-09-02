; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -simplifycfg -instsimplify -gvn -adce -S | FileCheck %s

source_filename = "/app/example.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind readnone uwtable
define dso_local double @square(double %arg) #0 {
entry:
  %i = alloca [30 x double], align 16
  %i1 = bitcast [30 x double]* %i to i8*
  br label %loop


loop:                                              ; preds = %bb6, %bb
  %i4 = phi i64 [ 0, %entry ], [ %i11, %loopExit ]
  %i5 = phi double [ 0.000000e+00, %entry ], [ %i10, %loopExit ]
  call void @llvm.lifetime.start.p0i8(i64 240, i8* nonnull %i1) #4
  br label %setLoop

setLoop:                                             ; preds = %bb13, %bb3
  %i14 = phi i64 [ 0, %loop ], [ %i21, %setExit ]
  %i15 = and i64 %i14, 1
  %i16 = icmp eq i64 %i15, 0
  br i1 %i16, label %if.true, label %if.false

if.true:
  %i17 = trunc i64 %i14 to i32
  %i18 = call fast double @llvm.powi.f64(double %arg, i32 %i17)
  br label %setExit

if.false:
  br label %setExit

setExit:
  %i19 = phi double [ %i18, %if.true ], [ 0.000000e+00, %if.false ]
  %i20 = getelementptr inbounds [30 x double], [30 x double]* %i, i64 0, i64 %i14
  store double %i19, double* %i20, align 8, !tbaa !2
  %i21 = add nuw nsw i64 %i14, 1
  %i22 = icmp eq i64 %i21, 30
  br i1 %i22, label %loopExit, label %setLoop

loopExit:                                              ; preds = %bb13
  %i7 = getelementptr inbounds [30 x double], [30 x double]* %i, i64 0, i64 %i4
  %i8 = load double, double* %i7, align 8, !tbaa !2
  %i9 = fmul fast double %i8, %i8
  %i10 = fadd fast double %i9, %i5
  call void @llvm.lifetime.end.p0i8(i64 240, i8* nonnull %i1) #4
  %i11 = add nuw nsw i64 %i4, 1
  %i12 = icmp eq i64 %i11, 20
  br i1 %i12, label %exit, label %loop

exit:                                              ; preds = %bb6
  ret double %i10
}

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

; Function Attrs: nounwind uwtable
define dso_local double @dsquare(double %arg) local_unnamed_addr #2 {
bb:
  %i = call fast double @__enzyme_autodiff(i8* bitcast (double (double)* @square to i8*), double %arg) #4
  ret double %i
}

declare dso_local double @__enzyme_autodiff(i8*, double)

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare double @llvm.powi.f64(double, i32) #3

attributes #0 = { nounwind readnone uwtable }
attributes #1 = { argmemonly nounwind }
attributes #2 = { nounwind uwtable }
attributes #3 = { nounwind readnone speculatable }
attributes #4 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 12.0.0 (https://github.com/llvm/llvm-project.git d28af7c654d8db0b68c175db5ce212d74fb5e9bc)"}
!2 = !{!3, !3, i64 0}
!3 = !{!"double", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}

; CHECK: define internal { double } @diffesquare(double %arg, double %differeturn) 
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"i'ipa" = alloca [30 x double], align 16
; CHECK:        %i = alloca [30 x double], align 16
; CHECK-NEXT:   br label %loop

; CHECK: loopExit:                                         ; preds = %setLoop
; CHECK-NEXT:   %i12 = icmp eq i64 %iv.next, 20
; CHECK-NEXT:   br i1 %i12, label %remat_enter, label %loop

; CHECK: invertentry:                                      ; preds = %invertloop
; CHECK-NEXT:   %0 = insertvalue { double } undef, double %"arg'de.0", 0
; CHECK-NEXT:   ret { double } %0

; CHECK: invertloop:                                       ; preds = %invertsetLoop
; CHECK-NEXT:   %1 = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   %2 = select {{(fast )?}}i1 %1, double 0.000000e+00, double %"i10'de.0"
; CHECK-NEXT:   br i1 %1, label %invertentry, label %incinvertloop

; CHECK: incinvertloop:                                    ; preds = %invertloop
; CHECK-NEXT:   %3 = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %remat_enter

; CHECK: invertsetLoop:                                    ; preds = %invertsetExit, %invertif.true
; CHECK-NEXT:   %"i18'de.0" = phi double [ 0.000000e+00, %invertif.true ], [ %16, %invertsetExit ]
; CHECK-NEXT:   %"arg'de.0" = phi double [ %13, %invertif.true ], [ %"arg'de.1", %invertsetExit ]
; CHECK-NEXT:   %4 = icmp eq i64 %"iv1'ac.0", 0
; CHECK-NEXT:   br i1 %4, label %invertloop, label %incinvertsetLoop

; CHECK: incinvertsetLoop:                                 ; preds = %invertsetLoop
; CHECK-NEXT:   %5 = add nsw i64 %"iv1'ac.0", -1
; CHECK-NEXT:   br label %invertsetExit

; CHECK: invertif.true:                                    ; preds = %invertsetExit
; CHECK-NEXT:   %i17_unwrap4 = trunc i64 %"iv1'ac.0" to i32
; CHECK-NEXT:   %6 = sub i32 %i17_unwrap4, 1
; CHECK-NEXT:   %7 = call fast double @llvm.powi.f64{{(\.i32)?}}(double %arg, i32 %6)
; CHECK-DAG:    %[[a8:.+]] = sitofp i32 %i17_unwrap4 to double
; CHECK-DAG:    %[[a9:.+]] = fmul fast double %16, %7
; CHECK-NEXT:   %10 = fmul fast double %[[a9]], %[[a8]]
; CHECK-NEXT:   %11 = icmp eq i32 0, %i17_unwrap4
; CHECK-NEXT:   %12 = fadd fast double %"arg'de.1", %10
; CHECK-NEXT:   %13 = select {{(fast )?}}i1 %11, double %"arg'de.1", double %12
; CHECK-NEXT:   br label %invertsetLoop

; CHECK: invertsetExit:                                    ; preds = %remat_loop_loopExit, %incinvertsetLoop
; CHECK-NEXT:   %"i18'de.1" = phi double [ %"i18'de.2", %remat_loop_loopExit ], [ %"i18'de.0", %incinvertsetLoop ]
; CHECK-NEXT:   %"arg'de.1" = phi double [ %"arg'de.2", %remat_loop_loopExit ], [ %"arg'de.0", %incinvertsetLoop ]
; CHECK-NEXT:   %"iv1'ac.0" = phi i64 [ 29, %remat_loop_loopExit ], [ %5, %incinvertsetLoop ]
; CHECK-NEXT:   %"i20'ipg_unwrap" = getelementptr inbounds [30 x double], [30 x double]* %"i'ipa", i64 0, i64 %"iv1'ac.0"
; CHECK-NEXT:   %14 = load double, double* %"i20'ipg_unwrap", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"i20'ipg_unwrap", align 8
; CHECK-NEXT:   %i15_unwrap5 = and i64 %"iv1'ac.0", 1
; CHECK-NEXT:   %i16_unwrap6 = icmp eq i64 %i15_unwrap5, 0
; CHECK-NEXT:   %15 = fadd fast double %"i18'de.1", %14
; CHECK-NEXT:   %16 = select {{(fast )?}}i1 %i16_unwrap6, double %15, double %"i18'de.1"
; CHECK-NEXT:   br i1 %i16_unwrap6, label %invertif.true, label %invertsetLoop

; CHECK: remat_enter:                                      ; preds = %loopExit, %incinvertloop
; CHECK-NEXT:   %"i18'de.2" = phi double [ %"i18'de.0", %incinvertloop ], [ 0.000000e+00, %loopExit ]
; CHECK-NEXT:   %"arg'de.2" = phi double [ %"arg'de.0", %incinvertloop ], [ 0.000000e+00, %loopExit ]
; CHECK-NEXT:   %"i10'de.0" = phi double [ %2, %incinvertloop ], [ %differeturn, %loopExit ]
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %3, %incinvertloop ], [ 19, %loopExit ]
; CHECK-NEXT:   br label %remat_loop_setLoop

; CHECK: remat_loop_setLoop:                               ; preds = %remat_loop_setLoop, %remat_enter
; CHECK-NEXT:   %fiv = phi i64 [ %17, %remat_loop_setLoop ], [ 0, %remat_enter ]
; CHECK-NEXT:   %17 = add i64 %fiv, 1
; CHECK-DAG:   %i20_unwrap = getelementptr inbounds [30 x double], [30 x double]* %i, i64 0, i64 %fiv
; CHECK-DAG:   %i15_unwrap1 = and i64 %fiv, 1
; CHECK-DAG:   %i16_unwrap2 = icmp eq i64 %i15_unwrap1, 0
; CHECK-DAG   %i17_unwrap = trunc i64 %fiv to i32
; CHECK-DAG:   %18 = call fast double @llvm.powi.f64{{(\.i32)?}}(double %arg, i32 %i17_unwrap) 
; CHECK-DAG:   %19 = select i1 %i16_unwrap2, double %18, double 0.000000e+00
; CHECK-DAG:   store double %19, double* %i20_unwrap, align 8
; CHECK-NEXT:   %i22_unwrap = icmp eq i64 %17, 30
; CHECK-NEXT:   br i1 %i22_unwrap, label %remat_loop_loopExit, label %remat_loop_setLoop

; CHECK: remat_loop_loopExit:                              ; preds = %remat_loop_setLoop
; CHECK-NEXT:   %i7_unwrap = getelementptr inbounds [30 x double], [30 x double]* %i, i64 0, i64 %"iv'ac.0"
; CHECK-NEXT:   %i8_unwrap = load double, double* %i7_unwrap, align 8, !tbaa !2, !invariant.group !
; CHECK-NEXT:   %m0diffei8 = fmul fast double %"i10'de.0", %i8_unwrap
; CHECK-NEXT:   %20 = fadd fast double %m0diffei8, %m0diffei8
; CHECK-NEXT:   %"i7'ipg_unwrap" = getelementptr inbounds [30 x double], [30 x double]* %"i'ipa", i64 0, i64 %"iv'ac.0"
; CHECK-NEXT:   %21 = load double, double* %"i7'ipg_unwrap", align 8
; CHECK-NEXT:   %22 = fadd fast double %21, %20
; CHECK-NEXT:   store double %22, double* %"i7'ipg_unwrap", align 8
; CHECK-NEXT:   br label %invertsetExit
