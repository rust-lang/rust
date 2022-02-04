; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -correlated-propagation -adce -S | FileCheck %s
source_filename = "text"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

; Function Attrs: cold noreturn nounwind
declare void @llvm.trap() #0

declare dso_local i32 @MPI_Comm_rank(i64, i64)

define double @sum(double* %arg, i64 %comm) {
bb:
  %alloc = alloca i32, align 8
  %i5 = ptrtoint i32* %alloc to i64
  br label %bb11

bb11:                                             ; preds = %bb
  %idx = phi i64 [ 0, %bb ], [ %inc, %bb22 ]
  %sum = phi double [ 0.000000e+00, %bb ], [ %add, %bb22 ]
  %inc = add i64 %idx, 1
  %i13 = getelementptr inbounds double, double* %arg, i64 %idx
  %i14 = load double, double* %i13, align 8
  %i16 = fmul double %i14, %i14
  %i19 = call i32 @MPI_Comm_rank(i64 %comm, i64 %i5)
  %ld = load i32, i32* %alloc
  %cf = uitofp i32 %ld to double
  %mm = fmul double %i16, %cf
  %add = fadd double %sum, %mm
  %i20 = icmp eq i32 %i19, 0
  br i1 %i20, label %bb22, label %bb21

bb21:                                             ; preds = %bb11, %bb
  call void @llvm.trap() #1
  unreachable

bb22:
  %cmp = icmp eq i64 %idx, 9
  br i1 %cmp, label %exit, label %bb11

exit:
  ret double %add
}

define void @dsum(double* %x, double* %xp, i64 %n) {
entry:
  %0 = tail call double (double (double*, i64)*, ...) @__enzyme_autodiff(double (double*, i64)* nonnull @sum, double* %x, double* %xp, i64 %n)
  ret void
}

declare double @__enzyme_autodiff(double (double*, i64)*, ...)

attributes #0 = { cold noreturn nounwind }
attributes #1 = { noreturn }

; CHECK: define internal void @diffesum(double* %arg, double* %"arg'", i64 %comm, double %differeturn) 
; CHECK-NEXT: bb:
; CHECK-NEXT:   %0 = alloca i32
; CHECK-NEXT:   %1 = alloca i32
; CHECK-NEXT:   br label %bb11

; CHECK: bb11:                                             ; preds = %bb11, %bb
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %bb11 ], [ 0, %bb ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %2 = bitcast i32* %1 to i8*
; CHECK-NEXT:   call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %2)
; CHECK-NEXT:   %3 = ptrtoint i32* %1 to i64
; CHECK-NEXT:   %4 = call i32 @MPI_Comm_rank(i64 %comm, i64 %3) 
; CHECK-NEXT:   %5 = bitcast i32* %1 to i8*
; CHECK-NEXT:   call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %5)
; CHECK-NEXT:   %cmp = icmp eq i64 %iv, 9
; CHECK-NEXT:   br i1 %cmp, label %invertbb22, label %bb11

; CHECK: invertbb:                                         ; preds = %invertbb22
; CHECK-NEXT:   ret void

; CHECK: incinvertbb11:                                    ; preds = %invertbb22
; CHECK-NEXT:   %6 = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertbb22

; CHECK: invertbb22:                                       ; preds = %bb11, %incinvertbb11
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %6, %incinvertbb11 ], [ 9, %bb11 ]
; CHECK-NEXT:   %7 = bitcast i32* %0 to i8*
; CHECK-NEXT:   call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %7)
; CHECK-NEXT:   %8 = ptrtoint i32* %0 to i64
; CHECK-NEXT:   %9 = call i32 @MPI_Comm_rank(i64 %comm, i64 %8) 
; CHECK-NEXT:   %10 = load i32, i32* %0
; CHECK-NEXT:   %11 = bitcast i32* %0 to i8*
; CHECK-NEXT:   call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %11)
; CHECK-NEXT:   %cf_unwrap = uitofp i32 %10 to double
; CHECK-NEXT:   %m0diffei16 = fmul fast double %differeturn, %cf_unwrap
; CHECK-NEXT:   %i13_unwrap = getelementptr inbounds double, double* %arg, i64 %"iv'ac.0"
; CHECK-NEXT:   %i14_unwrap = load double, double* %i13_unwrap, align 8, !invariant.group !0
; CHECK-NEXT:   %m0diffei14 = fmul fast double %m0diffei16, %i14_unwrap
; CHECK-NEXT:   %m1diffei14 = fmul fast double %m0diffei16, %i14_unwrap
; CHECK-NEXT:   %12 = fadd fast double %m0diffei14, %m1diffei14
; CHECK-NEXT:   %"i13'ipg_unwrap" = getelementptr inbounds double, double* %"arg'", i64 %"iv'ac.0"
; CHECK-NEXT:   %13 = load double, double* %"i13'ipg_unwrap", align 8
; CHECK-NEXT:   %14 = fadd fast double %13, %12
; CHECK-NEXT:   store double %14, double* %"i13'ipg_unwrap", align 8
; CHECK-NEXT:   %15 = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %15, label %invertbb, label %incinvertbb11
; CHECK-NEXT: }
