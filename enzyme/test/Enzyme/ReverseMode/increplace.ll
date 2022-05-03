; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -early-cse -instsimplify -simplifycfg -loop-deletion -simplifycfg -S | FileCheck %s

source_filename = "text"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:10:11:12:13"
target triple = "x86_64-linux-gnu"

declare void @julia_throw_boundserror_1767(i64 %d)

define double @sub(double* %i58) {
bb:
  br label %bb22

bb22:                                             ; preds = %bb32, %bb
  %i23 = phi i64 [ %i78, %bb32 ], [ 0, %bb ]
  %i78 = add nuw nsw i64 %i23, 1
  %i26 = icmp sle i64 %i23, 10
  br i1 %i26, label %bb32, label %bb29

bb29:                                             ; preds = %bb22
  %i30 = add nuw nsw i64 %i23, 1
  call void @julia_throw_boundserror_1767(i64 %i30)
  unreachable

bb32:                                             ; preds = %bb22
  %c = icmp sle i64 %i23, 5
  br i1 %c, label %bb22, label %exit

exit:
  %v = load double, double* %i58, align 8
  ret double %v
}


declare dso_local double @__enzyme_autodiff(i8*, i64*, double*, double*)

define void @main(double* %arg, double* %arg1) {
bb:
  %enzyme_dup = alloca i64, align 8
  %i = tail call double @__enzyme_autodiff(i8* bitcast (double (double*)* @julia_arsum2_1761 to i8*), i64* %enzyme_dup, double* %arg, double* %arg1)
  ret void
}

define double @julia_arsum2_1761(double* %arg) {
bb:
  %i23 = call double @sub(double* %arg)
  store double 0.000000e+00, double* %arg
  ret double %i23
}

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = !{!3, !3, i64 0}
!3 = !{!"jtbaa_data", !4, i64 0}
!4 = !{!"jtbaa", !5, i64 0}
!5 = !{!"jtbaa"}

; Creating this should not segfault due to the use of the replaced iv in the unreachable block
; CHECK: define internal void @diffesub(double* %i58, double* %"i58'", double %differeturn)
; CHECK-NEXT: bb:
; CHECK-NEXT:   %0 = load double, double* %"i58'", align 8
; CHECK-NEXT:   %1 = fadd fast double %0, %differeturn
; CHECK-NEXT:   store double %1, double* %"i58'", align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
