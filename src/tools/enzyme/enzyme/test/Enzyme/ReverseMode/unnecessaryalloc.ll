; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -simplifycfg -S | FileCheck %s

source_filename = "/mnt/pci4/wmdata/Enzyme/enzyme/test/Integration/ReverseMode/mycos.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: readnone
declare dso_local i64 @__enzyme_iter(i64, i64) #2

; Function Attrs: noinline nounwind optnone uwtable
define dso_local double @d_mysin2(double %x) {
entry:
  %r1 = call double @diffemy_sin2(double %x, double 1.000000e+00)
  %x2 = fmul double %r1, %r1
  ret double %x2
}

declare dso_local double @__enzyme_autodiff(i8*, double)

; Function Attrs: noinline nounwind optnone uwtable
define dso_local double @dd_mysin2(double %x) {
entry:
  %call = call double @__enzyme_autodiff(i8* bitcast (double (double)* @d_mysin2 to i8*), double %x)
  ret double %call
}

; Function Attrs: noinline nounwind uwtable willreturn mustprogress
define internal double @diffemy_sin2(double %x, double %differeturn) {
entry:
  %call = call i64 @__enzyme_iter(i64 13, i64 1)
  %i = add i64 %call, 1
  %i1 = add nuw i64 %i, 1
  %mallocsize = mul nuw nsw i64 %i1, 8
  %malloccall = tail call noalias nonnull i8* @malloc(i64 %mallocsize)
  tail call void @free(i8* nonnull %malloccall)
  ret double %x
}

declare noalias i8* @malloc(i64)

declare void @free(i8*)

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 12.0.1 (git@github.com:llvm/llvm-project 4973ce53ca8abfc14233a3d8b3045673e0e8543c)"}
!2 = distinct !{!2, !3}
!3 = !{!"llvm.loop.mustprogress"}
!4 = distinct !{!4, !3}
!5 = distinct !{}


; CHECK: define internal double @augmented_diffemy_sin2(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %i = add i64 14, 1
; CHECK-NEXT:   %i1 = add nuw i64 %i, 1
; CHECK-NEXT:   %mallocsize = mul nuw nsw i64 %i1, 8
; CHECK-NEXT:   %malloccall = tail call noalias nonnull i8* @malloc(i64 %mallocsize)
; CHECK-NEXT:   tail call void @free(i8* nonnull %malloccall)
; CHECK-NEXT:   %0 = insertvalue { i8*, double } undef, double %x, 1
; CHECK-NEXT:   ret double %x
; CHECK-NEXT: }

; CHECK: define internal { double } @diffediffemy_sin2(double %x, double %differeturn1, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = insertvalue { double } undef, double %differeturn, 0
; CHECK-NEXT:   ret { double } %0
; CHECK-NEXT: }
