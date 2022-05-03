; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -early-cse -simplifycfg -instsimplify -correlated-propagation -adce -S | FileCheck %s

source_filename = "/mnt/pci4/wmdata/Enzyme/enzyme/test/Integration/ReverseMode/rwrmeta.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline norecurse nounwind readonly uwtable willreturn
define dso_local double @loadSq(double** noalias nocapture readonly %x) local_unnamed_addr #0 {
entry:
  %l0 = load double*, double** %x, align 8, !tbaa !2
  %l1 = load double, double* %l0, align 8, !tbaa !6
  %mul = fmul double %l1, %l1
  ret double %mul
}

; Function Attrs: nofree norecurse nounwind uwtable willreturn
define dso_local double @alldiv(double** noalias nocapture readonly %x) #2 {
entry:
  %0 = load double*, double** %x, align 8, !tbaa !2
  %call = call double @loadSq(double** nonnull %x)
  store double 0x400921FB53C8D4F1, double* %0, align 8, !tbaa !6
  ret double %call
}

define double @meta(double** %xx, double** %dxx) {
entry:
  %call = call double (i8*, ...) @__enzyme_autodiff(i8* bitcast (double (double**)* @alldiv to i8*), double** nonnull %xx, double** nonnull %dxx) #8
  ret double %call
}

declare dso_local double @__enzyme_autodiff(i8*, ...)

attributes #0 = { noinline norecurse nounwind readonly }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 12.0.1 (git@github.com:llvm/llvm-project 4973ce53ca8abfc14233a3d8b3045673e0e8543c)"}
!2 = !{!3, !3, i64 0}
!3 = !{!"any pointer", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
!6 = !{!7, !7, i64 0}
!7 = !{!"double", !4, i64 0}

; CHECK: define internal double @augmented_loadSq(double** noalias nocapture readonly %x, double** nocapture %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %l0 = load double*, double** %x, align 8, !tbaa !
; CHECK-NEXT:   %l1 = load double, double* %l0, align 8, !tbaa !
; CHECK-NEXT:   ret double %l1
; CHECK-NEXT: }

; CHECK: define internal void @diffeloadSq(double** noalias nocapture readonly %x, double** nocapture %"x'", double %differeturn, double %l1)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"l0'ipl" = load double*, double** %"x'", align 8
; CHECK-NEXT:   %m0diffel1 = fmul fast double %differeturn, %l1
; CHECK-NEXT:   %0 = fadd fast double %m0diffel1, %m0diffel1
; CHECK-NEXT:   %1 = load double, double* %"l0'ipl", align 8
; CHECK-NEXT:   %2 = fadd fast double %1, %0
; CHECK-NEXT:   store double %2, double* %"l0'ipl", align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
