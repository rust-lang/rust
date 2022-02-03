; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -simplifycfg -instsimplify -gvn -adce -S | FileCheck %s
source_filename = "/app/example.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: mustprogress nofree noinline nosync nounwind readonly uwtable willreturn
define dso_local double @foo(double* nocapture readonly %x) {
entry:
  %0 = load double, double* %x, align 8
  %arrayidx1 = getelementptr inbounds double, double* %x, i64 2
  %1 = load double, double* %arrayidx1, align 8
  %f2 = tail call double @fma(double %0, double 2.000000e+00, double %1)
  %mul = fmul double %f2, %f2
  ret double %mul
}

declare double @fma(double, double, double) readnone

; Function Attrs: mustprogress nofree nosync nounwind uwtable willreturn
define dso_local double @square(double* nocapture %x) {
entry:
  %call = tail call double @foo(double* %x)
  store double 0.000000e+00, double* %x, align 8
  ret double %call
}

define dso_local double @dsquare(double* %x, double* %dx) {
entry:
  %call = call double (i8*, ...) @__enzyme_autodiff(i8* bitcast (double (double*)* @square to i8*), double* %x, double* %dx)
  ret double %call
}

declare dso_local double @__enzyme_autodiff(i8*, ...)

; CHECK: define internal double @augmented_foo(double* nocapture readonly %x, double* nocapture %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = load double, double* %x, align 8
; CHECK-NEXT:   %arrayidx1 = getelementptr inbounds double, double* %x, i64 2
; CHECK-NEXT:   %1 = load double, double* %arrayidx1, align 8
; CHECK-NEXT:   %f2 = tail call double @fma(double %0, double 2.000000e+00, double %1)
; CHECK-NEXT:   ret double %f2
; CHECK-NEXT: }

; CHECK: define internal void @diffefoo(double* nocapture readonly %x, double* nocapture %"x'", double %differeturn, double %f2)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"arrayidx1'ipg" = getelementptr inbounds double, double* %"x'", i64 2
; CHECK-NEXT:   %m0diffef2 = fmul fast double %differeturn, %f2
; CHECK-NEXT:   %0 = fadd fast double %m0diffef2, %m0diffef2
; CHECK-NEXT:   %1 = fmul fast double %0, 2.000000e+00
; CHECK-NEXT:   %2 = load double, double* %"arrayidx1'ipg", align 8
; CHECK-NEXT:   %3 = fadd fast double %2, %0
; CHECK-NEXT:   store double %3, double* %"arrayidx1'ipg", align 8
; CHECK-NEXT:   %4 = load double, double* %"x'", align 8
; CHECK-NEXT:   %5 = fadd fast double %4, %1
; CHECK-NEXT:   store double %5, double* %"x'", align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
