; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -simplifycfg -early-cse -S | FileCheck %s 

source_filename = "exer2.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@__enzyme_register_splitderivative_add = dso_local local_unnamed_addr global [3 x i8*] [i8* bitcast (double (double, double)* @add to i8*), i8* bitcast ({ i8*, double, double } (double, double)* @add_aug to i8*), i8* bitcast ({ double, double } (double, double, double, double, i8*)* @add_err to i8*)], align 16

declare double @add(double %x, double %y) #0

declare { i8*, double, double } @add_aug(double %v1, double %v2)

declare { double, double } @add_err(double %v1, double %v1err, double %v2, double %v2err, i8* %tape)

; Function Attrs: norecurse nounwind readnone uwtable willreturn
define double @f(double %x) {
entry:
  %call = call double @add(double %x, double %x)
  %mul = fmul double %call, %call
  ret double %mul
}

; Function Attrs: nounwind uwtable
define double @caller(double %x, double %dx) {
entry:
  %call = call double (i8*, ...) @__enzyme_fwdsplit(i8* bitcast (double (double)* @f to i8*), double %x, double %dx, i8* null)
  ret double %call
}

declare dso_local double @__enzyme_fwdsplit(i8*, ...)

attributes #0 = { norecurse nounwind readnone }

; CHECK: define internal i8* @augmented_f(double %x, double %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %malloccall = tail call noalias nonnull dereferenceable(16) dereferenceable_or_null(16) i8* @malloc(i64 16)
; CHECK-NEXT:   %tapemem = bitcast i8* %malloccall to { i8*, double }*
; CHECK-NEXT:   %call_augmented = call { i8*, double, double } @add_aug(double %x, double %x)
; CHECK-NEXT:   %subcache = extractvalue { i8*, double, double } %call_augmented, 0
; CHECK-NEXT:   %0 = getelementptr inbounds { i8*, double }, { i8*, double }* %tapemem, i32 0, i32 0
; CHECK-NEXT:   store i8* %subcache, i8** %0
; CHECK-NEXT:   %call = extractvalue { i8*, double, double } %call_augmented, 1
; CHECK-NEXT:   %1 = getelementptr inbounds { i8*, double }, { i8*, double }* %tapemem, i32 0, i32 1
; CHECK-NEXT:   store double %call, double* %1
; CHECK-NEXT:   ret i8* %malloccall
; CHECK-NEXT: }

; CHECK: define internal double @fwddiffef(double %x, double %"x'", i8* %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = bitcast i8* %tapeArg to { i8*, double }*
; CHECK-NEXT:   %truetape = load { i8*, double }, { i8*, double }* %0
; CHECK-NEXT:   tail call void @free(i8* nonnull %tapeArg)
; CHECK-NEXT:   %tapeArg1 = extractvalue { i8*, double } %truetape, 0
; CHECK-NEXT:   %1 = call { double, double } @add_err(double %x, double %"x'", double %x, double %"x'", i8* %tapeArg1)
; CHECK-NEXT:   %2 = extractvalue { double, double } %1, 0
; CHECK-NEXT:   %3 = extractvalue { double, double } %1, 1
; CHECK-NEXT:   %4 = fmul fast double %3, %2
; CHECK-NEXT:   %5 = fadd fast double %4, %4
; CHECK-NEXT:   ret double %5
; CHECK-NEXT: }
