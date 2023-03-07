; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

; ModuleID = 'ptr12.ll'
source_filename = "ld-temp.o"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define { double } @foo2(double %i13) {
bb:
  %i151 = insertvalue { double } undef, double %i13, 0
  ret { double } %i151
}

declare i8* @_Z17__enzyme_virtualreversePv(...)

define void @caller() {
  %1 = tail call i8* (...) @_Z17__enzyme_virtualreversePv({ double } (double)* nonnull @foo2)
  ret void
}

; CHECK: define internal { i8*, { double } } @augmented_foo2(double %i13)
; CHECK-NEXT: bb:
; CHECK-NEXT:   %0 = alloca { i8*, { double } }
; CHECK-NEXT:   %1 = getelementptr inbounds { i8*, { double } }, { i8*, { double } }* %0, i32 0, i32 0
; CHECK-NEXT:   store i8* null, i8** %1
; CHECK-NEXT:   %i151 = insertvalue { double } undef, double %i13, 0
; CHECK-NEXT:   %2 = getelementptr inbounds { i8*, { double } }, { i8*, { double } }* %0, i32 0, i32 1
; CHECK-NEXT:   store { double } %i151, { double }* %2
; CHECK-NEXT:   %3 = load { i8*, { double } }, { i8*, { double } }* %0
; CHECK-NEXT:   ret { i8*, { double } } %3
; CHECK-NEXT: }

; CHECK: define internal { double } @diffefoo2(double %i13, { double } %differeturn, i8* %tapeArg)
; CHECK-NEXT: bb:
; CHECK-NEXT:   tail call void @free(i8* nonnull %tapeArg)
; CHECK-NEXT:   ret { double } %differeturn
; CHECK-NEXT: }
