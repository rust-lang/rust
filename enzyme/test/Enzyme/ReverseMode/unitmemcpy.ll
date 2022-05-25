; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -instsimplify -S | FileCheck %s

source_filename = "<source>"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@enzyme_dup = dso_local global i32 0, align 4

define dso_local void @_Z6squarePi(i8* %i0) {
  %i2 = call noalias i8* @malloc(i64 16)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %i0, i8* %i2, i64 16, i1 false)
  ret void
}

; Function Attrs: nounwind
declare dso_local noalias i8* @malloc(i64) #1

declare void @llvm.memcpy.p0i8.p0i8.i64(i8*, i8*, i64, i1)

define dso_local void @_Z7dsquarePdS_(double* %a0, double* %a1) {
  %ed = load i32, i32* @enzyme_dup, align 4
  call void @_Z17__enzyme_autodiffPviPdS0_(i8* bitcast (void (i8*)* @_Z6squarePi to i8*), i32 %ed, double* %a0, double* %a1)
  ret void
}

declare dso_local void @_Z17__enzyme_autodiffPviPdS0_(i8*, i32, double*, double*)

; CHECK: define internal void @diffe_Z6squarePi(i8* %i0, i8* %"i0'")
; CHECK-NEXT:   %i2 = call noalias nonnull dereferenceable(16) dereferenceable_or_null(16) i8* @malloc(i64 16)
; CHECK-NEXT:   %"i2'mi" = call noalias nonnull dereferenceable(16) dereferenceable_or_null(16) i8* @malloc(i64 16)
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull dereferenceable(16) dereferenceable_or_null(16) %"i2'mi", i8 0, i64 16, i1 false)
; CHECK-NEXT:   br label %invert

; CHECK: invert: 
; CHECK-NEXT:   tail call void @free(i8* nonnull %"i2'mi")
; CHECK-NEXT:   tail call void @free(i8* nonnull %i2)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
