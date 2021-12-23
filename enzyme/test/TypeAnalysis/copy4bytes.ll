; RUN: %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=copy -o /dev/null | FileCheck %s

source_filename = "nullcp.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @copy(double* nocapture %src) {
entry:
  %data = alloca [32 x i8], align 8
  %dst8 = bitcast [32 x i8]* %data to i8*
  %src8 = bitcast double* %src to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %dst8, i8* align 8 %src8, i64 16, i1 false)
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1)

; CHECK: copy - {} |{[-1]:Pointer, [-1,-1]:Float@double}:{}
; CHECK-NEXT: double* %src: {[-1]:Pointer, [-1,-1]:Float@double}
; CHECK-NEXT: entry
; CHECK-NEXT:   %data = alloca [32 x i8], align 8: {[-1]:Pointer, [-1,0]:Float@double, [-1,8]:Float@double}
; CHECK-NEXT:   %dst8 = bitcast [32 x i8]* %data to i8*: {[-1]:Pointer, [-1,0]:Float@double, [-1,8]:Float@double}
; CHECK-NEXT:   %src8 = bitcast double* %src to i8*: {[-1]:Pointer, [-1,-1]:Float@double}
; CHECK-NEXT:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %dst8, i8* align 8 %src8, i64 16, i1 false): {}
; CHECK-NEXT:   ret void: {}
