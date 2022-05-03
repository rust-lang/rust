; RUN: %opt < %s %loadEnzyme -print-activity-analysis -activity-analysis-func=matvec -activity-analysis-inactive-args -o /dev/null | FileCheck %s

source_filename = "text"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:10:11:12:13"
target triple = "x86_64-pc-linux-gnu"

declare float** @jl_array_copy()

define float @matvec({} addrspace(10)* nocapture nonnull readonly align 16 dereferenceable(40) %arg, {} addrspace(10)* nonnull align 16 dereferenceable(40) %arg1, i8 zeroext %arg2) {
entry:
  %i10 = call noalias float** @jl_array_copy()
  %i11 = load float*, float** %i10, align 8
  %i12 = load float, float* %i11, align 4;, !tbaa !21
  ret float %i12
}

; CHECK: {} addrspace(10)* %arg: icv:1
; CHECK: {} addrspace(10)* %arg1: icv:1
; CHECK: i8 %arg2: icv:1
; CHECK: entry
; CHECK-NEXT:   %i10 = call noalias float** @jl_array_copy(): icv:1 ici:1
; CHECK-NEXT:   %i11 = load float*, float** %i10, align 8: icv:1 ici:1
; CHECK-NEXT:   %i12 = load float, float* %i11, align 4: icv:1 ici:1
; CHECK-NEXT:   ret float %i12: icv:1 ici:1
