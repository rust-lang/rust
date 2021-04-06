; RUN: %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=caller -o /dev/null | FileCheck %s

source_filename = "cudaMM.cu"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@_ZZ19gpu_square_elem_mulPfS_S_mE6tile_a = internal unnamed_addr addrspace(3) global [2 x float] undef, align 32

; Function Attrs: convergent nounwind
define dso_local void @caller() {
bb:
  %gep = getelementptr [2 x float], [2 x float] addrspace(3)* @_ZZ19gpu_square_elem_mulPfS_S_mE6tile_a, i32 0, i64 0
  %tmp18 = load float, float addrspace(3)* %gep, align 4, !tbaa !13
  %gep3 = getelementptr [2 x float], [2 x float] addrspace(3)* @_ZZ19gpu_square_elem_mulPfS_S_mE6tile_a, i32 0, i64 1
  %tmp38 = load float, float addrspace(3)* %gep3, align 4, !tbaa !13
  ret void
}

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!8}
!nvvmir.version = !{!9}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 10, i32 1]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!4 = !{null, !"align", i32 8}
!5 = !{null, !"align", i32 8, !"align", i32 65544, !"align", i32 131080}
!6 = !{null, !"align", i32 16}
!7 = !{null, !"align", i32 16, !"align", i32 65552, !"align", i32 131088}
!8 = !{!"Ubuntu clang version 10.0.1-++20200809072545+ef32c611aa2-1~exp1~20200809173142.193"}
!9 = !{i32 1, i32 4}
!10 = !{i32 0, i32 65535}
!11 = !{i32 0, i32 1024}
!12 = !{i32 0, i32 2147483647}
!13 = !{!14, !14, i64 0}
!14 = !{!"float", !15, i64 0}
!15 = !{!"omnipotent char", !16, i64 0}
!16 = !{!"Simple C++ TBAA"}

; CHECK: caller - {} |
; CHECK-NEXT: bb
; CHECK-NEXT:   %gep = getelementptr [2 x float], [2 x float] addrspace(3)* @_ZZ19gpu_square_elem_mulPfS_S_mE6tile_a, i32 0, i64 0: {[-1]:Pointer, [-1,-1]:Float@float}
; CHECK-NEXT:   %tmp18 = load float, float addrspace(3)* %gep, align 4, !tbaa !5: {[-1]:Float@float}
; CHECK-NEXT:   %gep3 = getelementptr [2 x float], [2 x float] addrspace(3)* @_ZZ19gpu_square_elem_mulPfS_S_mE6tile_a, i32 0, i64 1: {[-1]:Pointer, [-1,0]:Float@float}
; CHECK-NEXT:   %tmp38 = load float, float addrspace(3)* %gep3, align 4, !tbaa !5: {[-1]:Float@float}
; CHECK-NEXT:   ret void: {}