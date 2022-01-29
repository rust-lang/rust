; RUN: if [ %llvmver -ge 9 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi

source_filename = "cudaMM.cu"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@_ZZ19gpu_square_elem_mulPfS_S_mE6tile_a = internal unnamed_addr addrspace(3) global float undef, align 32

; Function Attrs: convergent nounwind
define dso_local void @_Z19gpu_square_elem_mulPfS_S_m(float* nocapture readonly %arg, float* nocapture readonly %arg1, float* nocapture %arg2, i64 %arg3) {
bb:
  %tmp = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #4, !range !10
  %tmp4 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.y() #4, !range !11
  %tmp5 = add nuw nsw i32 %tmp4, %tmp
  %tmp6 = zext i32 %tmp5 to i64
  %tmp7 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #4, !range !12
  %tmp8 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4, !range !11
  %tmp9 = add nuw i32 %tmp8, %tmp7
  %tmp10 = zext i32 %tmp9 to i64
  %tmp11 = mul i64 %tmp6, %arg3
  %tmp12 = add i64 %tmp11, %tmp10
  %tmp13 = getelementptr inbounds float, float* %arg, i64 %tmp12
  %tmp14 = bitcast float* %tmp13 to i32*
  %tmp15 = load i32, i32* %tmp14, align 4, !tbaa !13
  store i32 %tmp15, i32* addrspacecast (i32 addrspace(3)* bitcast (float addrspace(3)* @_ZZ19gpu_square_elem_mulPfS_S_mE6tile_a to i32 addrspace(3)*) to i32*), align 4, !tbaa !13
  %tmp16 = getelementptr inbounds float, float* %arg1, i64 %tmp12
  %tmp17 = load float, float* %tmp16, align 4, !tbaa !13
  tail call void @llvm.nvvm.barrier0()
  %tmp18 = load float, float* addrspacecast (float addrspace(3)* @_ZZ19gpu_square_elem_mulPfS_S_mE6tile_a to float*), align 4, !tbaa !13
  %tmp19 = fmul contract float %tmp17, %tmp18
  %tmp20 = getelementptr inbounds float, float* %arg2, i64 %tmp12
  store float %tmp19, float* %tmp20, align 4, !tbaa !13
  ret void
}

; Function Attrs: convergent nounwind
declare void @llvm.nvvm.barrier0() #2

define void @_Z4axpyPfS_S_S_S_S_m(float* %a, float* %b, float* %c, float* %d, float* %e, float* %f, i64 %sz) {
bb:
  tail call void @_Z17__enzyme_autodiffPvS_S_S_S_S_S_m(i8* bitcast (void (float*, float*, float*, i64)* @_Z19gpu_square_elem_mulPfS_S_m to i8*), float* %a, float* %b, float* %c, float* %d, float* %e, float* %f, i64 %sz)
  ret void
}

declare void @_Z17__enzyme_autodiffPvS_S_S_S_S_S_m(i8*, float*, float*, float*, float*, float*, float*, i64)

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #3

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.y() #3

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #3

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.x() #3

attributes #0 = { argmemonly nounwind willreturn }
attributes #2 = { convergent nounwind }
attributes #3 = { nounwind readnone }
attributes #4 = { nounwind }

!llvm.module.flags = !{!0, !1, !2}
!nvvm.annotations = !{!3, !4, !5, !4, !6, !6, !6, !6, !7, !7, !6}
!llvm.ident = !{!8}
!nvvmir.version = !{!9}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 10, i32 1]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!3 = !{void (float*, float*, float*, float*, float*, float*, i64)* @_Z4axpyPfS_S_S_S_S_m, !"kernel", i32 1}
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

; CHECK: @_ZZ19gpu_square_elem_mulPfS_S_mE6tile_a_shadow = internal unnamed_addr addrspace(3) global float undef, align 32

; CHECK: define internal void @diffe_Z19gpu_square_elem_mulPfS_S_m(float* nocapture readonly %arg, float* nocapture %"arg'", float* nocapture readonly %arg1, float* nocapture %"arg1'", float* nocapture %arg2, float* nocapture %"arg2'", i64 %arg3)
; CHECK: bb:
; CHECK-NEXT:   %0 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
; CHECK-NEXT:   %1 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y()
; CHECK-NEXT:   %2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.z()
; CHECK-NEXT:   %3 = or i32 %0, %1
; CHECK-NEXT:   %4 = or i32 %3, %2
; CHECK-NEXT:   %5 = icmp eq i32 %4, 0
; CHECK-NEXT:   br i1 %5, label %shblock, label %invertbb

; CHECK: shblock:                                          ; preds = %bb
; CHECK-NEXT:   store float 0.000000e+00, float addrspace(3)* @_ZZ19gpu_square_elem_mulPfS_S_mE6tile_a_shadow, align 32
; CHECK-NEXT:   br label %invertbb

; CHECK: invertbb:                                         ; preds = %shblock, %bb
; CHECK-NEXT:   call void @llvm.nvvm.barrier0()
; CHECK-NEXT:   %tmp = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #{{.*}}, !range !12
; CHECK-NEXT:   %tmp4 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.y() #{{.*}}, !range !13
; CHECK-NEXT:   %tmp5 = add nuw nsw i32 %tmp4, %tmp
; CHECK-NEXT:   %tmp6 = zext i32 %tmp5 to i64
; CHECK-NEXT:   %tmp7 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #{{.*}}, !range !14
; CHECK-NEXT:   %tmp8 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #{{.*}}, !range !13
; CHECK-NEXT:   %tmp9 = add nuw i32 %tmp8, %tmp7
; CHECK-NEXT:   %tmp10 = zext i32 %tmp9 to i64
; CHECK-NEXT:   %tmp11 = mul i64 %tmp6, %arg3
; CHECK-NEXT:   %tmp12 = add i64 %tmp11, %tmp10
; CHECK-NEXT:   %"tmp13'ipg" = getelementptr inbounds float, float* %"arg'", i64 %tmp12
; CHECK-NEXT:   %tmp13 = getelementptr inbounds float, float* %arg, i64 %tmp12
; CHECK-NEXT:   %tmp14 = bitcast float* %tmp13 to i32*
; CHECK-NEXT:   %tmp15 = load i32, i32* %tmp14, align 4, !tbaa !15
; CHECK-NEXT:   store i32 %tmp15, i32* addrspacecast (i32 addrspace(3)* bitcast (float addrspace(3)* @_ZZ19gpu_square_elem_mulPfS_S_mE6tile_a to i32 addrspace(3)*) to i32*), align 4, !tbaa !15
; CHECK-NEXT:   %"tmp16'ipg" = getelementptr inbounds float, float* %"arg1'", i64 %tmp12
; CHECK-NEXT:   %tmp16 = getelementptr inbounds float, float* %arg1, i64 %tmp12
; CHECK-NEXT:   %tmp17 = load float, float* %tmp16, align 4, !tbaa !15
; CHECK-NEXT:   tail call void @llvm.nvvm.barrier0()
; CHECK-NEXT:   %tmp18 = load float, float* addrspacecast (float addrspace(3)* @_ZZ19gpu_square_elem_mulPfS_S_mE6tile_a to float*), align 4, !tbaa !15
; CHECK-NEXT:   %tmp19 = fmul contract float %tmp17, %tmp18
; CHECK-NEXT:   %"tmp20'ipg" = getelementptr inbounds float, float* %"arg2'", i64 %tmp12
; CHECK-NEXT:   %tmp20 = getelementptr inbounds float, float* %arg2, i64 %tmp12
; CHECK-NEXT:   store float %tmp19, float* %tmp20, align 4, !tbaa !15
; CHECK-NEXT:   %[[tload:.+]] = load float, float* %"tmp20'ipg", align 4
; CHECK-NEXT:   store float 0.000000e+00, float* %"tmp20'ipg", align 4
; CHECK-NEXT:   %m0diffetmp17 = fmul fast float %[[tload]], %tmp18
; CHECK-NEXT:   %m1diffetmp18 = fmul fast float %[[tload]], %tmp17
; CHECK-NEXT:   %{{.+}} = atomicrmw fadd float* addrspacecast (float addrspace(3)* @_ZZ19gpu_square_elem_mulPfS_S_mE6tile_a_shadow to float*), float %m1diffetmp18 monotonic
; CHECK-NEXT:   call void @llvm.nvvm.barrier0()
; CHECK-NEXT:   %{{.+}} = atomicrmw fadd float* %"tmp16'ipg", float %m0diffetmp17 monotonic
; CHECK-NEXT:   %[[shload:.+]] = load i32, i32* addrspacecast (i32 addrspace(3)* bitcast (float addrspace(3)* @_ZZ19gpu_square_elem_mulPfS_S_mE6tile_a_shadow to i32 addrspace(3)*) to i32*), align 4
; CHECK-NEXT:   store i32 0, i32* addrspacecast (i32 addrspace(3)* bitcast (float addrspace(3)* @_ZZ19gpu_square_elem_mulPfS_S_mE6tile_a_shadow to i32 addrspace(3)*) to i32*), align 4
; CHECK-NEXT:   %[[bc:.+]] = bitcast i32 %[[shload]] to float
; CHECK-NEXT:   %{{.+}} = atomicrmw fadd float* %"tmp13'ipg", float %[[bc]] monotonic
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
