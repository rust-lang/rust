; RUN: if [ %llvmver -ge 9 ]; then %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=tload -o /dev/null | FileCheck %s --check-prefixes=LCHECK; fi
; RUN: if [ %llvmver -ge 9 ]; then %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=tstore -o /dev/null | FileCheck %s --check-prefixes=SCHECK; fi
; RUN: if [ %llvmver -ge 9 ]; then %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=tmm -o /dev/null | FileCheck %s --check-prefixes=MCHECK; fi

; ModuleID = 'cuda.cu'
source_filename = "cuda.cu"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

declare  { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.load.a.col.stride.f16.p1i8(i8 addrspace(1)*, i32)

define void @tload(i8 addrspace(1)* %in) {
entry:
  %res = call { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.load.a.col.stride.f16.p1i8(i8 addrspace(1)* %in, i32 16)
  ret void
}

; LCHECK: tload - {} |{[-1]:Pointer}:{} 
; LCHECK-NEXT: i8 addrspace(1)* %in: {[-1]:Pointer, [-1,0]:Float@half}
; LCHECK-NEXT: entry
; LCHECK-NEXT:   %res = call { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.load.a.col.stride.f16.p1i8(i8 addrspace(1)* %in, i32 16): {[-1]:Float@half}
; LCHECK-NEXT:   ret void: {}

declare void @llvm.nvvm.wmma.m16n16k16.store.d.col.stride.f32.p1i8(i8 addrspace(1)*, float, float, float, float, float, float, float, float, i32)

define void @tstore(i8 addrspace(1)* %out, i8* %in) {
entry:
  %p = bitcast i8* %in to float*
  %l0 = load float, float* %p, align 4
  %p1 = getelementptr float, float* %p, i64 1
  %l1 = load float, float* %p1, align 4
  %p2 = getelementptr float, float* %p, i64 2
  %l2 = load float, float* %p2, align 4
  %p3 = getelementptr float, float* %p, i64 3
  %l3 = load float, float* %p3, align 4
  %p4 = getelementptr float, float* %p, i64 4
  %l4 = load float, float* %p4, align 4
  %p5 = getelementptr float, float* %p, i64 5
  %l5 = load float, float* %p5, align 4
  %p6 = getelementptr float, float* %p, i64 6
  %l6 = load float, float* %p6, align 4
  %p7 = getelementptr float, float* %p, i64 7
  %l7 = load float, float* %p7, align 4
  call void @llvm.nvvm.wmma.m16n16k16.store.d.col.stride.f32.p1i8(i8 addrspace(1)* %out, float %l0, float %l1, float %l2, float %l3, float %l4, float %l5, float %l6, float %l7, i32 16)
  ret void
}

; SCHECK: tstore - {} |{[-1]:Pointer}:{} {[-1]:Pointer}:{} 
; SCHECK-NEXT: i8 addrspace(1)* %out: {[-1]:Pointer, [-1,0]:Float@float}
; SCHECK-NEXT: i8* %in: {[-1]:Pointer, [-1,0]:Float@float, [-1,4]:Float@float, [-1,8]:Float@float, [-1,12]:Float@float, [-1,16]:Float@float, [-1,20]:Float@float, [-1,24]:Float@float, [-1,28]:Float@float}
; SCHECK-NEXT: entry
; SCHECK-NEXT:   %p = bitcast i8* %in to float*: {[-1]:Pointer, [-1,0]:Float@float, [-1,4]:Float@float, [-1,8]:Float@float, [-1,12]:Float@float, [-1,16]:Float@float, [-1,20]:Float@float, [-1,24]:Float@float, [-1,28]:Float@float}
; SCHECK-NEXT:   %l0 = load float, float* %p, align 4: {[-1]:Float@float}
; SCHECK-NEXT:   %p1 = getelementptr float, float* %p, i64 1: {[-1]:Pointer, [-1,0]:Float@float, [-1,4]:Float@float, [-1,8]:Float@float, [-1,12]:Float@float, [-1,16]:Float@float, [-1,20]:Float@float, [-1,24]:Float@float}
; SCHECK-NEXT:   %l1 = load float, float* %p1, align 4: {[-1]:Float@float}
; SCHECK-NEXT:   %p2 = getelementptr float, float* %p, i64 2: {[-1]:Pointer, [-1,0]:Float@float, [-1,4]:Float@float, [-1,8]:Float@float, [-1,12]:Float@float, [-1,16]:Float@float, [-1,20]:Float@float}
; SCHECK-NEXT:   %l2 = load float, float* %p2, align 4: {[-1]:Float@float}
; SCHECK-NEXT:   %p3 = getelementptr float, float* %p, i64 3: {[-1]:Pointer, [-1,0]:Float@float, [-1,4]:Float@float, [-1,8]:Float@float, [-1,12]:Float@float, [-1,16]:Float@float}
; SCHECK-NEXT:   %l3 = load float, float* %p3, align 4: {[-1]:Float@float}
; SCHECK-NEXT:   %p4 = getelementptr float, float* %p, i64 4: {[-1]:Pointer, [-1,0]:Float@float, [-1,4]:Float@float, [-1,8]:Float@float, [-1,12]:Float@float}
; SCHECK-NEXT:   %l4 = load float, float* %p4, align 4: {[-1]:Float@float}
; SCHECK-NEXT:   %p5 = getelementptr float, float* %p, i64 5: {[-1]:Pointer, [-1,0]:Float@float, [-1,4]:Float@float, [-1,8]:Float@float}
; SCHECK-NEXT:   %l5 = load float, float* %p5, align 4: {[-1]:Float@float}
; SCHECK-NEXT:   %p6 = getelementptr float, float* %p, i64 6: {[-1]:Pointer, [-1,0]:Float@float, [-1,4]:Float@float}
; SCHECK-NEXT:   %l6 = load float, float* %p6, align 4: {[-1]:Float@float}
; SCHECK-NEXT:   %p7 = getelementptr float, float* %p, i64 7: {[-1]:Pointer, [-1,0]:Float@float}
; SCHECK-NEXT:   %l7 = load float, float* %p7, align 4: {[-1]:Float@float}
; SCHECK-NEXT:   call void @llvm.nvvm.wmma.m16n16k16.store.d.col.stride.f32.p1i8(i8 addrspace(1)* %out, float %l0, float %l1, float %l2, float %l3, float %l4, float %l5, float %l6, float %l7, i32 16): {}
; SCHECK-NEXT:   ret void: {}

declare { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k16.mma.col.col.f32.f32(<2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, float, float, float, float, float, float, float, float)

define void @tmm(i8 addrspace(1)* %out, i8* %in, i8* %ain, i8* %bin) {
entry:
  %p = bitcast i8* %in to float*
  %l0 = load float, float* %p, align 4
  %p1 = getelementptr float, float* %p, i64 1
  %l1 = load float, float* %p1, align 4
  %p2 = getelementptr float, float* %p, i64 2
  %l2 = load float, float* %p2, align 4
  %p3 = getelementptr float, float* %p, i64 3
  %l3 = load float, float* %p3, align 4
  %p4 = getelementptr float, float* %p, i64 4
  %l4 = load float, float* %p4, align 4
  %p5 = getelementptr float, float* %p, i64 5
  %l5 = load float, float* %p5, align 4
  %p6 = getelementptr float, float* %p, i64 6
  %l6 = load float, float* %p6, align 4
  %p7 = getelementptr float, float* %p, i64 7
  %l7 = load float, float* %p7, align 4

  %fp = bitcast i8* %ain to <2 x half>*
  %f0 = load <2 x half>, <2 x half>* %fp, align 4
  %fp1 = getelementptr <2 x half>, <2 x half>* %fp, i64 1
  %f1 = load <2 x half>, <2 x half>* %fp1, align 4
  %fp2 = getelementptr <2 x half>, <2 x half>* %fp, i64 2
  %f2 = load <2 x half>, <2 x half>* %fp2, align 4
  %fp3 = getelementptr <2 x half>, <2 x half>* %fp, i64 3
  %f3 = load <2 x half>, <2 x half>* %fp3, align 4
  %fp4 = getelementptr <2 x half>, <2 x half>* %fp, i64 4
  %f4 = load <2 x half>, <2 x half>* %fp4, align 4
  %fp5 = getelementptr <2 x half>, <2 x half>* %fp, i64 5
  %f5 = load <2 x half>, <2 x half>* %fp5, align 4
  %fp6 = getelementptr <2 x half>, <2 x half>* %fp, i64 6
  %f6 = load <2 x half>, <2 x half>* %fp6, align 4
  %fp7 = getelementptr <2 x half>, <2 x half>* %fp, i64 7
  %f7 = load <2 x half>, <2 x half>* %fp7, align 4

  %bp = bitcast i8* %bin to <2 x half>*
  %b0 = load <2 x half>, <2 x half>* %bp, align 4
  %bp1 = getelementptr <2 x half>, <2 x half>* %bp, i64 1
  %b1 = load <2 x half>, <2 x half>* %bp1, align 4
  %bp2 = getelementptr <2 x half>, <2 x half>* %bp, i64 2
  %b2 = load <2 x half>, <2 x half>* %bp2, align 4
  %bp3 = getelementptr <2 x half>, <2 x half>* %bp, i64 3
  %b3 = load <2 x half>, <2 x half>* %bp3, align 4
  %bp4 = getelementptr <2 x half>, <2 x half>* %bp, i64 4
  %b4 = load <2 x half>, <2 x half>* %bp4, align 4
  %bp5 = getelementptr <2 x half>, <2 x half>* %bp, i64 5
  %b5 = load <2 x half>, <2 x half>* %bp5, align 4
  %bp6 = getelementptr <2 x half>, <2 x half>* %bp, i64 6
  %b6 = load <2 x half>, <2 x half>* %bp6, align 4
  %bp7 = getelementptr <2 x half>, <2 x half>* %bp, i64 7
  %b7 = load <2 x half>, <2 x half>* %bp7, align 4

  %res = call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k16.mma.col.col.f32.f32(<2 x half> %f0, <2 x half> %f1, <2 x half> %f2, <2 x half> %f3, <2 x half> %f4, <2 x half> %f5, <2 x half> %f6, <2 x half> %f7, <2 x half> %b0, <2 x half> %b1, <2 x half> %b2, <2 x half> %b3, <2 x half> %b4, <2 x half> %b5, <2 x half> %b6, <2 x half> %b7, float %l0, float %l1, float %l2, float %l3, float %l4, float %l5, float %l6, float %l7)
  ret void
}

; MCHECK: tmm - {} |{[-1]:Pointer}:{} {[-1]:Pointer}:{} {[-1]:Pointer}:{} {[-1]:Pointer}:{} 
; MCHECK-NEXT: i8 addrspace(1)* %out: {[-1]:Pointer}
; MCHECK-NEXT: i8* %in: {[-1]:Pointer, [-1,0]:Float@float, [-1,4]:Float@float, [-1,8]:Float@float, [-1,12]:Float@float, [-1,16]:Float@float, [-1,20]:Float@float, [-1,24]:Float@float, [-1,28]:Float@float}
; MCHECK-NEXT: i8* %ain: {[-1]:Pointer, [-1,0]:Float@half, [-1,2]:Float@half, [-1,4]:Float@half, [-1,6]:Float@half, [-1,8]:Float@half, [-1,10]:Float@half, [-1,12]:Float@half, [-1,14]:Float@half, [-1,16]:Float@half, [-1,18]:Float@half, [-1,20]:Float@half, [-1,22]:Float@half, [-1,24]:Float@half, [-1,26]:Float@half, [-1,28]:Float@half, [-1,30]:Float@half}
; MCHECK-NEXT: i8* %bin: {[-1]:Pointer, [-1,0]:Float@half, [-1,2]:Float@half, [-1,4]:Float@half, [-1,6]:Float@half, [-1,8]:Float@half, [-1,10]:Float@half, [-1,12]:Float@half, [-1,14]:Float@half, [-1,16]:Float@half, [-1,18]:Float@half, [-1,20]:Float@half, [-1,22]:Float@half, [-1,24]:Float@half, [-1,26]:Float@half, [-1,28]:Float@half, [-1,30]:Float@half}
; MCHECK-NEXT: entry
; MCHECK-NEXT:   %p = bitcast i8* %in to float*: {[-1]:Pointer, [-1,0]:Float@float, [-1,4]:Float@float, [-1,8]:Float@float, [-1,12]:Float@float, [-1,16]:Float@float, [-1,20]:Float@float, [-1,24]:Float@float, [-1,28]:Float@float}
; MCHECK-NEXT:   %l0 = load float, float* %p, align 4: {[-1]:Float@float}
; MCHECK-NEXT:   %p1 = getelementptr float, float* %p, i64 1: {[-1]:Pointer, [-1,0]:Float@float, [-1,4]:Float@float, [-1,8]:Float@float, [-1,12]:Float@float, [-1,16]:Float@float, [-1,20]:Float@float, [-1,24]:Float@float}
; MCHECK-NEXT:   %l1 = load float, float* %p1, align 4: {[-1]:Float@float}
; MCHECK-NEXT:   %p2 = getelementptr float, float* %p, i64 2: {[-1]:Pointer, [-1,0]:Float@float, [-1,4]:Float@float, [-1,8]:Float@float, [-1,12]:Float@float, [-1,16]:Float@float, [-1,20]:Float@float}
; MCHECK-NEXT:   %l2 = load float, float* %p2, align 4: {[-1]:Float@float}
; MCHECK-NEXT:   %p3 = getelementptr float, float* %p, i64 3: {[-1]:Pointer, [-1,0]:Float@float, [-1,4]:Float@float, [-1,8]:Float@float, [-1,12]:Float@float, [-1,16]:Float@float}
; MCHECK-NEXT:   %l3 = load float, float* %p3, align 4: {[-1]:Float@float}
; MCHECK-NEXT:   %p4 = getelementptr float, float* %p, i64 4: {[-1]:Pointer, [-1,0]:Float@float, [-1,4]:Float@float, [-1,8]:Float@float, [-1,12]:Float@float}
; MCHECK-NEXT:   %l4 = load float, float* %p4, align 4: {[-1]:Float@float}
; MCHECK-NEXT:   %p5 = getelementptr float, float* %p, i64 5: {[-1]:Pointer, [-1,0]:Float@float, [-1,4]:Float@float, [-1,8]:Float@float}
; MCHECK-NEXT:   %l5 = load float, float* %p5, align 4: {[-1]:Float@float}
; MCHECK-NEXT:   %p6 = getelementptr float, float* %p, i64 6: {[-1]:Pointer, [-1,0]:Float@float, [-1,4]:Float@float}
; MCHECK-NEXT:   %l6 = load float, float* %p6, align 4: {[-1]:Float@float}
; MCHECK-NEXT:   %p7 = getelementptr float, float* %p, i64 7: {[-1]:Pointer, [-1,0]:Float@float}
; MCHECK-NEXT:   %l7 = load float, float* %p7, align 4: {[-1]:Float@float}
; MCHECK-NEXT:   %fp = bitcast i8* %ain to <2 x half>*: {[-1]:Pointer, [-1,0]:Float@half, [-1,2]:Float@half, [-1,4]:Float@half, [-1,6]:Float@half, [-1,8]:Float@half, [-1,10]:Float@half, [-1,12]:Float@half, [-1,14]:Float@half, [-1,16]:Float@half, [-1,18]:Float@half, [-1,20]:Float@half, [-1,22]:Float@half, [-1,24]:Float@half, [-1,26]:Float@half, [-1,28]:Float@half, [-1,30]:Float@half}
; MCHECK-NEXT:   %f0 = load <2 x half>, <2 x half>* %fp, align 4: {[-1]:Float@half}
; MCHECK-NEXT:   %fp1 = getelementptr <2 x half>, <2 x half>* %fp, i64 1: {[-1]:Pointer, [-1,0]:Float@half, [-1,2]:Float@half, [-1,4]:Float@half, [-1,6]:Float@half, [-1,8]:Float@half, [-1,10]:Float@half, [-1,12]:Float@half, [-1,14]:Float@half, [-1,16]:Float@half, [-1,18]:Float@half, [-1,20]:Float@half, [-1,22]:Float@half, [-1,24]:Float@half, [-1,26]:Float@half}
; MCHECK-NEXT:   %f1 = load <2 x half>, <2 x half>* %fp1, align 4: {[-1]:Float@half}
; MCHECK-NEXT:   %fp2 = getelementptr <2 x half>, <2 x half>* %fp, i64 2: {[-1]:Pointer, [-1,0]:Float@half, [-1,2]:Float@half, [-1,4]:Float@half, [-1,6]:Float@half, [-1,8]:Float@half, [-1,10]:Float@half, [-1,12]:Float@half, [-1,14]:Float@half, [-1,16]:Float@half, [-1,18]:Float@half, [-1,20]:Float@half, [-1,22]:Float@half}
; MCHECK-NEXT:   %f2 = load <2 x half>, <2 x half>* %fp2, align 4: {[-1]:Float@half}
; MCHECK-NEXT:   %fp3 = getelementptr <2 x half>, <2 x half>* %fp, i64 3: {[-1]:Pointer, [-1,0]:Float@half, [-1,2]:Float@half, [-1,4]:Float@half, [-1,6]:Float@half, [-1,8]:Float@half, [-1,10]:Float@half, [-1,12]:Float@half, [-1,14]:Float@half, [-1,16]:Float@half, [-1,18]:Float@half}
; MCHECK-NEXT:   %f3 = load <2 x half>, <2 x half>* %fp3, align 4: {[-1]:Float@half}
; MCHECK-NEXT:   %fp4 = getelementptr <2 x half>, <2 x half>* %fp, i64 4: {[-1]:Pointer, [-1,0]:Float@half, [-1,2]:Float@half, [-1,4]:Float@half, [-1,6]:Float@half, [-1,8]:Float@half, [-1,10]:Float@half, [-1,12]:Float@half, [-1,14]:Float@half}
; MCHECK-NEXT:   %f4 = load <2 x half>, <2 x half>* %fp4, align 4: {[-1]:Float@half}
; MCHECK-NEXT:   %fp5 = getelementptr <2 x half>, <2 x half>* %fp, i64 5: {[-1]:Pointer, [-1,0]:Float@half, [-1,2]:Float@half, [-1,4]:Float@half, [-1,6]:Float@half, [-1,8]:Float@half, [-1,10]:Float@half}
; MCHECK-NEXT:   %f5 = load <2 x half>, <2 x half>* %fp5, align 4: {[-1]:Float@half}
; MCHECK-NEXT:   %fp6 = getelementptr <2 x half>, <2 x half>* %fp, i64 6: {[-1]:Pointer, [-1,0]:Float@half, [-1,2]:Float@half, [-1,4]:Float@half, [-1,6]:Float@half}
; MCHECK-NEXT:   %f6 = load <2 x half>, <2 x half>* %fp6, align 4: {[-1]:Float@half}
; MCHECK-NEXT:   %fp7 = getelementptr <2 x half>, <2 x half>* %fp, i64 7: {[-1]:Pointer, [-1,0]:Float@half, [-1,2]:Float@half}
; MCHECK-NEXT:   %f7 = load <2 x half>, <2 x half>* %fp7, align 4: {[-1]:Float@half}
; MCHECK-NEXT:   %bp = bitcast i8* %bin to <2 x half>*: {[-1]:Pointer, [-1,0]:Float@half, [-1,2]:Float@half, [-1,4]:Float@half, [-1,6]:Float@half, [-1,8]:Float@half, [-1,10]:Float@half, [-1,12]:Float@half, [-1,14]:Float@half, [-1,16]:Float@half, [-1,18]:Float@half, [-1,20]:Float@half, [-1,22]:Float@half, [-1,24]:Float@half, [-1,26]:Float@half, [-1,28]:Float@half, [-1,30]:Float@half}
; MCHECK-NEXT:   %b0 = load <2 x half>, <2 x half>* %bp, align 4: {[-1]:Float@half}
; MCHECK-NEXT:   %bp1 = getelementptr <2 x half>, <2 x half>* %bp, i64 1: {[-1]:Pointer, [-1,0]:Float@half, [-1,2]:Float@half, [-1,4]:Float@half, [-1,6]:Float@half, [-1,8]:Float@half, [-1,10]:Float@half, [-1,12]:Float@half, [-1,14]:Float@half, [-1,16]:Float@half, [-1,18]:Float@half, [-1,20]:Float@half, [-1,22]:Float@half, [-1,24]:Float@half, [-1,26]:Float@half}
; MCHECK-NEXT:   %b1 = load <2 x half>, <2 x half>* %bp1, align 4: {[-1]:Float@half}
; MCHECK-NEXT:   %bp2 = getelementptr <2 x half>, <2 x half>* %bp, i64 2: {[-1]:Pointer, [-1,0]:Float@half, [-1,2]:Float@half, [-1,4]:Float@half, [-1,6]:Float@half, [-1,8]:Float@half, [-1,10]:Float@half, [-1,12]:Float@half, [-1,14]:Float@half, [-1,16]:Float@half, [-1,18]:Float@half, [-1,20]:Float@half, [-1,22]:Float@half}
; MCHECK-NEXT:   %b2 = load <2 x half>, <2 x half>* %bp2, align 4: {[-1]:Float@half}
; MCHECK-NEXT:   %bp3 = getelementptr <2 x half>, <2 x half>* %bp, i64 3: {[-1]:Pointer, [-1,0]:Float@half, [-1,2]:Float@half, [-1,4]:Float@half, [-1,6]:Float@half, [-1,8]:Float@half, [-1,10]:Float@half, [-1,12]:Float@half, [-1,14]:Float@half, [-1,16]:Float@half, [-1,18]:Float@half}
; MCHECK-NEXT:   %b3 = load <2 x half>, <2 x half>* %bp3, align 4: {[-1]:Float@half}
; MCHECK-NEXT:   %bp4 = getelementptr <2 x half>, <2 x half>* %bp, i64 4: {[-1]:Pointer, [-1,0]:Float@half, [-1,2]:Float@half, [-1,4]:Float@half, [-1,6]:Float@half, [-1,8]:Float@half, [-1,10]:Float@half, [-1,12]:Float@half, [-1,14]:Float@half}
; MCHECK-NEXT:   %b4 = load <2 x half>, <2 x half>* %bp4, align 4: {[-1]:Float@half}
; MCHECK-NEXT:   %bp5 = getelementptr <2 x half>, <2 x half>* %bp, i64 5: {[-1]:Pointer, [-1,0]:Float@half, [-1,2]:Float@half, [-1,4]:Float@half, [-1,6]:Float@half, [-1,8]:Float@half, [-1,10]:Float@half}
; MCHECK-NEXT:   %b5 = load <2 x half>, <2 x half>* %bp5, align 4: {[-1]:Float@half}
; MCHECK-NEXT:   %bp6 = getelementptr <2 x half>, <2 x half>* %bp, i64 6: {[-1]:Pointer, [-1,0]:Float@half, [-1,2]:Float@half, [-1,4]:Float@half, [-1,6]:Float@half}
; MCHECK-NEXT:   %b6 = load <2 x half>, <2 x half>* %bp6, align 4: {[-1]:Float@half}
; MCHECK-NEXT:   %bp7 = getelementptr <2 x half>, <2 x half>* %bp, i64 7: {[-1]:Pointer, [-1,0]:Float@half, [-1,2]:Float@half}
; MCHECK-NEXT:   %b7 = load <2 x half>, <2 x half>* %bp7, align 4: {[-1]:Float@half}
; MCHECK-NEXT:   %res = call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k16.mma.col.col.f32.f32(<2 x half> %f0, <2 x half> %f1, <2 x half> %f2, <2 x half> %f3, <2 x half> %f4, <2 x half> %f5, <2 x half> %f6, <2 x half> %f7, <2 x half> %b0, <2 x half> %b1, <2 x half> %b2, <2 x half> %b3, <2 x half> %b4, <2 x half> %b5, <2 x half> %b6, <2 x half> %b7, float %l0, float %l1, float %l2, float %l3, float %l4, float %l5, float %l6, float %l7): {[-1]:Float@float}
; MCHECK-NEXT:   ret void: {}
