; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -simplifycfg -dce -instcombine -S | FileCheck %s

; Function Attrs: nounwind uwtable
define dso_local void @memcpy_float(double addrspace(13)* nocapture %dst, double addrspace(10)* nocapture readonly %src, i64 %num) #0 {
entry:
  %0 = bitcast double addrspace(13)* %dst to i8 addrspace(13)*
  %1 = bitcast double addrspace(10)* %src to i8 addrspace(10)*
  tail call void @llvm.memcpy.p13i8.p10i8.i64(i8 addrspace(13)* align 1 %0, i8 addrspace(10)* align 1 %1, i64 %num, i1 false)
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p13i8.p10i8.i64(i8 addrspace(13)* nocapture writeonly, i8 addrspace(10)* nocapture readonly, i64, i1) #1

; Function Attrs: nounwind uwtable
define dso_local void @dmemcpy_float(double addrspace(13)* %dst, double addrspace(13)* %dstp, double addrspace(10)* %src, double addrspace(10)* %srcp, i64 %n) local_unnamed_addr #0 {
entry:
  tail call void (...) @__enzyme_autodiff.f64(void (double addrspace(13)*, double addrspace(10)*, i64)* nonnull @memcpy_float, double addrspace(13)* %dst, double addrspace(13)* %dstp, double addrspace(10)* %src, double addrspace(10)* %srcp, i64 %n) #3
  ret void
}

declare void @__enzyme_autodiff.f64(...) 

attributes #0 = { nounwind uwtable }
attributes #1 = { argmemonly nounwind }
attributes #2 = { noinline nounwind uwtable }
attributes #3 = { nounwind }

; CHECK: define internal void @diffememcpy_float(double addrspace(13)* nocapture %dst, double addrspace(13)* nocapture %"dst'", double addrspace(10)* nocapture readonly %src, double addrspace(10)* nocapture %"src'", i64 %num)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = bitcast double addrspace(13)* %dst to i8 addrspace(13)*
; CHECK-NEXT:   %1 = bitcast double addrspace(10)* %src to i8 addrspace(10)*
; CHECK-NEXT:   tail call void @llvm.memcpy.p13i8.p10i8.i64(i8 addrspace(13)* align 1 %0, i8 addrspace(10)* align 1 %1, i64 %num, i1 false)
; CHECK-NEXT:   %2 = lshr i64 %num, 3
; CHECK-NEXT:   call void @__enzyme_memcpyadd_doubleda1sa1dadd13sadd10(double addrspace(13)* %"dst'", double addrspace(10)* %"src'", i64 %2)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
