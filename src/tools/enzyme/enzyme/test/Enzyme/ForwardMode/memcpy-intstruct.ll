; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

; Function Attrs: nounwind uwtable
define dso_local void @memcpy_ptr(i8* nocapture %dst, i8* nocapture readonly %src, i64 %num) {
entry:
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %dst, i8* align 8 %src, i64 %num, i1 false), !tbaa !17, !tbaa.struct !19
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1) #0

; Function Attrs: nounwind uwtable
define dso_local void @dmemcpy_ptr(i8* %dst, i8* %dstp, i8* %src, i8* %srcp, i64 %n) {
entry:
  %0 = tail call double (...) @__enzyme_fwddiff.f64(void (i8*, i8*, i64)* nonnull @memcpy_ptr, metadata !"enzyme_dup", i8* %dst, i8* %dstp, metadata !"enzyme_dup", i8* %src, i8* %srcp, i64 %n)
  ret void
}

declare double @__enzyme_fwddiff.f64(...) local_unnamed_addr

attributes #0 = { argmemonly nounwind }

!17 = !{!18, !18, i64 0, i64 32}
!18 = !{!4, i64 32, !"_ZTSSt5arrayIlLm4EE", !9, i64 0, i64 32}

!19 = !{i64 0, i64 32, !20}
!20 = !{!9, !9, i64 0, i64 32}
!9 = !{!4, i64 8, !"long"}
!4 = !{!5, i64 1, !"omnipotent char"}
!5 = !{!"Simple C++ TBAA"}


; CHECK: define internal void @fwddiffememcpy_ptr(i8* nocapture %dst, i8* nocapture %"dst'", i8* nocapture readonly %src, i8* nocapture %"src'", i64 %num)
; CHECK-NEXT: entry:
; CHECK-NEXT:   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %dst, i8* align 8 %src, i64 %num, i1 false)
; CHECK-NEXT:   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %"dst'", i8* align 8 %"src'", i64 %num, i1 false)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
