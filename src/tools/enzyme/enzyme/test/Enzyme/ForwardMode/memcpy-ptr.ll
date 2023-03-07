; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

; Function Attrs: nounwind uwtable
define dso_local void @memcpy_ptr(double** nocapture %dst, double** nocapture readonly %src, i64 %num) #0 {
entry:
  %0 = bitcast double** %dst to i8*
  %1 = bitcast double** %src to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %0, i8* align 1 %1, i64 %num, i1 false)
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1) #1

; Function Attrs: nounwind uwtable
define dso_local void @dmemcpy_ptr(double** %dst, double** %dstp, double** %src, double** %srcp, i64 %n) local_unnamed_addr #0 {
entry:
  %0 = tail call double (...) @__enzyme_fwddiff.f64(void (double**, double**, i64)* nonnull @memcpy_ptr, double** %dst, double** %dstp, double** %src, double** %srcp, i64 %n) #3
  ret void
}

declare double @__enzyme_fwddiff.f64(...) local_unnamed_addr


attributes #0 = { nounwind uwtable }
attributes #1 = { argmemonly nounwind }
attributes #2 = { noinline nounwind uwtable }
attributes #3 = { nounwind }


; CHECK: define internal void @fwddiffememcpy_ptr(double** nocapture %dst, double** nocapture %"dst'", double** nocapture readonly %src, double** nocapture %"src'", i64 %num)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[ipc:.+]] = bitcast double** %"dst'" to i8*
; CHECK-NEXT:   %0 = bitcast double** %dst to i8*
; CHECK-NEXT:   %[[ipc1:.+]] = bitcast double** %"src'" to i8*
; CHECK-NEXT:   %1 = bitcast double** %src to i8*
; CHECK-NEXT:   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %0, i8* align 1 %1, i64 %num, i1 false)
; CHECK-NEXT:   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %[[ipc]], i8* align 1 %[[ipc1]], i64 %num, i1 false)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
