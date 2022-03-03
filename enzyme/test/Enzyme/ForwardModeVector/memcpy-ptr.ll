; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -simplifycfg -dce -S | FileCheck %s

; Function Attrs: nounwind
declare void @__enzyme_fwddiff.f64(...)

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
define dso_local void @dmemcpy_ptr(double** %dst, double** %dstp1, double** %dstp2, double** %dstp3, double** %src, double** %srcp1, double** %srcp2, double** %srcp3, i64 %n) local_unnamed_addr #0 {
entry:
  tail call void (...) @__enzyme_fwddiff.f64(void (double**, double**, i64)* nonnull @memcpy_ptr, metadata !"enzyme_width", i64 3, double** %dst, double** %dstp1, double** %dstp2, double** %dstp3, double** %src, double** %srcp1, double** %srcp2, double** %srcp3, i64 %n) #3
  ret void
}

attributes #0 = { nounwind uwtable }
attributes #1 = { argmemonly nounwind }
attributes #2 = { noinline nounwind uwtable }
attributes #3 = { nounwind }


; CHECK: define internal void @fwddiffe3memcpy_ptr(double** nocapture %dst, [3 x double**] %"dst'", double** nocapture readonly %src, [3 x double**] %"src'", i64 %num)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = extractvalue [3 x double**] %"dst'", 0
; CHECK-NEXT:   %"'ipc" = bitcast double** %0 to i8*
; CHECK-NEXT:   %1 = insertvalue [3 x i8*] undef, i8* %"'ipc", 0
; CHECK-NEXT:   %2 = extractvalue [3 x double**] %"dst'", 1
; CHECK-NEXT:   %"'ipc1" = bitcast double** %2 to i8*
; CHECK-NEXT:   %3 = insertvalue [3 x i8*] %1, i8* %"'ipc1", 1
; CHECK-NEXT:   %4 = extractvalue [3 x double**] %"dst'", 2
; CHECK-NEXT:   %"'ipc2" = bitcast double** %4 to i8*
; CHECK-NEXT:   %5 = insertvalue [3 x i8*] %3, i8* %"'ipc2", 2
; CHECK-NEXT:   %6 = bitcast double** %dst to i8*
; CHECK-NEXT:   %7 = extractvalue [3 x double**] %"src'", 0
; CHECK-NEXT:   %"'ipc3" = bitcast double** %7 to i8*
; CHECK-NEXT:   %8 = insertvalue [3 x i8*] undef, i8* %"'ipc3", 0
; CHECK-NEXT:   %9 = extractvalue [3 x double**] %"src'", 1
; CHECK-NEXT:   %"'ipc4" = bitcast double** %9 to i8*
; CHECK-NEXT:   %10 = insertvalue [3 x i8*] %8, i8* %"'ipc4", 1
; CHECK-NEXT:   %11 = extractvalue [3 x double**] %"src'", 2
; CHECK-NEXT:   %"'ipc5" = bitcast double** %11 to i8*
; CHECK-NEXT:   %12 = insertvalue [3 x i8*] %10, i8* %"'ipc5", 2
; CHECK-NEXT:   %13 = bitcast double** %src to i8*
; CHECK-NEXT:   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %6, i8* align 1 %13, i64 %num, i1 false)
; CHECK-NEXT:   %14 = extractvalue [3 x i8*] %5, 0
; CHECK-NEXT:   %15 = extractvalue [3 x i8*] %12, 0
; CHECK-NEXT:   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %14, i8* align 1 %15, i64 %num, i1 false)
; CHECK-NEXT:   %16 = extractvalue [3 x i8*] %5, 1
; CHECK-NEXT:   %17 = extractvalue [3 x i8*] %12, 1
; CHECK-NEXT:   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %16, i8* align 1 %17, i64 %num, i1 false)
; CHECK-NEXT:   %18 = extractvalue [3 x i8*] %5, 2
; CHECK-NEXT:   %19 = extractvalue [3 x i8*] %12, 2
; CHECK-NEXT:   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %18, i8* align 1 %19, i64 %num, i1 false)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }