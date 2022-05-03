; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -simplifycfg -dce -instcombine -S | FileCheck %s

%struct.Gradients = type { double, double, double, double, double, double }

declare %struct.Gradients @__enzyme_fwddiff.f64(...) 

declare <2 x double>  @llvm.masked.load.v2f64.p0v2f64  (<2 x double>*, i32, <2 x i1>, <2 x double>)

; Function Attrs: nounwind uwtable
define dso_local <2 x double> @loader(<2 x double>* %ptr, <2 x i1> %mask, <2 x double> %other) {
entry:
  %res = call <2 x double> @llvm.masked.load.v2f64.p0v2f64(<2 x double>* %ptr, i32 16, <2 x i1> %mask, <2 x double> %other)
  ret <2 x double> %res
}


; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1) #1

; Function Attrs: nounwind uwtable
define %struct.Gradients @dloader(i8* %ptr, i8* %dptr1, i8* %dptr2, i8* %dptr3, <2 x i1> %mask, <2 x double> %other,  <2 x double> %dother1, <2 x double> %dother2, <2 x double> %dother3) {
entry:
  %res = tail call %struct.Gradients (...) @__enzyme_fwddiff.f64(<2 x double> (<2 x double>*, <2 x i1>, <2 x double>)* @loader, metadata !"enzyme_width", i64 3, i8* %ptr, i8* %dptr1, i8* %dptr2, i8* %dptr3, <2 x i1> %mask, <2 x double> %other, <2 x double> %dother1, <2 x double> %dother2, <2 x double> %dother3)
  ret %struct.Gradients %res
}


; CHECK: define internal [3 x <2 x double>] @fwddiffe3loader(<2 x double>* %ptr, [3 x <2 x double>*] %"ptr'", <2 x i1> %mask, <2 x double> %other, [3 x <2 x double>] %"other'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = extractvalue [3 x <2 x double>*] %"ptr'", 0
; CHECK-NEXT:   %1 = extractvalue [3 x <2 x double>] %"other'", 0
; CHECK-NEXT:   %2 = call fast <2 x double> @llvm.masked.load.v2f64.p0v2f64(<2 x double>* %0, i32 16, <2 x i1> %mask, <2 x double> %1)
; CHECK-NEXT:   %3 = insertvalue [3 x <2 x double>] undef, <2 x double> %2, 0
; CHECK-NEXT:   %4 = extractvalue [3 x <2 x double>*] %"ptr'", 1
; CHECK-NEXT:   %5 = extractvalue [3 x <2 x double>] %"other'", 1
; CHECK-NEXT:   %6 = call fast <2 x double> @llvm.masked.load.v2f64.p0v2f64(<2 x double>* %4, i32 16, <2 x i1> %mask, <2 x double> %5)
; CHECK-NEXT:   %7 = insertvalue [3 x <2 x double>] %3, <2 x double> %6, 1
; CHECK-NEXT:   %8 = extractvalue [3 x <2 x double>*] %"ptr'", 2
; CHECK-NEXT:   %9 = extractvalue [3 x <2 x double>] %"other'", 2
; CHECK-NEXT:   %10 = call fast <2 x double> @llvm.masked.load.v2f64.p0v2f64(<2 x double>* %8, i32 16, <2 x i1> %mask, <2 x double> %9)
; CHECK-NEXT:   %11 = insertvalue [3 x <2 x double>] %7, <2 x double> %10, 2
; CHECK-NEXT:   ret [3 x <2 x double>] %11
; CHECK-NEXT: }