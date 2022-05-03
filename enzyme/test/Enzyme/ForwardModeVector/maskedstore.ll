; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -simplifycfg -dce -instcombine -S | FileCheck %s

%struct.Gradients = type { double, double, double, double, double, double }

declare %struct.Gradients @__enzyme_fwddiff.f64(...) 

declare void @llvm.masked.store.v2f64.p0v2f64  (<2 x double>, <2 x double>*, i32, <2 x i1>)

; Function Attrs: nounwind uwtable
define dso_local void @loader(<2 x double>* %ptr, <2 x i1> %mask, <2 x double> %val) {
entry:
  call void @llvm.masked.store.v2f64.p0v2f64(<2 x double> %val, <2 x double>* %ptr, i32 16, <2 x i1> %mask)
  ret void
}


; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1) #1

; Function Attrs: nounwind uwtable
define %struct.Gradients @dloader(i8* %ptr, i8* %dptr1, i8* %dptr2, i8* %dptr3, <2 x i1> %mask, <2 x double> %other, <2 x double> %dother1, <2 x double> %dother2, <2 x double> %dother3) {
entry:
  %res = tail call %struct.Gradients (...) @__enzyme_fwddiff.f64(void (<2 x double>*, <2 x i1>, <2 x double>)* @loader, metadata !"enzyme_width", i64 3, i8* %ptr, i8* %dptr1, i8* %dptr2, i8* %dptr3, <2 x i1> %mask, <2 x double> %other, <2 x double> %dother1, <2 x double> %dother2, <2 x double> %dother3)
  ret %struct.Gradients %res
}


; CHECK: define internal void @fwddiffe3loader(<2 x double>* %ptr, [3 x <2 x double>*] %"ptr'", <2 x i1> %mask, <2 x double> %val, [3 x <2 x double>] %"val'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @llvm.masked.store.v2f64.p0v2f64(<2 x double> %val, <2 x double>* %ptr, i32 16, <2 x i1> %mask)
; CHECK-NEXT:   %0 = extractvalue [3 x <2 x double>*] %"ptr'", 0
; CHECK-NEXT:   %1 = extractvalue [3 x <2 x double>] %"val'", 0
; CHECK-NEXT:   call void @llvm.masked.store.v2f64.p0v2f64(<2 x double> %1, <2 x double>* %0, i32 16, <2 x i1> %mask)
; CHECK-NEXT:   %2 = extractvalue [3 x <2 x double>*] %"ptr'", 1
; CHECK-NEXT:   %3 = extractvalue [3 x <2 x double>] %"val'", 1
; CHECK-NEXT:   call void @llvm.masked.store.v2f64.p0v2f64(<2 x double> %3, <2 x double>* %2, i32 16, <2 x i1> %mask)
; CHECK-NEXT:   %4 = extractvalue [3 x <2 x double>*] %"ptr'", 2
; CHECK-NEXT:   %5 = extractvalue [3 x <2 x double>] %"val'", 2
; CHECK-NEXT:   call void @llvm.masked.store.v2f64.p0v2f64(<2 x double> %5, <2 x double>* %4, i32 16, <2 x i1> %mask)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
