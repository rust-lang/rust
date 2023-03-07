; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

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
define <2 x double> @dloader(i8* %ptr, i8* %dptr, <2 x i1> %mask, <2 x double> %other, <2 x double> %dother) {
entry:
  %res = tail call <2 x double> (...) @__enzyme_fwddiff.f64(void (<2 x double>*, <2 x i1>, <2 x double>)* @loader, i8* %ptr, i8* %dptr, <2 x i1> %mask, <2 x double> %other, <2 x double> %dother)
  ret <2 x double> %res
}

declare <2 x double> @__enzyme_fwddiff.f64(...) 

; CHECK: define internal void @fwddiffeloader(<2 x double>* %ptr, <2 x double>* %"ptr'", <2 x i1> %mask, <2 x double> %val, <2 x double> %"val'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @llvm.masked.store.v2f64.p0v2f64(<2 x double> %val, <2 x double>* %ptr, i32 16, <2 x i1> %mask)
; CHECK-NEXT:   call void @llvm.masked.store.v2f64.p0v2f64(<2 x double> %"val'", <2 x double>* %"ptr'", i32 16, <2 x i1> %mask)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
