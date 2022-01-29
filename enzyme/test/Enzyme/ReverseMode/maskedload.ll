; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -simplifycfg -dce -instcombine -S | FileCheck %s

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
define <2 x double> @dloader(i8* %ptr, i8* %dptr, <2 x i1> %mask, <2 x double> %other) {
entry:
  %res = tail call <2 x double> (...) @__enzyme_autodiff.f64(<2 x double> (<2 x double>*, <2 x i1>, <2 x double>)* @loader, i8* %ptr, i8* %dptr, <2 x i1> %mask, <2 x double> %other)
  ret <2 x double> %res
}

declare <2 x double> @__enzyme_autodiff.f64(...) 

; CHECK: define internal { <2 x double> } @diffeloader(<2 x double>* %ptr, <2 x double>* %"ptr'", <2 x i1> %mask, <2 x double> %other, <2 x double> %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"other'de" = alloca <2 x double>, align 16
; CHECK-NEXT:   store <2 x double> zeroinitializer, <2 x double>* %"other'de", align 16
; CHECK-NEXT:   %0 = call fast <2 x double> @llvm.masked.load.v2f64.p0v2f64(<2 x double>* %"ptr'", i32 16, <2 x i1> %mask, <2 x double> zeroinitializer)
; CHECK-NEXT:   %1 = fadd fast <2 x double> %0, %differeturn
; CHECK-NEXT:   call void @llvm.masked.store.v2f64.p0v2f64(<2 x double> %1, <2 x double>* %"ptr'", i32 16, <2 x i1> %mask)
; CHECK-NEXT:   %2 = xor <2 x i1> %mask, <i1 true, i1 true>
; CHECK-NEXT:   call void @llvm.masked.store.v2f64.p0v2f64(<2 x double> %differeturn, <2 x double>* %"other'de", i32 16, <2 x i1> %2)
; CHECK-NEXT:   %3 = load <2 x double>, <2 x double>* %"other'de", align 16
; CHECK-NEXT:   %4 = insertvalue { <2 x double> } undef, <2 x double> %3, 0
; CHECK-NEXT:   ret { <2 x double> } %4
; CHECK-NEXT: }
