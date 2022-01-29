; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s
source_filename = "/mnt/Data/git/Enzyme/enzyme/test/Integration/eigentensorfull.cpp"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }
%"struct.std::array.6" = type { [2 x i64] }

define i32 @caller(%"struct.std::array.6"* dereferenceable(16) %arr, %"struct.std::array.6"* dereferenceable(16) %darr) {
entry:
  %call86 = call double (i8*, ...) @__enzyme_autodiff(i8* bitcast (void (%"struct.std::array.6"*, i64)* @todiff to i8*), metadata !"enzyme_dup", %"struct.std::array.6"* dereferenceable(16) %arr, %"struct.std::array.6"* dereferenceable(16) %darr, i64 1)
  ret i32 0
}

declare dso_local double @__enzyme_autodiff(i8*, ...)

define void @todiff(%"struct.std::array.6"* dereferenceable(16) %arr, i64 %identity) {
entry:
  %arr.i = alloca %"struct.std::array.6", align 8
  %agg.tmp.sroa.0.0..sroa_idx = getelementptr inbounds %"struct.std::array.6", %"struct.std::array.6"* %arr, i64 0, i32 0, i64 0
  %agg.tmp.sroa.0.0.copyload = load i64, i64* %agg.tmp.sroa.0.0..sroa_idx, align 8, !tbaa !2
  %agg.tmp.sroa.2.0..sroa_idx1 = getelementptr inbounds %"struct.std::array.6", %"struct.std::array.6"* %arr, i64 0, i32 0, i64 1
  %agg.tmp.sroa.2.0.copyload = load i64, i64* %agg.tmp.sroa.2.0..sroa_idx1, align 8, !tbaa !2
  %0 = bitcast %"struct.std::array.6"* %arr.i to i8*
  %1 = getelementptr inbounds %"struct.std::array.6", %"struct.std::array.6"* %arr.i, i64 0, i32 0, i64 0
  store i64 %agg.tmp.sroa.0.0.copyload, i64* %1, align 8
  %2 = getelementptr inbounds %"struct.std::array.6", %"struct.std::array.6"* %arr.i, i64 0, i32 0, i64 1
  store i64 %agg.tmp.sroa.2.0.copyload, i64* %2, align 8
  ret void
}

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0, i64 16}
!3 = !{!4, i64 16, !"_ZTSSt5arrayIlLm2EE", !6, i64 0, i64 16}
!4 = !{!5, i64 1, !"omnipotent char"}
!5 = !{!"Simple C++ TBAA"}
!6 = !{!4, i64 8, !"long"}

; CHECK: define internal void @diffetodiff(%"struct.std::array.6"* dereferenceable(16) %arr, %"struct.std::array.6"* %"arr'", i64 %identity)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %arr.i = alloca %"struct.std::array.6", align 8
; CHECK-NEXT:   %agg.tmp.sroa.0.0..sroa_idx = getelementptr inbounds %"struct.std::array.6", %"struct.std::array.6"* %arr, i64 0, i32 0, i64 0
; CHECK-NEXT:   %agg.tmp.sroa.0.0.copyload = load i64, i64* %agg.tmp.sroa.0.0..sroa_idx, align 8, !tbaa !2
; CHECK-NEXT:   %agg.tmp.sroa.2.0..sroa_idx1 = getelementptr inbounds %"struct.std::array.6", %"struct.std::array.6"* %arr, i64 0, i32 0, i64 1
; CHECK-NEXT:   %agg.tmp.sroa.2.0.copyload = load i64, i64* %agg.tmp.sroa.2.0..sroa_idx1, align 8, !tbaa !2
; TODO:   %[[ipge1:.+]] = getelementptr inbounds %"struct.std::array.6", %"struct.std::array.6"* %"arr.i'ipa", i64 0, i32 0, i64 0
; CHECK-NEXT:   %[[realgep:.+]] = getelementptr inbounds %"struct.std::array.6", %"struct.std::array.6"* %arr.i, i64 0, i32 0, i64 0
; TODO:   store i64 %agg.tmp.sroa.0.0.copyload, i64* %[[ipge1]], align 8
; CHECK-NEXT:   store i64 %agg.tmp.sroa.0.0.copyload, i64* %[[realgep]], align 8
; TODO:   %[[ipge:.+]] = getelementptr inbounds %"struct.std::array.6", %"struct.std::array.6"* %"arr.i'ipa", i64 0, i32 0, i64 1
; CHECK-NEXT:   %[[origep:.+]] = getelementptr inbounds %"struct.std::array.6", %"struct.std::array.6"* %arr.i, i64 0, i32 0, i64 1
; TODO:   store i64 %agg.tmp.sroa.2.0.copyload, i64* %[[ipge]], align 8
; CHECK-NEXT:   store i64 %agg.tmp.sroa.2.0.copyload, i64* %[[origep]], align 8
; CHECK-NEXT:   br label %invertentry

; CHECK: invertentry:                                      ; preds = %entry
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
