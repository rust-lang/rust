; RUN: %opt < %s %loadEnzyme -print-activity-analysis -activity-analysis-func=_Z2fnv -o /dev/null | FileCheck %s

define void @_Z2fnv() {
entry:
  %ref.tmp = alloca i8*, align 8
  %i35 = load i8*, i8** %ref.tmp, align 8
  
  %alloc2 = alloca i64*, align 8
  %i40 = bitcast i64** %alloc2 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %i40, i8* %i35, i64 6, i1 false)
  
  %i16 = load i64*, i64** %alloc2, align 8
  %vbase = load i64, i64* %i16, align 8
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8*, i8*, i64, i1)

; CHECK: entry
; CHECK-NEXT:   %ref.tmp = alloca i8*, align 8: icv:1 ici:1
; CHECK-NEXT:   %i35 = load i8*, i8** %ref.tmp, align 8: icv:1 ici:1
; CHECK-NEXT:   %alloc2 = alloca i64*, align 8: icv:1 ici:1
; CHECK-NEXT:   %i40 = bitcast i64** %alloc2 to i8*: icv:1 ici:1
; CHECK-NEXT:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* %i40, i8* %i35, i64 6, i1 false): icv:1 ici:1
; CHECK-NEXT:   %i16 = load i64*, i64** %alloc2, align 8: icv:1 ici:1
; CHECK-NEXT:   %vbase = load i64, i64* %i16, align 8: icv:1 ici:1
; CHECK-NEXT:   ret void: icv:1 ici:1
