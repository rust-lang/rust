; RUN: %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=callee -o /dev/null | FileCheck %s


source_filename = "Awesome.bc"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx11.0.0"

%TSf = type <{ float }>
%TSi = type <{ i64 }>
%swift.type_descriptor = type opaque
%swift.type = type { i64 }
%swift.protocol_conformance_descriptor = type { i32, i32, i32, i32 }
%Ts26DefaultStringInterpolationV = type <{ %TSS }>
%TSS = type <{ %Ts11_StringGutsV }>
%Ts11_StringGutsV = type <{ %Ts13_StringObjectV }>
%Ts13_StringObjectV = type <{ %Ts6UInt64V, %swift.bridge* }>
%Ts6UInt64V = type <{ i64 }>
%swift.bridge = type opaque
%swift.refcounted = type { %swift.type*, i64 }
%swift.opaque = type opaque

@"$s7Awesome5valueSfvp" = hidden global %TSf zeroinitializer, align 4
@"$s7Awesome11derivativesSf_Sftvp" = hidden global <{ %TSf, %TSf }> zeroinitializer, align 4
@"$s7Awesome3sumSfvp" = hidden global %TSf zeroinitializer, align 4
@"$s7Awesome10iterationsSivp" = hidden local_unnamed_addr global %TSi zeroinitializer, align 8
@"$ss23_ContiguousArrayStorageCMn" = external global %swift.type_descriptor, align 4
@"got.$ss23_ContiguousArrayStorageCMn" = private unnamed_addr constant %swift.type_descriptor* @"$ss23_ContiguousArrayStorageCMn"
@"symbolic _____yypG s23_ContiguousArrayStorageC" = linkonce_odr hidden constant <{ i8, i32, [4 x i8], i8 }> <{ i8 2, i32 trunc (i64 sub (i64 ptrtoint (%swift.type_descriptor** @"got.$ss23_ContiguousArrayStorageCMn" to i64), i64 ptrtoint (i32* getelementptr inbounds (<{ i8, i32, [4 x i8], i8 }>, <{ i8, i32, [4 x i8], i8 }>* @"symbolic _____yypG s23_ContiguousArrayStorageC", i32 0, i32 1) to i64)) to i32), [4 x i8] c"yypG", i8 0 }>, section "__TEXT,__swift5_typeref, regular, no_dead_strip", align 2
@ptr = linkonce_odr hidden global { i32, i32 } { i32 ptrtoint (<{ i8, i32, [4 x i8], i8 }>* @"symbolic _____yypG s23_ContiguousArrayStorageC" to i32), i32 17 }

define void @callee() {
entry:
  %loadnotype = load i32, i32* getelementptr inbounds ({ i32, i32 }, { i32, i32 }* @ptr, i64 0, i32 1), align 4
  ret void
}

!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}
!7 = !{!"double", !5, i64 0}
!8 = !{!7, !7, i64 0}


; CHECK: callee - {} |
; CHECK-NEXT: entry
; CHECK-NEXT:   %loadnotype = load i32, i32* getelementptr inbounds ({ i32, i32 }, { i32, i32 }* @ptr, i64 0, i32 1), align 4: {}
; CHECK-NEXT:   ret void: {}
