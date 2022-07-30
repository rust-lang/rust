; RUN: %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=f -enzyme-strict-aliasing=0 -o /dev/null | FileCheck %s

source_filename = "<source>"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%class.Testing = type { %struct.Header, %struct.Header }
%struct.Header = type { %struct.Base, i32 }
%struct.Base = type { %struct.Base*, %struct.Base* }

define dso_local void @f(%class.Testing* %arg) {
bb:
  %i = getelementptr inbounds %class.Testing, %class.Testing* %arg, i64 0, i32 0, i32 0
  %i1 = getelementptr inbounds %class.Testing, %class.Testing* %arg, i64 0, i32 0, i32 0, i32 0
  %i13 = getelementptr inbounds %class.Testing, %class.Testing* %arg, i64 0, i32 1, i32 0
  %i14 = getelementptr inbounds %class.Testing, %class.Testing* %arg, i64 0, i32 1, i32 0, i32 0
  br label %bb2

bb2:                                              ; preds = %bb2, %bb
  %i3 = phi %struct.Base** [ %i1, %bb ], [ %i7, %bb2 ]
  %i4 = phi %struct.Base* [ %i, %bb ], [ %i5, %bb2 ]
  %i5 = load %struct.Base*, %struct.Base** %i3, align 8, !tbaa !3
  %i6 = icmp eq %struct.Base* %i5, null
  %i7 = getelementptr inbounds %struct.Base, %struct.Base* %i5, i64 0, i32 1
  br i1 %i6, label %bb8, label %bb2, !llvm.loop !7

bb8:                                              ; preds = %bb2
  %i9 = getelementptr inbounds %struct.Base, %struct.Base* %i4, i64 1, i32 1
  %i10 = bitcast %struct.Base** %i9 to double*
  %i11 = load double, double* %i10, align 8, !tbaa !9
  br label %bb15

bb15:                                             ; preds = %bb15, %bb8
  %i16 = phi %struct.Base** [ %i14, %bb8 ], [ %i20, %bb15 ]
  %i17 = phi %struct.Base* [ %i13, %bb8 ], [ %i18, %bb15 ]
  %i18 = load %struct.Base*, %struct.Base** %i16, align 8, !tbaa !3
  %i19 = icmp eq %struct.Base* %i18, null
  %i20 = getelementptr inbounds %struct.Base, %struct.Base* %i18, i64 0, i32 1
  br i1 %i19, label %bb21, label %bb15, !llvm.loop !7

bb21:                                             ; preds = %bb15
  %i22 = getelementptr inbounds %struct.Base, %struct.Base* %i17, i64 1, i32 1
  %i23 = bitcast %struct.Base** %i22 to double*
  %i24 = load double, double* %i23, align 8, !tbaa !9
  tail call void @_Z5printdd(double %i11, double %i24)
  ret void
}

declare void @_Z5printdd(double, double)

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 7, !"Dwarf Version", i32 4}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{!"clang version 12.0.1 (https://github.com/llvm/llvm-project.git fed41342a82f5a3a9201819a82bf7a48313e296b)"}
!3 = !{!4, !4, i64 0}
!4 = !{!"any pointer", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}
!7 = distinct !{!7, !8}
!8 = !{!"llvm.loop.mustprogress"}
!9 = !{!10, !10, i64 0}
!10 = !{!"double", !5, i64 0}

; CHECK: %class.Testing* %arg: {[-1]:Pointer}
; CHECK-NEXT: bb
; CHECK-NEXT:   %i = getelementptr inbounds %class.Testing, %class.Testing* %arg, i64 0, i32 0, i32 0: {[-1]:Pointer}
; CHECK-NEXT:   %i1 = getelementptr inbounds %class.Testing, %class.Testing* %arg, i64 0, i32 0, i32 0, i32 0: {[-1]:Pointer}
; CHECK-NEXT:   %i13 = getelementptr inbounds %class.Testing, %class.Testing* %arg, i64 0, i32 1, i32 0: {[-1]:Pointer}
; CHECK-NEXT:   %i14 = getelementptr inbounds %class.Testing, %class.Testing* %arg, i64 0, i32 1, i32 0, i32 0: {[-1]:Pointer}
; CHECK-NEXT:   br label %bb2: {}
; CHECK-NEXT: bb2
; CHECK-NEXT:   %i3 = phi %struct.Base** [ %i1, %bb ], [ %i7, %bb2 ]: {[-1]:Pointer, [-1,0]:Pointer}
; CHECK-NEXT:   %i4 = phi %struct.Base* [ %i, %bb ], [ %i5, %bb2 ]: {[-1]:Pointer, [-1,24]:Float@double}
; CHECK-NEXT:   %i5 = load %struct.Base*, %struct.Base** %i3, align 8, !tbaa !3: {[-1]:Pointer}
; CHECK-NEXT:   %i6 = icmp eq %struct.Base* %i5, null: {[-1]:Integer}
; CHECK-NEXT:   %i7 = getelementptr inbounds %struct.Base, %struct.Base* %i5, i64 0, i32 1: {[-1]:Pointer}
; CHECK-NEXT:   br i1 %i6, label %bb8, label %bb2, !llvm.loop !7: {}
; CHECK-NEXT: bb8
; CHECK-NEXT:   %i9 = getelementptr inbounds %struct.Base, %struct.Base* %i4, i64 1, i32 1: {[-1]:Pointer, [-1,0]:Float@double}
; CHECK-NEXT:   %i10 = bitcast %struct.Base** %i9 to double*: {[-1]:Pointer, [-1,0]:Float@double}
; CHECK-NEXT:   %i11 = load double, double* %i10, align 8, !tbaa !9: {[-1]:Float@double}
; CHECK-NEXT:   br label %bb15: {}
; CHECK-NEXT: bb15
; CHECK-NEXT:   %i16 = phi %struct.Base** [ %i14, %bb8 ], [ %i20, %bb15 ]: {[-1]:Pointer, [-1,0]:Pointer}
; CHECK-NEXT:   %i17 = phi %struct.Base* [ %i13, %bb8 ], [ %i18, %bb15 ]: {[-1]:Pointer, [-1,24]:Float@double}
; CHECK-NEXT:   %i18 = load %struct.Base*, %struct.Base** %i16, align 8, !tbaa !3: {[-1]:Pointer}
; CHECK-NEXT:   %i19 = icmp eq %struct.Base* %i18, null: {[-1]:Integer}
; CHECK-NEXT:   %i20 = getelementptr inbounds %struct.Base, %struct.Base* %i18, i64 0, i32 1: {[-1]:Pointer}
; CHECK-NEXT:   br i1 %i19, label %bb21, label %bb15, !llvm.loop !7: {}
; CHECK-NEXT: bb21
; CHECK-NEXT:   %i22 = getelementptr inbounds %struct.Base, %struct.Base* %i17, i64 1, i32 1: {[-1]:Pointer, [-1,0]:Float@double}
; CHECK-NEXT:   %i23 = bitcast %struct.Base** %i22 to double*: {[-1]:Pointer, [-1,0]:Float@double}
; CHECK-NEXT:   %i24 = load double, double* %i23, align 8, !tbaa !9: {[-1]:Float@double}
; CHECK-NEXT:   tail call void @_Z5printdd(double %i11, double %i24): {}
; CHECK-NEXT:   ret void: {}
