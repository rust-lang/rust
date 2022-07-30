; RUN: %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=f -enzyme-strict-aliasing=0 -o /dev/null | FileCheck %s

source_filename = "<source>"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%class.Testing = type { %struct.Header, %struct.Header }
%struct.Header = type { %struct.Base, i32 }
%struct.Base = type { %struct.Base* }

define dso_local void @f(%class.Testing* nocapture nonnull readonly %arg) {
bb:
  %i = getelementptr inbounds %class.Testing, %class.Testing* %arg, i64 0, i32 0, i32 0
  %i1 = getelementptr inbounds %class.Testing, %class.Testing* %arg, i64 0, i32 0, i32 0, i32 0
  %i2 = load %struct.Base*, %struct.Base** %i1, align 8, !tbaa !3
  %i3 = icmp eq %struct.Base* %i2, null
  %i4 = select i1 %i3, %struct.Base* %i, %struct.Base* %i2
  %i5 = getelementptr inbounds %struct.Base, %struct.Base* %i4, i64 2
  %i6 = bitcast %struct.Base* %i5 to double*
  %i7 = load double, double* %i6, align 8, !tbaa !10
  %i8 = getelementptr inbounds %class.Testing, %class.Testing* %arg, i64 0, i32 1
  %i9 = getelementptr inbounds %struct.Header, %struct.Header* %i8, i64 0, i32 0
  %i10 = getelementptr inbounds %struct.Header, %struct.Header* %i8, i64 0, i32 0, i32 0
  %i11 = load %struct.Base*, %struct.Base** %i10, align 8, !tbaa !3
  %i12 = icmp eq %struct.Base* %i11, null
  %i13 = select i1 %i12, %struct.Base* %i9, %struct.Base* %i11
  %i14 = getelementptr inbounds %struct.Base, %struct.Base* %i13, i64 2
  %i15 = bitcast %struct.Base* %i14 to double*
  %i16 = load double, double* %i15, align 8, !tbaa !10
  tail call void (...) @_Z6printfPKcz(double %i7, double %i16)
  ret void
}

declare void @_Z6printfPKcz(...)

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 7, !"Dwarf Version", i32 4}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{!"clang version 12.0.1 (https://github.com/llvm/llvm-project.git fed41342a82f5a3a9201819a82bf7a48313e296b)"}
!3 = !{!4, !6, i64 0}
!4 = !{!"_ZTS6Header", !5, i64 0, !9, i64 8}
!5 = !{!"_ZTS4Base", !6, i64 0}
!6 = !{!"any pointer", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C++ TBAA"}
!9 = !{!"int", !7, i64 0}
!10 = !{!11, !11, i64 0}
!11 = !{!"double", !7, i64 0}

; CHECK: f - {} |{[-1]:Pointer}:{} 
; CHECK-NEXT: %class.Testing* %arg: {[-1]:Pointer, [-1,0]:Pointer, [-1,16]:Pointer}
; CHECK-NEXT: bb
; CHECK-NEXT:   %i = getelementptr inbounds %class.Testing, %class.Testing* %arg, i64 0, i32 0, i32 0: {[-1]:Pointer, [-1,0]:Pointer}
; CHECK-NEXT:   %i1 = getelementptr inbounds %class.Testing, %class.Testing* %arg, i64 0, i32 0, i32 0, i32 0: {[-1]:Pointer, [-1,0]:Pointer}
; CHECK-NEXT:   %i2 = load %struct.Base*, %struct.Base** %i1, align 8, !tbaa !3: {[-1]:Pointer}
; CHECK-NEXT:   %i3 = icmp eq %struct.Base* %i2, null: {[-1]:Integer}
; CHECK-NEXT:   %i4 = select i1 %i3, %struct.Base* %i, %struct.Base* %i2: {[-1]:Pointer, [-1,16]:Float@double}
; CHECK-NEXT:   %i5 = getelementptr inbounds %struct.Base, %struct.Base* %i4, i64 2: {[-1]:Pointer, [-1,0]:Float@double}
; CHECK-NEXT:   %i6 = bitcast %struct.Base* %i5 to double*: {[-1]:Pointer, [-1,0]:Float@double}
; CHECK-NEXT:   %i7 = load double, double* %i6, align 8, !tbaa !10: {[-1]:Float@double}
; CHECK-NEXT:   %i8 = getelementptr inbounds %class.Testing, %class.Testing* %arg, i64 0, i32 1: {[-1]:Pointer, [-1,0]:Pointer}
; CHECK-NEXT:   %i9 = getelementptr inbounds %struct.Header, %struct.Header* %i8, i64 0, i32 0: {[-1]:Pointer, [-1,0]:Pointer}
; CHECK-NEXT:   %i10 = getelementptr inbounds %struct.Header, %struct.Header* %i8, i64 0, i32 0, i32 0: {[-1]:Pointer, [-1,0]:Pointer}
; CHECK-NEXT:   %i11 = load %struct.Base*, %struct.Base** %i10, align 8, !tbaa !3: {[-1]:Pointer}
; CHECK-NEXT:   %i12 = icmp eq %struct.Base* %i11, null: {[-1]:Integer}
; CHECK-NEXT:   %i13 = select i1 %i12, %struct.Base* %i9, %struct.Base* %i11: {[-1]:Pointer, [-1,16]:Float@double}
; CHECK-NEXT:   %i14 = getelementptr inbounds %struct.Base, %struct.Base* %i13, i64 2: {[-1]:Pointer, [-1,0]:Float@double}
; CHECK-NEXT:   %i15 = bitcast %struct.Base* %i14 to double*: {[-1]:Pointer, [-1,0]:Float@double}
; CHECK-NEXT:   %i16 = load double, double* %i15, align 8, !tbaa !10: {[-1]:Float@double}
; CHECK-NEXT:   tail call void (...) @_Z6printfPKcz(double %i7, double %i16): {}
; CHECK-NEXT:   ret void: {}
