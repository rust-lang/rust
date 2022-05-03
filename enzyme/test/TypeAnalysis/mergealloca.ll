; RUN: %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=caller -o /dev/null | FileCheck %s


define void @caller() {
entry:
  %v = alloca [3 x double], align 8
  %v0 = getelementptr inbounds [3 x double], [3 x double]* %v, i32 0, i32 0
  store double 0.000000, double* %v0, align 8, !tbaa !2
  %v1 = getelementptr inbounds [3 x double], [3 x double]* %v, i32 0, i32 1
  store double 0.000000, double* %v1, align 8, !tbaa !2
  %v2 = getelementptr inbounds [3 x double], [3 x double]* %v, i32 0, i32 2
  store double 0.000000, double* %v2, align 8, !tbaa !2
  ret void
}

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0, i64 8}
!3 = !{!4, i64 8, !"double"}
!4 = !{!5, i64 1, !"omnipotent char"}
!5 = !{!"Simple C++ TBAA"}


; CHECK: caller - {} |
; CHECK-NEXT: entry
; CHECK-NEXT:   %v = alloca [3 x double], align 8: {[-1]:Pointer, [-1,-1]:Float@double}
; CHECK-NEXT:   %v0 = getelementptr inbounds [3 x double], [3 x double]* %v, i32 0, i32 0: {[-1]:Pointer, [-1,-1]:Float@double}
; CHECK-NEXT:   store double 0.000000e+00, double* %v0, align 8, !tbaa !2: {}
; CHECK-NEXT:   %v1 = getelementptr inbounds [3 x double], [3 x double]* %v, i32 0, i32 1: {[-1]:Pointer, [-1,-1]:Float@double}
; CHECK-NEXT:   store double 0.000000e+00, double* %v1, align 8, !tbaa !2: {}
; CHECK-NEXT:   %v2 = getelementptr inbounds [3 x double], [3 x double]* %v, i32 0, i32 2: {[-1]:Pointer, [-1,-1]:Float@double}
; CHECK-NEXT:   store double 0.000000e+00, double* %v2, align 8, !tbaa !2: {}
; CHECK-NEXT:   ret void: {}

