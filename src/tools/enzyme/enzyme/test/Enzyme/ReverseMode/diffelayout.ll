; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -simplifycfg -instcombine -adce -S | FileCheck %s

%struct.S = type { i32, double, double }

define void @dup(%struct.S* nocapture readonly %from, %struct.S* nocapture %to) {
entry:
  %num_elems = getelementptr inbounds %struct.S, %struct.S* %to, i64 0, i32 0
  store i32 2, i32* %num_elems, align 8, !tbaa !2
  %data1 = getelementptr inbounds %struct.S, %struct.S* %from, i64 0, i32 1
  %0 = bitcast double* %data1 to i64*
  %1 = load i64, i64* %0, align 8, !tbaa !8
  %data11 = getelementptr inbounds %struct.S, %struct.S* %to, i64 0, i32 1
  %2 = bitcast double* %data11 to i64*
  store i64 %1, i64* %2, align 8, !tbaa !8
  %3 = load i64, i64* %0, align 8, !tbaa !8
  %data2 = getelementptr inbounds %struct.S, %struct.S* %to, i64 0, i32 2
  %4 = bitcast double* %data2 to i64*
  store i64 %3, i64* %4, align 8, !tbaa !9
  ret void
}

define void @caller(%struct.S* %a, %struct.S* %ap, %struct.S* %b, %struct.S* %bp) {
entry:
  tail call void @__enzyme_autodiff(i8* bitcast (void (%struct.S*, %struct.S*)* @dup to i8*), %struct.S* %a, %struct.S* %ap, %struct.S* %b, %struct.S* %bp)
  ret void
}

declare void @__enzyme_autodiff(i8*, %struct.S*, %struct.S*, %struct.S*, %struct.S*)

!2 = !{!3, !4, i64 0}
!3 = !{!"", !4, i64 0, !7, i64 8, !7, i64 16}
!4 = !{!"int", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C/C++ TBAA"}
!7 = !{!"double", !5, i64 0}
!8 = !{!3, !7, i64 8}
!9 = !{!3, !7, i64 16}

; CHECK: define internal void @diffedup(%struct.S* nocapture readonly %from, %struct.S* nocapture %"from'", %struct.S* nocapture %to, %struct.S* nocapture %"to'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[numelemsipge:.+]] = getelementptr inbounds %struct.S, %struct.S* %"to'", i64 0, i32 0
; CHECK-NEXT:   %num_elems = getelementptr inbounds %struct.S, %struct.S* %to, i64 0, i32 0
; CHECK-NEXT:   store i32 2, i32* %[[numelemsipge]], align 8
; CHECK-NEXT:   store i32 2, i32* %num_elems, align 8, !tbaa !0
; CHECK-NEXT:   %[[data1ipge:.+]] = getelementptr inbounds %struct.S, %struct.S* %"from'", i64 0, i32 1
; CHECK-NEXT:   %data1 = getelementptr inbounds %struct.S, %struct.S* %from, i64 0, i32 1
; CHECK-NEXT:   %0 = bitcast double* %data1 to i64*
; CHECK-NEXT:   %1 = load i64, i64* %0, align 8, !tbaa !6
; CHECK-NEXT:   %[[data11ipge:.+]] = getelementptr inbounds %struct.S, %struct.S* %"to'", i64 0, i32 1
; CHECK-NEXT:   %data11 = getelementptr inbounds %struct.S, %struct.S* %to, i64 0, i32 1
; CHECK-NEXT:   %[[ipc2:.+]] = bitcast double* %[[data11ipge]] to i64*
; CHECK-NEXT:   %2 = bitcast double* %data11 to i64*
; CHECK-NEXT:   store i64 %1, i64* %2, align 8, !tbaa !6
; CHECK-NEXT:   %3 = load i64, i64* %0, align 8, !tbaa !6
; CHECK-NEXT:   %[[data2ipge:.+]] = getelementptr inbounds %struct.S, %struct.S* %"to'", i64 0, i32 2
; CHECK-NEXT:   %data2 = getelementptr inbounds %struct.S, %struct.S* %to, i64 0, i32 2
; CHECK-NEXT:   %[[ipc:.+]] = bitcast double* %[[data2ipge]] to i64*
; CHECK-NEXT:   %4 = bitcast double* %data2 to i64*
; CHECK-NEXT:   store i64 %3, i64* %4, align 8, !tbaa !7
; CHECK-NEXT:   %5 = load double, double* %[[data2ipge]], align 8
; CHECK-NEXT:   store i64 0, i64* %[[ipc]], align 8
; CHECK-NEXT:   %6 = load double, double* %[[data1ipge]], align 8
; CHECK-NEXT:   %7 = fadd fast double %6, %5
; CHECK-NEXT:   store double %7, double* %[[data1ipge]], align 8
; CHECK-NEXT:   %8 = load double, double* %[[data11ipge]], align 8
; CHECK-NEXT:   store i64 0, i64* %[[ipc2]], align 8
; CHECK-NEXT:   %9 = load double, double* %[[data1ipge]], align 8
; CHECK-NEXT:   %10 = fadd fast double %9, %8
; CHECK-NEXT:   store double %10, double* %[[data1ipge]], align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
