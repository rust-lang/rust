; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -simplifycfg -instcombine -adce -S | FileCheck %s

%struct.S = type { i32, double, double }

; Function Attrs: noinline norecurse nounwind uwtable
define dso_local void @dup(%struct.S* nocapture readonly %from, %struct.S* nocapture %to) {
entry:
  %data1 = getelementptr inbounds %struct.S, %struct.S* %from, i64 0, i32 1
  %data11 = getelementptr inbounds %struct.S, %struct.S* %to, i64 0, i32 1
  %0 = bitcast double* %data1 to <2 x i64>*
  %1 = load <2 x i64>, <2 x i64>* %0, align 8, !tbaa !2
  %2 = bitcast double* %data11 to <2 x i64>*
  store <2 x i64> %1, <2 x i64>* %2, align 8, !tbaa !2
  ret void
}

; Function Attrs: nounwind uwtable
define void @caller(%struct.S* %a, %struct.S* %ap, %struct.S* %b, %struct.S* %bp) {
entry:
  tail call void @__enzyme_autodiff(i8* bitcast (void (%struct.S*, %struct.S*)* @dup to i8*), %struct.S* %a, %struct.S* %ap, %struct.S* %b, %struct.S* %bp)
  ret void
}

declare void @__enzyme_autodiff(i8*, %struct.S*, %struct.S*, %struct.S*, %struct.S*)

!2 = !{!3, !3, i64 0}
!3 = !{!"double", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}

; CHECK: define internal void @diffedup(%struct.S* nocapture readonly %from, %struct.S* nocapture %"from'", %struct.S* nocapture %to, %struct.S* nocapture %"to'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[data1ipge:.+]] = getelementptr inbounds %struct.S, %struct.S* %"from'", i64 0, i32 1
; CHECK-NEXT:   %data1 = getelementptr inbounds %struct.S, %struct.S* %from, i64 0, i32 1
; CHECK-NEXT:   %[[data11ipge:.+]] = getelementptr inbounds %struct.S, %struct.S* %"to'", i64 0, i32 1
; CHECK-NEXT:   %data11 = getelementptr inbounds %struct.S, %struct.S* %to, i64 0, i32 1
; CHECK-NEXT:   %0 = bitcast double* %data1 to <2 x i64>*
; CHECK-NEXT:   %1 = load <2 x i64>, <2 x i64>* %0, align 8, !tbaa !0
; CHECK-NEXT:   %[[ipc1:.+]] = bitcast double* %[[data11ipge]] to <2 x i64>*
; CHECK-NEXT:   %2 = bitcast double* %data11 to <2 x i64>*
; CHECK-NEXT:   store <2 x i64> %1, <2 x i64>* %2, align 8, !tbaa !0
; CHECK-NEXT:   %3 = bitcast double* %[[data11ipge]] to <2 x double>*
; CHECK-NEXT:   %4 = load <2 x double>, <2 x double>* %3, align 8
; CHECK-NEXT:   store <2 x i64> zeroinitializer, <2 x i64>* %[[ipc1]], align 8
; CHECK-NEXT:   %5 = bitcast double* %[[data1ipge]] to <2 x double>*
; CHECK-NEXT:   %6 = load <2 x double>, <2 x double>* %5, align 8
; CHECK-NEXT:   %7 = fadd fast <2 x double> %6, %4
; CHECK-NEXT:   store <2 x double> %7, <2 x double>* %5, align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
