; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -instsimplify -simplifycfg -S | FileCheck %s

declare void @__enzyme_autodiff(...)

define <2 x double> @f(i64* %inp) {
  %x = load i64, i64* %inp, align 8, !tbaa !2
  %ie = insertelement <2 x i64> undef, i64 %x, i32 0
  %bc = bitcast <2 x i64> %ie to <2 x double>
  %sv = shufflevector <2 x double> %bc, <2 x double> undef, <2 x i32> zeroinitializer
  ret <2 x double> %sv
}

define double @caller() local_unnamed_addr #0 {
entry:
  %kernel = alloca i64, align 8
  %kernelp = alloca i64, align 8
  call void (...) @__enzyme_autodiff(<2 x double> (i64*)* nonnull @f, metadata !"enzyme_dup", i64* nonnull %kernel, i64* nonnull %kernelp)
  ret double 0.000000e+00
}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0, i64 8}
!3 = !{!4, i64 8, !"double"}
!4 = !{!5, i64 1, !"omnipotent char"}
!5 = !{!"Simple C++ TBAA"}

; CHECK: define internal void @diffef(i64* %inp, i64* %"inp'", <2 x double> %differeturn)
; CHECK-NEXT: invert:
; CHECK-NEXT:   %0 = extractelement <2 x double> %differeturn, i64 0
; CHECK-NEXT:   %"bc'de.0.vec.insert" = insertelement <2 x double> zeroinitializer, double %0, i32 0
; CHECK-NEXT:   %1 = extractelement <2 x double> %differeturn, i64 1
; CHECK-NEXT:   %2 = fadd fast double %0, %1
; CHECK-NEXT:   %"bc'de.0.vec.insert6" = insertelement <2 x double> %"bc'de.0.vec.insert", double %2, i32 0
; CHECK-NEXT:   %3 = bitcast <2 x double> %"bc'de.0.vec.insert6" to <2 x i64>
; CHECK-NEXT:   %4 = extractelement <2 x i64> %3, i32 0
; CHECK-NEXT:   %5 = bitcast i64* %"inp'" to double*
; CHECK-DAG:   %[[b6:.+]] = bitcast i64 %4 to double
; CHECK-DAG:   %[[b7:.+]] = load double, double* %5, align 8
; CHECK-NEXT:   %8 = fadd fast double %[[b7]], %[[b6]]
; CHECK-NEXT:   store double %8, double* %5, align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
