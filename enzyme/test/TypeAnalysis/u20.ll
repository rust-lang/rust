; RUN: %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=f -o /dev/null | FileCheck %s

declare void @__enzyme_autodiff(...)

define void @f(i64* %inp) {
  %x = load i64, i64* %inp, align 8, !tbaa !2
  %ie = insertelement <2 x i64> undef, i64 %x, i32 0
  %bc = bitcast <2 x i64> %ie to <2 x double>
  %sv = shufflevector <2 x double> %bc, <2 x double> undef, <2 x i32> zeroinitializer
  ret void
}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0, i64 8}
!3 = !{!4, i64 8, !"double"}
!4 = !{!5, i64 1, !"omnipotent char"}
!5 = !{!"Simple C++ TBAA"}

; CHECK: f - {} |{[-1]:Pointer}:{} 
; CHECK: i64* %inp: {[-1]:Pointer, [-1,0]:Float@double}

; CHECK:   %x = load i64, i64* %inp, align 8, !tbaa !0: {[-1]:Float@double}
; CHECK:   %ie = insertelement <2 x i64> undef, i64 %x, i32 0: {[0]:Float@double, [8]:Anything, [9]:Anything, [10]:Anything, [11]:Anything, [12]:Anything, [13]:Anything, [14]:Anything, [15]:Anything}
; CHECK:   %bc = bitcast <2 x i64> %ie to <2 x double>: {[0]:Float@double, [8]:Anything, [9]:Anything, [10]:Anything, [11]:Anything, [12]:Anything, [13]:Anything, [14]:Anything, [15]:Anything}
; CHECK:   %sv = shufflevector <2 x double> %bc, <2 x double> undef, <2 x i32> zeroinitializer: {[-1]:Float@double}
; CHECK:   ret void: {}
