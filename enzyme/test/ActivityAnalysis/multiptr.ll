; RUN: %opt < %s %loadEnzyme -print-activity-analysis -activity-analysis-func=f -o /dev/null | FileCheck %s

define internal double @f(double %wide.load.i.i) {
entry:
  %call.i.i4.i.i.i.i.i.i = alloca i8, i64 10, align 8
  %t2 = ptrtoint i8* %call.i.i4.i.i.i.i.i.i to i64
  %srcEvaluator.i.sroa.0.0.vec.insert = insertelement <2 x i64> zeroinitializer, i64 %t2, i32 0
  %srcEvaluator.i.sroa.0.8.vec.insert = insertelement <2 x i64> %srcEvaluator.i.sroa.0.0.vec.insert, i64 5, i32 1
  %tmp23 = bitcast i8* %call.i.i4.i.i.i.i.i.i to double*
  store double %wide.load.i.i, double* %tmp23, align 8
  %a5 = bitcast <2 x i64> %srcEvaluator.i.sroa.0.8.vec.insert to i128
  %a6 = trunc i128 %a5 to i64
  %a7 = inttoptr i64 %a6 to double*
  %atmp = load double, double* %a7, align 8
  ret double %atmp
}

; CHECK: double %wide.load.i.i: icv:0
; CHECK-NEXT: entry
; CHECK-NEXT:   %call.i.i4.i.i.i.i.i.i = alloca i8, i64 10, align 8: icv:0 ici:1
; CHECK-NEXT:   %t2 = ptrtoint i8* %call.i.i4.i.i.i.i.i.i to i64: icv:0 ici:1
; CHECK-NEXT:   %srcEvaluator.i.sroa.0.0.vec.insert = insertelement <2 x i64> zeroinitializer, i64 %t2, i32 0: icv:0 ici:1
; CHECK-NEXT:   %srcEvaluator.i.sroa.0.8.vec.insert = insertelement <2 x i64> %srcEvaluator.i.sroa.0.0.vec.insert, i64 5, i32 1: icv:0 ici:1
; CHECK-NEXT:   %tmp23 = bitcast i8* %call.i.i4.i.i.i.i.i.i to double*: icv:0 ici:1
; CHECK-NEXT:   store double %wide.load.i.i, double* %tmp23, align 8: icv:1 ici:0
; CHECK-NEXT:   %a5 = bitcast <2 x i64> %srcEvaluator.i.sroa.0.8.vec.insert to i128: icv:0 ici:1
; CHECK-NEXT:   %a6 = trunc i128 %a5 to i64: icv:0 ici:1
; CHECK-NEXT:   %a7 = inttoptr i64 %a6 to double*: icv:0 ici:1
; CHECK-NEXT:   %atmp = load double, double* %a7, align 8: icv:0 ici:0
; CHECK-NEXT:   ret double %atmp: icv:1 ici:1
