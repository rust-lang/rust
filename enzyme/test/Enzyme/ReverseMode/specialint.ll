; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -simplifycfg -instcombine -adce -S | FileCheck %s

%Type = type { float, i64 }

declare dso_local double @__enzyme_autodiff(i8*, ...)

; Function Attrs: alwaysinline norecurse nounwind uwtable
define double @caller(%Type* %K, %Type* %Kp) local_unnamed_addr #0 {
entry:
  %call86 = call double (i8*, ...) @__enzyme_autodiff(i8* bitcast (void (%Type*)* @matvec to i8*), metadata !"enzyme_dup", %Type* noalias %K, %Type* noalias %Kp) #4
  ret double %call86
}

define internal void @matvec(%Type* %evaluator.i.i) {
entry:
  %dims = getelementptr inbounds %Type, %Type* %evaluator.i.i, i64 0, i32 1
  %call = call i64 @total(i64* %dims) #4
  %flt = uitofp i64 %call to float
  %data = getelementptr inbounds %Type, %Type* %evaluator.i.i, i64 0, i32 0
  store float %flt, float* %data, align 4
  ret void
}

; Function Attrs: readnone
define i64 @meta(i64 %inp) #3 {
entry:
  %arr = alloca i64
  store i64 %inp, i64* %arr
  %call.i = call i64* @sub(i64* %arr)
  %a1 = load i64, i64* %call.i, !tbaa !2
  ret i64 %a1
}

define i64* @sub(i64* %a) {
entry:
  ret i64* %a
}

define i64 @total(i64* %this) {
entry:
  %loaded = load i64, i64* %this
  %mcall = tail call i64 @meta(i64 %loaded)
  ret i64 %mcall
}

attributes #3 = { readnone }

!2 = !{!3, !3, i64 0, i64 8}
!3 = !{!4, i64 8, !"long"}
!4 = !{!5, i64 1, !"omnipotent char"}
!5 = !{!"Simple C++ TBAA"}

; CHECK: define internal void @diffematvec(%Type* %evaluator.i.i, %Type* %"evaluator.i.i'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[dimsipge:.+]] = getelementptr inbounds %Type, %Type* %"evaluator.i.i'", i64 0, i32 1
; CHECK-NEXT:   %dims = getelementptr inbounds %Type, %Type* %evaluator.i.i, i64 0, i32 1
; CHECK-NEXT:   %call = call i64 @augmented_total(i64* nonnull %dims, i64* nonnull %[[dimsipge]])
; CHECK-NEXT:   %flt = uitofp i64 %call to float
; CHECK-NEXT:   %[[dataipge:.+]] = getelementptr inbounds %Type, %Type* %"evaluator.i.i'", i64 0, i32 0
; CHECK-NEXT:   %data = getelementptr inbounds %Type, %Type* %evaluator.i.i, i64 0, i32 0
; CHECK-NEXT:   store float %flt, float* %data, align 4
; CHECK-NEXT:   store float 0.000000e+00, float* %[[dataipge]], align 4
; CHECK-NEXT:   call void @diffetotal(i64* nonnull %dims, i64* nonnull %[[dimsipge]])
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal i64 @augmented_total(i64* %this, i64* %"this'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %loaded = load i64, i64* %this, align 4
; CHECK-NEXT:   %mcall = tail call i64 @meta(i64 %loaded)
; CHECK-NEXT:   ret i64 %mcall
; CHECK-NEXT: }

; CHECK: define internal void @diffetotal(i64* %this, i64* %"this'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
