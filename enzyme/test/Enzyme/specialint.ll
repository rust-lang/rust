; ModuleID = 'inp.ll'
source_filename = "inp.ll"

%Type = type { float, i64 }

declare dso_local double @__enzyme_autodiff(i8*, ...)

; Function Attrs: alwaysinline norecurse nounwind uwtable
define double @caller(%Type* %K, %Type* %Kp) local_unnamed_addr #0 {
entry:
  %call86 = call double (i8*, ...) @__enzyme_autodiff(i8* bitcast (void (%Type*)* @matvec to i8*), metadata !"diffe_dup", %Type* noalias %K, %Type* noalias %Kp) #4
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
