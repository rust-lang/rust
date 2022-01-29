; RUN: if [ %llvmver -le 11 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -simplifycfg -instcombine -early-cse -adce -S | FileCheck %s; fi

; This fails because activity analysis deduces that sub value is not active, when it indeed is

%Type = type { float, double }

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
  %call = call double @total(double* %dims) #4
  %flt = fptrunc double %call to float
  %data = getelementptr inbounds %Type, %Type* %evaluator.i.i, i64 0, i32 0
  store float %flt, float* %data, align 4
  ret void
}

; Function Attrs: readnone
define double @meta(double %inp) #3 {
entry:
  %arr = alloca double
  store double %inp, double* %arr
  %call.i = call double* @sub(double* %arr)
  %a1 = load double, double* %call.i
  ret double %a1
}

define double* @sub(double* %a) {
entry:
  ret double* %a
}

define double @total(double* %this) {
entry:
  %loaded = load double, double* %this
  %mcall = tail call double @meta(double %loaded)
  ret double %mcall
}

attributes #3 = { readnone }

; CHECK: define internal void @diffematvec(%Type* %evaluator.i.i, %Type* %"evaluator.i.i'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[dimsipge:.+]] = getelementptr inbounds %Type, %Type* %"evaluator.i.i'", i64 0, i32 1
; CHECK-NEXT:   %dims = getelementptr inbounds %Type, %Type* %evaluator.i.i, i64 0, i32 1
; CHECK-NEXT:   %call = call fast double @augmented_total(double* nonnull %dims, double* nonnull %[[dimsipge]])
; CHECK-NEXT:   %flt = fptrunc double %call to float
; CHECK-NEXT:   %[[dataipge:.+]] = getelementptr inbounds %Type, %Type* %"evaluator.i.i'", i64 0, i32 0
; CHECK-NEXT:   %data = getelementptr inbounds %Type, %Type* %evaluator.i.i, i64 0, i32 0
; CHECK-NEXT:   store float %flt, float* %data, align 4
; CHECK-NEXT:   %0 = load float, float* %[[dataipge:.+]], align 4
; CHECK-NEXT:   store float 0.000000e+00, float* %[[dataipge:.+]], align 4
; CHECK-NEXT:   %1 = fpext float %0 to double
; CHECK-NEXT:   call void @diffetotal(double* nonnull %dims, double* nonnull %[[dimsipge]], double %1)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal double @augmented_total(double* %this, double* %"this'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %loaded = load double, double* %this, align 8
; CHECK-NEXT:   %mcall = tail call double @meta(double %loaded)
; CHECK-NEXT:   ret double %mcall
; CHECK-NEXT: }

; CHECK: define internal void @diffetotal(double* %this, double* %"this'", double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[loaded:.+]] = load double, double* %this, align 8
; CHECK-NEXT:   %[[dmetastruct:.+]] = call { double } @diffemeta(double %[[loaded]], double %differeturn)
; CHECK-NEXT:   %[[dmeta:.+]] = extractvalue { double } %[[dmetastruct]], 0
; CHECK-NEXT:   %[[prethis:.+]] = load double, double* %"this'", align 8
; CHECK-NEXT:   %[[postthis:.+]] = fadd fast double %[[prethis]], %[[dmeta]]
; CHECK-NEXT:   store double %[[postthis:.+]], double* %"this'", align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal { double } @diffemeta(double %inp, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"arr'ipa" = alloca double, align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"arr'ipa", align 8
; CHECK-NEXT:   %arr = alloca double, align 8
; CHECK-NEXT:   store double %inp, double* %arr, align 8
; CHECK-NEXT:   %call.i_augmented = call double* @augmented_sub(double*{{( nonnull)?}} %arr, double*{{( nonnull)?}} %"arr'ipa")
; CHECK-NEXT:   %0 = load double, double* %call.i_augmented, align 8
; CHECK-NEXT:   %1 = fadd fast double %0, %differeturn
; CHECK-NEXT:   store double %1, double* %call.i_augmented
; CHECK-NEXT:   call void @diffesub(double*{{( nonnull)?}} %arr, double*{{( nonnull)?}} %"arr'ipa")
; CHECK-NEXT:   %[[prevv:.+]] = load double, double* %"arr'ipa", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"arr'ipa", align 8
; CHECK-NEXT:   %[[res:.+]] = insertvalue { double } undef, double %[[prevv]], 0
; CHECK-NEXT:   ret { double } %[[res]]
; CHECK-NEXT: }

; CHECK: define internal double* @augmented_sub(double* %a, double* %"a'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret double* %"a'"
; CHECK-NEXT: }

; CHECK: define internal void @diffesub(double* %a, double* %"a'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
