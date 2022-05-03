; RUN: if [ %llvmver -le 11 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -simplifycfg -instsimplify -early-cse -adce -S | FileCheck %s; fi

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
  %mul = fmul double %a1, %a1
  ret double %mul
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
; CHECK-NEXT:   %call = call fast double @augmented_total(double* %dims, double* %[[dimsipge]])
; CHECK-NEXT:   %flt = fptrunc double %call to float
; CHECK-NEXT:   %[[dataipge:.+]] = getelementptr inbounds %Type, %Type* %"evaluator.i.i'", i64 0, i32 0
; CHECK-NEXT:   %data = getelementptr inbounds %Type, %Type* %evaluator.i.i, i64 0, i32 0
; CHECK-NEXT:   store float %flt, float* %data, align 4
; CHECK-NEXT:   %0 = load float, float* %[[dataipge:.+]], align 4
; CHECK-NEXT:   store float 0.000000e+00, float* %[[dataipge:.+]], align 4
; CHECK-NEXT:   %1 = fpext float %0 to double
; CHECK-NEXT:   call void @diffetotal(double* %dims, double* %[[dimsipge]], double %1)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal double @augmented_total(double* %this, double* %"this'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %loaded = load double, double* %this
; CHECK-NEXT:   %mcall = tail call double @meta(double %loaded)
; CHECK-NEXT:   ret double %mcall
; CHECK-NEXT: }

; CHECK: define internal void @diffetotal(double* %this, double* %"this'", double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[loaded:.+]] = load double, double* %this
; CHECK-NEXT:   %[[dmetastruct:.+]] = call { double } @diffemeta(double %[[loaded]], double %differeturn)
; CHECK-NEXT:   %[[dmeta:.+]] = extractvalue { double } %[[dmetastruct]], 0
; CHECK-NEXT:   %[[prethis:.+]] = load double, double* %"this'"
; CHECK-NEXT:   %[[postthis:.+]] = fadd fast double %[[prethis]], %[[dmeta]]
; CHECK-NEXT:   store double %[[postthis:.+]], double* %"this'"
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal { double } @diffemeta(double %inp, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"arr'ipa" = alloca double
; CHECK-NEXT:   store double 0.000000e+00, double* %"arr'ipa"
; CHECK-NEXT:   %arr = alloca double
; CHECK-NEXT:   store double %inp, double* %arr
; CHECK-NEXT:   %call.i_augmented = call { double*, double* } @augmented_sub(double*{{( nonnull)?}} %arr, double*{{( nonnull)?}} %"arr'ipa")
; CHECK-NEXT:   %[[oldptr:.+]] = extractvalue { double*, double* } %call.i_augmented, 0
; CHECK-NEXT:   %[[olddptr:.+]] = extractvalue { double*, double* } %call.i_augmented, 1
; CHECK-NEXT:   %[[load:.+]] = load double, double* %[[oldptr]]
; CHECK-NEXT:   %[[mul:.+]] = fmul fast double %differeturn, %[[load]]
; CHECK-NEXT:   %[[add:.+]] = fadd fast double %[[mul]], %[[mul]]
; CHECK-NEXT:   %[[dcall:.+]] = load double, double* %"call.i'ac"
; CHECK-NEXT:   %[[dadd:.+]] = fadd fast double %[[dcall]], %[[add]]
; CHECK-NEXT:   store double %[[dadd]], double* %"call.i'ac"
; CHECK-NEXT:   call void @diffesub(double*{{( nonnull)?}} %arr, double*{{( nonnull)?}} %"arr'ipa")
; CHECK-NEXT:   %[[darr:.+]] = load double, double* %"arr'ipa"
; CHECK-NEXT:   store double 0.000000e+00, double* %"arr'ipa"
; CHECK-NEXT:   %[[ret:.+]] = insertvalue { double } undef, double %[[darr]], 0
; CHECK-NEXT:   ret { double } %[[ret]]
; CHECK-NEXT: }

; TODO don't need the diffe ret
; CHECK: define internal { double*, double* } @augmented_sub(double* %a, double* %"a'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %.fca.0.insert = insertvalue { double*, double* } undef, double* %a, 0
; CHECK-NEXT:   %.fca.1.insert = insertvalue { double*, double* } %.fca.0.insert, double* %"a'", 1
; CHECK-NEXT:   ret { double*, double* } %.fca.1.insert
; CHECK-NEXT: }

; CHECK: define internal void @diffesub(double* %a, double* %"a'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
