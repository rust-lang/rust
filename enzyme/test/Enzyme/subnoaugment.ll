; RUN: %opt < %s %loadEnzyme -enzyme -enzyme_preopt=false -mem2reg -sroa -simplifycfg -instcombine -adce -S | FileCheck %s

%Type = type { float, double }

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
  ret double %inp
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

; CHECK: define internal {} @diffematvec(%Type* %evaluator.i.i, %Type* %"evaluator.i.i'") {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"dims'ipge" = getelementptr inbounds %Type, %Type* %"evaluator.i.i'", i64 0, i32 1
; CHECK-NEXT:   %dims = getelementptr inbounds %Type, %Type* %evaluator.i.i, i64 0, i32 1
; CHECK-NEXT:   %call_augmented = call { {}, double } @augmented_total(double* nonnull %dims, double* nonnull %"dims'ipge")
; CHECK-NEXT:   %call = extractvalue { {}, double } %call_augmented, 1
; CHECK-NEXT:   %flt = fptrunc double %call to float
; CHECK-NEXT:   %"data'ipge" = getelementptr inbounds %Type, %Type* %"evaluator.i.i'", i64 0, i32 0
; CHECK-NEXT:   %data = getelementptr inbounds %Type, %Type* %evaluator.i.i, i64 0, i32 0
; CHECK-NEXT:   store float %flt, float* %data, align 4
; CHECK-NEXT:   %0 = load float, float* %"data'ipge", align 4
; CHECK-NEXT:   store float 0.000000e+00, float* %"data'ipge", align 4
; CHECK-NEXT:   %1 = fpext float %0 to double
; CHECK-NEXT:   %[[unused:.+]] = call {} @diffetotal(double* nonnull %dims, double* nonnull %"dims'ipge", double %1, {} undef)
; CHECK-NEXT:   ret {} undef
; CHECK-NEXT: }

; CHECK: define internal { {}, double } @augmented_total(double* %this, double* %"this'") {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %loaded = load double, double* %this, align 8
; CHECK-NEXT:   %mcall = tail call double @meta(double %loaded)
; CHECK-NEXT:   %.fca.1.insert = insertvalue { {}, double } undef, double %mcall, 1
; CHECK-NEXT:   ret { {}, double } %.fca.1.insert
; CHECK-NEXT: }

; CHECK: define internal {} @diffetotal(double* %this, double* %"this'", double %differeturn, {} %tapeArg) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %loaded = load double, double* %this, align 8
; CHECK-NEXT:   %0 = call { double } @diffemeta(double %loaded, double %differeturn)
; CHECK-NEXT:   %1 = extractvalue { double } %0, 0
; CHECK-NEXT:   %2 = load double, double* %"this'", align 8
; CHECK-NEXT:   %3 = fadd fast double %2, %1
; CHECK-NEXT:   store double %3, double* %"this'", align 8
; CHECK-NEXT:   ret {} undef
; CHECK-NEXT: }

; CHECK: define internal { double } @diffemeta(double %inp, double %differeturn) #0 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %malloccall = tail call i8* @malloc(i64 8)
; CHECK-NEXT:   %"malloccall'mi" = tail call noalias nonnull i8* @malloc(i64 8)
; CHECK-NEXT:   %0 = bitcast i8* %"malloccall'mi" to i64*
; CHECK-NEXT:   store i64 0, i64* %0, align 1
; CHECK-NEXT:   %arr = bitcast i8* %malloccall to double*
; CHECK-NEXT:   store double %inp, double* %arr, align 8
; CHECK-NEXT:   %"arr'ipc1" = bitcast i8* %"malloccall'mi" to double*
; CHECK-NEXT:   %call.i_augmented = call { {}, double* } @augmented_sub(double* %arr, double*{{( nonnull)?}} %"arr'ipc1")
; CHECK-NEXT:   %"arr'ipc" = bitcast i8* %"malloccall'mi" to double*
; CHECK-NEXT:   %1 = call {} @diffesub(double* %arr, double*{{( nonnull)?}} %"arr'ipc", {} undef)
; CHECK-NEXT:   %"arr'ipc2" = bitcast i8* %"malloccall'mi" to double*
; CHECK-NEXT:   %2 = load double, double* %"arr'ipc2", align 8
; CHECK-NEXT:   %"arr'ipc3" = bitcast i8* %"malloccall'mi" to double*
; CHECK-NEXT:   store double 0.000000e+00, double* %"arr'ipc3", align 8
; CHECK-NEXT:   %3 = fadd fast double %2, %differeturn
; CHECK-NEXT:   tail call void @free(i8* nonnull %"malloccall'mi")
; CHECK-NEXT:   tail call void @free(i8* %malloccall)
; CHECK-NEXT:   %4 = insertvalue { double } undef, double %3, 0
; CHECK-NEXT:   ret { double } %4
; CHECK-NEXT: }

; CHECK: define internal { {}, double* } @augmented_sub(double* %a, double* %"a'") {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %.fca.1.insert = insertvalue { {}, double* } undef, double* %a, 1
; CHECK-NEXT:   ret { {}, double* } %.fca.1.insert
; CHECK-NEXT: }

; CHECK: define internal {} @diffesub(double* %a, double* %"a'", {} %tapeArg) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret {} undef
; CHECK-NEXT: }
