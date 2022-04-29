; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

define dso_local noalias nonnull double* @_Z6toHeapd(double %x) {
entry:
  %call = call noalias nonnull dereferenceable(8) i8* @_Znwm(i64 8)
  %0 = bitcast i8* %call to double*
  store double %x, double* %0, align 8
  ret double* %0
}

declare dso_local nonnull i8* @_Znwm(i64)

define dso_local double @_Z6squared(double %x) {
entry:
  %call = call double* @_Z6toHeapd(double %x)
  %0 = load double, double* %call, align 8
  %mul = fmul double %0, %x
  ret double %mul
}

define dso_local double @_Z7dsquared(double %x) {
entry:
  %call = call double (...) @_Z16__enzyme_fwdsplitz(i8* bitcast (double (double)* @_Z6squared to i8*), metadata !"enzyme_nofree", double %x, double 1.000000e+00, i8* null)
  ret double %call
}

declare dso_local double @_Z16__enzyme_fwdsplitz(...)



; CHECK: define dso_local double @_Z7dsquared(double %x)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call fast double @fwddiffe_Z6squared(double %x, double 1.000000e+00, i8* null)
; CHECK-NEXT:   ret double %0
; CHECK-NEXT: }


; CHECK: define internal double @fwddiffe_Z6squared(double %x, double %"x'", i8* %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = bitcast i8* %tapeArg to { { i8*, i8* }, double*, double }*
; CHECK-NEXT:   %truetape = load { { i8*, i8* }, double*, double }, { { i8*, i8* }, double*, double }* %0
; CHECK-NEXT:   %tapeArg1 = extractvalue { { i8*, i8* }, double*, double } %truetape, 0
; CHECK-NEXT:   %[[i1:.+]] = call { double*, double* } @fwddiffe_Z6toHeapd(double %x, double %"x'", { i8*, i8* } %tapeArg1)
; CHECK-NEXT:   %[[i2:.+]] = extractvalue { double*, double* } %[[i1]], 1
; CHECK-NEXT:   %[[i4:.+]] = load double, double* %[[i2]], align 8
; CHECK-NEXT:   %[[i3:.+]] = extractvalue { { i8*, i8* }, double*, double } %truetape, 2
; CHECK-NEXT:   %[[i5:.+]] = fmul fast double %[[i4]], %x
; CHECK-NEXT:   %[[i6:.+]] = fmul fast double %"x'", %[[i3]]
; CHECK-NEXT:   %[[i7:.+]] = fadd fast double %[[i5]], %[[i6]]
; CHECK-NEXT:   ret double %[[i7]]
; CHECK-NEXT: }

; CHECK: define internal { double*, double* } @fwddiffe_Z6toHeapd(double %x, double %"x'", { i8*, i8* } %tapeArg) 
; CHECK-NEXT: entry:
; CHECK-NEXT:   %call = extractvalue { i8*, i8* } %tapeArg, 1
; CHECK-NEXT:   %"call'mi" = extractvalue { i8*, i8* } %tapeArg, 0
; CHECK-NEXT:   %"'ipc" = bitcast i8* %"call'mi" to double*
; CHECK-NEXT:   %0 = bitcast i8* %call to double*
; CHECK-NEXT:   store double %"x'", double* %"'ipc", align 8
; CHECK-NEXT:   %1 = insertvalue { double*, double* } undef, double* %0, 0
; CHECK-NEXT:   %2 = insertvalue { double*, double* } %1, double* %"'ipc", 1
; CHECK-NEXT:   ret { double*, double* } %2
; CHECK-NEXT: }
