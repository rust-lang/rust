; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

%struct.Gradients = type { double, double, double }

; Function Attrs: nounwind
declare %struct.Gradients @__enzyme_fwddiff(...)

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

define dso_local %struct.Gradients @_Z7dsquared(double %x) {
entry:
  %call = call %struct.Gradients (...) @__enzyme_fwddiff(i8* bitcast (double (double)* @_Z6squared to i8*), metadata !"enzyme_width", i64 3, double %x, double 1.0, double 2.0, double 3.0)
  ret %struct.Gradients %call
}


; CHECK: define internal { double*, [3 x double*] } @fwddiffe3_Z6toHeapd(double %x, [3 x double] %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %call = call noalias nonnull dereferenceable(8) i8* @_Znwm(i64 8)
; CHECK-NEXT:   %0 = call noalias nonnull dereferenceable(8) i8* @_Znwm(i64 8)
; CHECK-NEXT:   %1 = call noalias nonnull dereferenceable(8) i8* @_Znwm(i64 8)
; CHECK-NEXT:   %2 = call noalias nonnull dereferenceable(8) i8* @_Znwm(i64 8)
; CHECK-NEXT:   %"'ipc" = bitcast i8* %0 to double*
; CHECK-NEXT:   %3 = insertvalue [3 x double*] undef, double* %"'ipc", 0
; CHECK-NEXT:   %"'ipc1" = bitcast i8* %1 to double*
; CHECK-NEXT:   %4 = insertvalue [3 x double*] %3, double* %"'ipc1", 1
; CHECK-NEXT:   %"'ipc2" = bitcast i8* %2 to double*
; CHECK-NEXT:   %5 = insertvalue [3 x double*] %4, double* %"'ipc2", 2
; CHECK-NEXT:   %6 = bitcast i8* %call to double*
; CHECK-NEXT:   store double %x, double* %6, align 8
; CHECK-NEXT:   %7 = extractvalue [3 x double] %"x'", 0
; CHECK-NEXT:   store double %7, double* %"'ipc", align 8
; CHECK-NEXT:   %8 = extractvalue [3 x double] %"x'", 1
; CHECK-NEXT:   store double %8, double* %"'ipc1", align 8
; CHECK-NEXT:   %9 = extractvalue [3 x double] %"x'", 2
; CHECK-NEXT:   store double %9, double* %"'ipc2", align 8
; CHECK-NEXT:   %10 = insertvalue { double*, [3 x double*] } undef, double* %6, 0
; CHECK-NEXT:   %11 = insertvalue { double*, [3 x double*] } %10, [3 x double*] %5, 1
; CHECK-NEXT:   ret { double*, [3 x double*] } %11
; CHECK-NEXT: }