; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -simplifycfg -instcombine -adce -S | FileCheck %s
; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | %lli - | FileCheck %s --check-prefix=EVAL

; EVAL: d_reduce_max(0)=1.000000

source_filename = "multivecmax.cpp"

@.str.1 = private unnamed_addr constant [21 x i8] c"d_reduce_max(%i)=%f\0A\00", align 1

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local double* @_ZNSt16allocator_traitsISaIdEE8allocateERS0_m(i64 %__n) {
entry:
  %mul = shl i64 %__n, 3
  %call5 = call i8* @_Znwm(i64 %mul)
  %aa1 = bitcast i8* %call5 to double*
  ret double* %aa1
}

; Function Attrs: nounwind
define dso_local double @_Z10reduce_maxPdi(double %v) {
entry:
  %call6.i.i = call double* @_ZNSt16allocator_traitsISaIdEE8allocateERS0_m(i64 1)
  store double %v, double* %call6.i.i, align 8
  %res = load double, double* %call6.i.i, align 8
  ret double %res
}

declare dso_local double @_Z17__enzyme_autodiffPvPdS0_i(...)

define dso_local i32 @main() {
entry:
  %r = call double (...) @_Z17__enzyme_autodiffPvPdS0_i(i8* bitcast (double (double)* @_Z10reduce_maxPdi to i8*), double 1.000000e+00)
  ; THIS SHOULD PRINT 1.0
  %call4 = call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([21 x i8], [21 x i8]* @.str.1, i64 0, i64 0), i32 0, double %r)
  ret i32 0
}

declare dso_local i32 @printf(i8* nocapture readonly, ...)

declare dso_local noalias nonnull i8* @_Znwm(i64)

; CHECK: define internal { double } @diffe_Z10reduce_maxPdi(double %v, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %call6.i.i_augmented = call { { i8*, i8* }, double*, double* } @augmented__ZNSt16allocator_traitsISaIdEE8allocateERS0_m(i64 1)
; CHECK-NEXT:   %subcache = extractvalue { { i8*, i8* }, double*, double* } %call6.i.i_augmented, 0
; CHECK-NEXT:   %call6.i.i = extractvalue { { i8*, i8* }, double*, double* } %call6.i.i_augmented, 1
; CHECK-NEXT:   %"call6.i.i'ac" = extractvalue { { i8*, i8* }, double*, double* } %call6.i.i_augmented, 2
; CHECK-NEXT:   store double %v, double* %call6.i.i, align 8, !alias.scope !0, !noalias !3
; CHECK-NEXT:   %0 = load double, double* %"call6.i.i'ac", align 8, !alias.scope !3, !noalias !0
; CHECK-NEXT:   %1 = fadd fast double %0, %differeturn
; CHECK-NEXT:   store double 0.000000e+00, double* %"call6.i.i'ac", align 8, !alias.scope !3, !noalias !0
; CHECK-NEXT:   call void @diffe_ZNSt16allocator_traitsISaIdEE8allocateERS0_m(i64 1, { i8*, i8* } %subcache)
; CHECK-NEXT:   %2 = insertvalue { double } {{(undef|poison)}}, double %1, 0
; CHECK-NEXT:   ret { double } %2
; CHECK-NEXT: }
