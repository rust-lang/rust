; RUN: %opt < %s %loadEnzyme -enzyme -enzyme_preopt=false -mem2reg -sroa -instsimplify -simplifycfg -S | FileCheck %s

declare double @__enzyme_autodiff(i8*, ...)

define void @lame(double* %data, double* %ddata, i64* %W, i64* %Wp) {
entry:
  %call11 = call double (i8*, ...) @__enzyme_autodiff(i8* bitcast (void (double*, i64*)* @matvec to i8*), double* %data, double* %ddata, metadata !"diffe_dup", i64* %W, i64* %Wp)
  ret void
}

define void @matvec(double* %this, i64* %d0) {
entry:
  %call = call double @metaloader(double* %this)
  store double %call, double* %this, align 8
  ret void
}

define double @metaloader(double* %a) {
entry:
  %call = call double @loader(double* %a)
  ret double %call
}

define double @loader(double* %a) {
entry:
  %0 = load double, double* %a, align 8
  %mul = fmul double %0, %0
  ret double %mul
}


; CHECK: define internal {} @diffematvec(double* %this, double* %"this'", i64* %d0, i64* %"d0'") {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %call_augmented = call { { { double } }, double } @augmented_metaloader(double* %this, double* %"this'")
; CHECK-NEXT:   %0 = extractvalue { { { double } }, double } %call_augmented, 0
; CHECK-NEXT:   %1 = extractvalue { { { double } }, double } %call_augmented, 1
; CHECK-NEXT:   store double %1, double* %this, align 8
; CHECK-NEXT:   %2 = load double, double* %"this'"
; CHECK-NEXT:   store double 0.000000e+00, double* %"this'", align 8
; CHECK-NEXT:   %3 = call {} @diffemetaloader(double* %this, double* %"this'", double %2, { { double } } %0)
; CHECK-NEXT:   ret {} undef
; CHECK-NEXT: }

; CHECK: define internal { { double }, double } @augmented_loader(double* %a, double* %"a'") {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = load double, double* %a, align 8
; CHECK-NEXT:   %mul = fmul double %0, %0
; CHECK-NEXT:   %.fca.0.0.insert = insertvalue { { double }, double } undef, double %0, 0, 0
; CHECK-NEXT:   %.fca.1.insert = insertvalue { { double }, double } %.fca.0.0.insert, double %mul, 1
; CHECK-NEXT:   ret { { double }, double } %.fca.1.insert
; CHECK-NEXT: }

; CHECK: define internal { { { double } }, double } @augmented_metaloader(double* %a, double* %"a'") {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %call_augmented = call { { double }, double } @augmented_loader(double* %a, double* %"a'")
; CHECK-NEXT:   %subcache = extractvalue { { double }, double } %call_augmented, 0
; CHECK-NEXT:   %subcache.fca.0.extract = extractvalue { double } %subcache, 0
; CHECK-NEXT:   %0 = extractvalue { { double }, double } %call_augmented, 1
; CHECK-NEXT:   %.fca.0.0.0.insert = insertvalue { { { double } }, double } undef, double %subcache.fca.0.extract, 0, 0, 0
; CHECK-NEXT:   %.fca.1.insert = insertvalue { { { double } }, double } %.fca.0.0.0.insert, double %0, 1
; CHECK-NEXT:   ret { { { double } }, double } %.fca.1.insert
; CHECK-NEXT: }

; CHECK: define internal {} @diffemetaloader(double* %a, double* %"a'", double %differeturn, { { double } } %tapeArg) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = extractvalue { { double } } %tapeArg, 0
; CHECK-NEXT:   %1 = call {} @diffeloader(double* %a, double* %"a'", double %differeturn, { double } %0)
; CHECK-NEXT:   ret {} undef
; CHECK-NEXT: }

; CHECK: define internal {} @diffeloader(double* %a, double* %"a'", double %differeturn, { double } %tapeArg) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %_fromtape_unwrap = extractvalue { double } %tapeArg, 0
; CHECK-NEXT:   %m0diffe = fmul fast double %differeturn, %_fromtape_unwrap
; CHECK-NEXT:   %m1diffe = fmul fast double %differeturn, %_fromtape_unwrap
; CHECK-NEXT:   %0 = fadd fast double %m0diffe, %m1diffe
; CHECK-NEXT:   %1 = load double, double* %"a'"
; CHECK-NEXT:   %2 = fadd fast double %1, %0
; CHECK-NEXT:   store double %2, double* %"a'"
; CHECK-NEXT:   ret {} undef
; CHECK-NEXT: }
