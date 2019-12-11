; RUN: %opt < %s %loadEnzyme -enzyme -enzyme_preopt=false -mem2reg -sroa -simplifycfg -instcombine -adce -S | FileCheck %s

declare dso_local double @__enzyme_autodiff(i8*, double)

define double @caller(double %in) {
entry:
  %0 = call double @diffefoobard()
  %oldret = insertvalue { double } undef, double %0, 0
  %1 = extractvalue { double } %oldret, 0
  ret double %1
}

define double @foobard(double %init) {
entry:
  call void @subfn(double undef, void (double*, double*, double)* nonnull undef)
  ret double 0.000000e+00
}

define void @subfn(double %init, void (double*, double*, double)* %system) {
entry:
  ret void
}

define void @indirect(double* %x, double* %dxdt, double %t) {
entry:
  %a1 = load double, double* %x, align 8
  %call1 = call double* @bad(double* %dxdt)
  store double %a1, double* %call1, align 8
  ret void
}

define double* @bad(double* %this) {
entry:
  ret double* %this
}

define double @preprocess_foobard(double %init) {
entry:
  call void @subfn(double undef, void (double*, double*, double)* nonnull undef)
  ret double 0.000000e+00
}

define internal double @diffefoobard() {
entry:
  %0 = call double @diffesubfn()
  %oldret1 = insertvalue { double } undef, double %0, 0
  %oldret = extractvalue { double } %oldret1, 0
  ret double %oldret
}

define void @preprocess_indirect(double* %x, double* %dxdt, double %t) {
entry:
  %a1 = load double, double* %x, align 8
  %call1 = call double* @bad(double* %dxdt)
  store double %a1, double* %call1, align 8
  ret void
}

define internal { double } @diffeindirect(double* %x, double* %"x'", double* %dxdt, double* %"dxdt'", double %t) {
entry:
  %a1 = load double, double* %x, align 8
  %call1_augmented = call { double*, double* } @augmented_bad(double* %dxdt, double* %"dxdt'")
  %newret = extractvalue { double*, double* } %call1_augmented, 0
  %oldret = insertvalue { {}, double*, double* } undef, double* %newret, 1
  %newret1 = extractvalue { double*, double* } %call1_augmented, 1
  %oldret2 = insertvalue { {}, double*, double* } %oldret, double* %newret1, 2
  %0 = extractvalue { {}, double*, double* } %oldret2, 2
  %1 = extractvalue { {}, double*, double* } %oldret2, 1
  store double %a1, double* %1, align 8
  %2 = load double, double* %0
  store double 0.000000e+00, double* %0, align 8
  call void @diffebad()
  %3 = load double, double* %"x'"
  %4 = fadd fast double %3, %2
  store double %4, double* %"x'"
  ret { double } zeroinitializer
}

declare i8* @realloc(i8*, i64)

define double* @preprocess_bad(double* %this) {
entry:
  ret double* %this
}

define internal { double*, double* } @augmented_bad(double* %this, double* %"this'") {
entry:
  %.fca.1.insert = insertvalue { {}, double*, double* } undef, double* %this, 1
  %.fca.2.insert = insertvalue { {}, double*, double* } %.fca.1.insert, double* %"this'", 2
  %oldret = extractvalue { {}, double*, double* } %.fca.2.insert, 1
  %newret = insertvalue { double*, double* } undef, double* %oldret, 0
  %oldret1 = extractvalue { {}, double*, double* } %.fca.2.insert, 2
  %newret2 = insertvalue { double*, double* } %newret, double* %oldret1, 1
  ret { double*, double* } %newret2
}

define internal void @diffebad() {
entry:
  ret void
}

define void @preprocess_subfn(double %init, void (double*, double*, double)* %system) {
entry:
  ret void
}

define internal double @diffesubfn() {
entry:
  %oldret = extractvalue { double } zeroinitializer, 0
  ret double %oldret
}
