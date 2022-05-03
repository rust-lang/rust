; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -simplifycfg -instcombine -adce -instsimplify -S | FileCheck %s

declare dso_local double @__enzyme_autodiff(i8*, double)

define double @caller(double %in) {
entry:
  %call = call double @__enzyme_autodiff(i8* bitcast (double (double)* @foobard to i8*), double %in)
  ret double %call
}

define double @foobard(double %init) {
entry:
  %res = call double @subfn(double %init, void (double*, double*, double)* nonnull @indirect)
  ret double %res
}

define double @subfn(double %init, void (double*, double*, double)* %system) {
entry:
  ret double %init
}

define void @indirect(double* %x, double* %dxdt, double %t) {
entry:
  %a1 = load double, double* %x, align 8
  %call1 = call double* @bad(double* %dxdt)
  store double %a1, double* %call1, align 8, !tbaa !15
  ret void
}

define double* @bad(double* %this) {
entry:
  ret double* %this
}

!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}

!15 = !{!16, !16, i64 0}
!16 = !{!"double", !5, i64 0}

; CHECK: @"_enzyme_reverse_indirect'" = internal constant { i8* (double*, double*, double*, double*, double)*, { double } (double*, double*, double*, double*, double, i8*)* } { i8* (double*, double*, double*, double*, double)* @augmented_indirect, { double } (double*, double*, double*, double*, double, i8*)* @diffeindirect }

; CHECK: define internal { double } @diffefoobard(double %init, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call { double } @diffesubfn(double %init, void (double*, double*, double)* nonnull @indirect, void (double*, double*, double)* bitcast ({ i8* (double*, double*, double*, double*, double)*, { double } (double*, double*, double*, double*, double, i8*)* }* @"_enzyme_reverse_indirect'" to void (double*, double*, double)*), double %differeturn)
; CHECK-NEXT:   ret { double } %0
; CHECK-NEXT: }

; CHECK: define internal { double*, double* } @augmented_bad(double* %this, double* %"this'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %.fca.0.insert = insertvalue { double*, double* } undef, double* %this, 0
; CHECK-NEXT:   %.fca.1.insert = insertvalue { double*, double* } %.fca.0.insert, double* %"this'", 1
; CHECK-NEXT:   ret { double*, double* } %.fca.1.insert
; CHECK-NEXT: }

; CHECK: define internal i8* @augmented_indirect(double* %x, double* %"x'", double* %dxdt, double* %"dxdt'", double %t)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %malloccall = tail call noalias nonnull dereferenceable(8) dereferenceable_or_null(8) i8* @malloc(i64 8)
; CHECK-NEXT:   %tapemem = bitcast i8* %malloccall to double**
; CHECK-NEXT:   %a1 = load double, double* %x, align 8
; CHECK-NEXT:   %call1_augmented = call { double*, double* } @augmented_bad(double* %dxdt, double* %"dxdt'")
; CHECK-NEXT:   %call1 = extractvalue { double*, double* } %call1_augmented, 0
; CHECK-NEXT:   %[[dcall:.+]] = extractvalue { double*, double* } %call1_augmented, 1
; CHECK-NEXT:   store double* %[[dcall]], double** %tapemem
; CHECK-NEXT:   store double %a1, double* %call1, align 8
; CHECK-NEXT:   ret i8* %malloccall
; CHECK-NEXT: }

; CHECK: define internal { double } @diffeindirect(double* %x, double* %"x'", double* %dxdt, double* %"dxdt'", double %t, i8* %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[elt1:.+]] = bitcast i8* %tapeArg to double**
; CHECK-NEXT:   %[[unpack:.+]] = load double*, double** %[[elt1]], align 8
; CHECK-NEXT:   tail call void @free(i8* nonnull %tapeArg)
; CHECK-NEXT:   %[[loadc:.+]] = load double, double* %[[unpack]], align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %[[unpack]], align 8
; CHECK-NEXT:   call void @diffebad(double* %dxdt, double* %"dxdt'")
; CHECK-NEXT:   %[[xpl:.+]] = load double, double* %"x'", align 8
; CHECK-NEXT:   %[[fadd:.+]] = fadd fast double %[[xpl]], %[[loadc]]
; CHECK-NEXT:   store double %[[fadd]], double* %"x'", align 8
; CHECK-NEXT:   ret { double } zeroinitializer
; CHECK-NEXT: }

; CHECK: define internal void @diffebad(double* %this, double* %"this'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal { double } @diffesubfn(double %init, void (double*, double*, double)* %system, void (double*, double*, double)* %"system'", double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = insertvalue { double } undef, double %differeturn, 0
; CHECK-NEXT:   ret { double } %0
; CHECK-NEXT: }
