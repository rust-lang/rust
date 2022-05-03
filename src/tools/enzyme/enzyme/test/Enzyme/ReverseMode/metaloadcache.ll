; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -instsimplify -simplifycfg -S | FileCheck %s

declare double @__enzyme_autodiff(i8*, ...)

define void @lame(double* %data, double* %ddata, i64* %W, i64* %Wp) {
entry:
  %call11 = call double (i8*, ...) @__enzyme_autodiff(i8* bitcast (void (double*, i64*)* @matvec to i8*), double* %data, double* %ddata, metadata !"enzyme_dup", i64* %W, i64* %Wp)
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


; CHECK: define internal void @diffematvec(double* %this, double* %"this'", i64* %d0, i64* %"d0'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %call_augmented = call { double, double } @augmented_metaloader(double* %this, double* %"this'")
; CHECK-NEXT:   %[[eca:.+]] = extractvalue { double, double } %call_augmented, 0
; CHECK-NEXT:   %call = extractvalue { double, double } %call_augmented, 1
; CHECK-NEXT:   store double %call, double* %this, align 8
; CHECK-NEXT:   %[[dth:.+]] = load double, double* %"this'"
; CHECK-NEXT:   store double 0.000000e+00, double* %"this'", align 8
; CHECK-NEXT:   call void @diffemetaloader(double* %this, double* %"this'", double %[[dth]], double %[[eca]])
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal { double, double } @augmented_loader(double* %a, double* %"a'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = load double, double* %a, align 8
; CHECK-NEXT:   %mul = fmul double %0, %0
; CHECK-NEXT:   %.fca.0.insert = insertvalue { double, double } undef, double %0, 0
; CHECK-NEXT:   %.fca.1.insert = insertvalue { double, double } %.fca.0.insert, double %mul, 1
; CHECK-NEXT:   ret { double, double } %.fca.1.insert
; CHECK-NEXT: }

; CHECK: define internal { double, double } @augmented_metaloader(double* %a, double* %"a'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %call_augmented = call { double, double } @augmented_loader(double* %a, double* %"a'")
; CHECK-NEXT:   ret { double, double } %call_augmented
; CHECK-NEXT: }

; CHECK: define internal void @diffemetaloader(double* %a, double* %"a'", double %differeturn, double %[[ev:.+]])
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @diffeloader(double* %a, double* %"a'", double %differeturn, double %[[ev]])
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @diffeloader(double* %a, double* %"a'", double %differeturn, double
; CHECK-NEXT: entry:
; CHECK-NEXT:   %m0diffe = fmul fast double %differeturn, %0
; CHECK-NEXT:   %m1diffe = fmul fast double %differeturn, %0
; CHECK-NEXT:   %[[de:.+]] = fadd fast double %m0diffe, %m1diffe
; CHECK-NEXT:   %[[pra:.+]] = load double, double* %"a'"
; CHECK-NEXT:   %[[pa:.+]] = fadd fast double %[[pra]], %[[de]]
; CHECK-NEXT:   store double %[[pa]], double* %"a'"
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
