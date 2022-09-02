; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -simplifycfg -instsimplify -gvn -adce -S | FileCheck %s

define void @set(double* nocapture %a, double %x) {
entry:
  store double %x, double* %a, align 8
  ret void
}

define double @above(double %i10) {
entry:
  %m = alloca double, align 8
  call void @set(double* nonnull %m, double %i10)
  %i12 = load double, double* %m, align 8
  ret double %i12
}

define double @msg(double %in) {
entry:
  %hst = call double @above(double %in)
  %r = fmul double %hst, %hst
  ret double %r
}

; Function Attrs: norecurse nounwind uwtable

define double @caller() {
entry:
  %r = call double (i8*, ...) @__enzyme_autodiff(i8* bitcast (double (double)* @msg to i8*), double 2.000000e+00)
  ret double %r
}

declare dso_local double @__enzyme_autodiff(i8*, ...)

; CHECK: define internal { double } @diffemsg(double %in, double %differeturn) 
; CHECK-NEXT: entry:
; CHECK-NEXT:   %hst_augmented = call { i8*, double } @augmented_above(double %in)
; CHECK-NEXT:   %subcache = extractvalue { i8*, double } %hst_augmented, 0
; CHECK-NEXT:   %hst = extractvalue { i8*, double } %hst_augmented, 1
; CHECK-NEXT:   %m0diffehst = fmul fast double %differeturn, %hst
; CHECK-NEXT:   %[[i1:.+]] = fadd fast double %m0diffehst, %m0diffehst
; CHECK-NEXT:   %[[i2:.+]] = call { double } @diffeabove(double %in, double %[[i1]], i8* %subcache)
; CHECK-NEXT:   ret { double } %[[i2]]
; CHECK-NEXT: }

; CHECK: define internal void @augmented_set(double* nocapture %a, double* nocapture %"a'", double %x) 
; CHECK-NEXT: entry:
; CHECK-NEXT:   store double %x, double* %a, align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal { i8*, double } @augmented_above(double %i10) 
; CHECK-NEXT: entry:
; CHECK-NEXT:   %malloccall = tail call noalias nonnull dereferenceable(8) dereferenceable_or_null(8) i8* @malloc(i64 8), !enzyme_fromstack !
; CHECK-NEXT:   %m = bitcast i8* %malloccall to double*
; CHECK-NEXT:   call void @augmented_set(double* %m, double* undef, double %i10)
; CHECK-NEXT:   %i12 = load double, double* %m, align 8
; CHECK-NEXT:   %.fca.0.insert = insertvalue { i8*, double } {{(undef|poison)}}, i8* %malloccall, 0
; CHECK-NEXT:   %.fca.1.insert = insertvalue { i8*, double } %.fca.0.insert, double %i12, 1
; CHECK-NEXT:   ret { i8*, double } %.fca.1.insert
; CHECK-NEXT: }
 
; TODO not need to cache the primal
; CHECK: define internal { double } @diffeabove(double %i10, double %differeturn, i8* %malloccall) 
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"malloccall'mi" = alloca i8, i64 8, align 8
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull dereferenceable(8) dereferenceable_or_null(8) %"malloccall'mi", i8 0, i64 8, i1 false)
; CHECK-NEXT:   %"m'ipc" = bitcast i8* %"malloccall'mi" to double*
; CHECK-NEXT:   %m = bitcast i8* %malloccall to double*
; CHECK-NEXT:   store double %differeturn, double* %"m'ipc", align 8
; CHECK-NEXT:   %0 = call { double } @diffeset(double* %m, double* %"m'ipc", double %i10)
; CHECK-NEXT:   tail call void @free(i8* %malloccall)
; CHECK-NEXT:   ret { double } %0
; CHECK-NEXT: }

; CHECK: define internal { double } @diffeset(double* nocapture %a, double* nocapture %"a'", double %x) 
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = load double, double* %"a'", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"a'", align 8
; CHECK-NEXT:   %1 = insertvalue { double } undef, double %0, 0
; CHECK-NEXT:   ret { double } %1
; CHECK-NEXT: }
