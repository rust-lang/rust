; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

declare dso_local double @__enzyme_fwddiff(i8*, double, double)

declare i8* @malloc(i64)

define { i8*, double } @augsquare(double %x) {
entry:
  %m = tail call noalias nonnull dereferenceable(64) dereferenceable_or_null(64) i8* @malloc(i64 64)
  %a = fadd double %x, %x
  %.fca.0.insert = insertvalue { i8*, double } undef, i8* %m, 0
  %.fca.1.insert = insertvalue { i8*, double } %.fca.0.insert, double %a, 1
  ret { i8*, double } %.fca.1.insert
}

define double @square(double %x) {
entry:
  %ext = call { i8*, double } @augsquare(double %x)
  %o = extractvalue { i8*, double } %ext, 1
  %mul = fmul double %o, %o
  ret double %mul
}

define double @dsquare(double %x) local_unnamed_addr {
entry:
  %call = tail call double @__enzyme_fwddiff(i8* bitcast (double (double)* @square to i8*), double %x, double 1.000000)
  ret double %call
}


; CHECK: define internal double @fwddiffesquare(double %x, double %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call { { i8*, double }, { i8*, double } } @fwddiffeaugsquare(double %x, double %"x'")
; CHECK-NEXT:   %1 = extractvalue { { i8*, double }, { i8*, double } } %0, 0
; CHECK-NEXT:   %2 = extractvalue { { i8*, double }, { i8*, double } } %0, 1
; CHECK-NEXT:   %[[i3:.+]] = extractvalue { i8*, double } %2, 1
; CHECK-NEXT:   %o = extractvalue { i8*, double } %1, 1
; CHECK-NEXT:   %[[i4:.+]] = fmul fast double %[[i3]], %o
; CHECK-NEXT:   %[[i5:.+]] = fmul fast double %[[i3]], %o
; CHECK-NEXT:   %[[i6:.+]] = fadd fast double %[[i4]], %[[i5]]
; CHECK-NEXT:   ret double %[[i6]]
; CHECK-NEXT: }


; CHECK: define internal { { i8*, double }, { i8*, double } } @fwddiffeaugsquare(double %x, double %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %m = tail call noalias nonnull dereferenceable(64) dereferenceable_or_null(64) i8* @malloc(i64 64)
; CHECK-NEXT:   %0 = tail call noalias nonnull dereferenceable(64) dereferenceable_or_null(64) i8* @malloc(i64 64)
; CHECK-NEXT:   %a = fadd double %x, %x
; CHECK-NEXT:   %1 = fadd fast double %"x'", %"x'"
; CHECK-NEXT:   %".fca.0.insert'ipiv" = insertvalue { i8*, double } zeroinitializer, i8* %0, 0
; CHECK-NEXT:   %.fca.0.insert = insertvalue { i8*, double } undef, i8* %m, 0
; CHECK-NEXT:   %".fca.1.insert'ipiv" = insertvalue { i8*, double } %".fca.0.insert'ipiv", double %1, 1
; CHECK-NEXT:   %.fca.1.insert = insertvalue { i8*, double } %.fca.0.insert, double %a, 1
; CHECK-NEXT:   %2 = insertvalue { { i8*, double }, { i8*, double } } undef, { i8*, double } %.fca.1.insert, 0
; CHECK-NEXT:   %3 = insertvalue { { i8*, double }, { i8*, double } } %2, { i8*, double } %".fca.1.insert'ipiv", 1
; CHECK-NEXT:   ret { { i8*, double }, { i8*, double } } %3
; CHECK-NEXT: }
