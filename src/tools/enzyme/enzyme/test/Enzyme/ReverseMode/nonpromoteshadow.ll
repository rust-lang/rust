; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

declare i8* @malloc(i64)

define void @set(double** writeonly nocapture %p) {
entry:
  %m = call i8* @malloc(i64 8)
  %ptr = bitcast i8* %m to double*
  store double* %ptr, double** %p, align 8
  ret void
}

define double @square(double %x) {
entry:
  %a = alloca double*, align 8
  call void @set(double** %a)
  %m = load double*, double** %a, align 8
  store double %x, double* %m, align 8
  %ld = load double, double* %m, align 8
  %mul = fmul double %ld, %ld
  ret double %mul
}

declare dso_local i8* @__enzyme_virtualreverse(i8*)

define i8* @dsquare(double %x) local_unnamed_addr {
entry:
  %call = tail call i8* @__enzyme_virtualreverse(i8* bitcast (double (double)* @square to i8*))
  ret i8* %call
}

; CHECK: define internal { double } @diffesquare(double %x, double %differeturn, i8* %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = bitcast i8* %tapeArg to { { i8*, i8* }, i8*, double }*
; CHECK-NEXT:   %truetape = load { { i8*, i8* }, i8*, double }, { { i8*, i8* }, i8*, double }* %0
; CHECK-NEXT:   tail call void @free(i8* nonnull %tapeArg)
; CHECK-NEXT:   %"malloccall'mi" = extractvalue { { i8*, i8* }, i8*, double } %truetape, 1
; CHECK-NEXT:   %"a'ipc" = bitcast i8* %"malloccall'mi" to double**
; CHECK-NEXT:   %tapeArg1 = extractvalue { { i8*, i8* }, i8*, double } %truetape, 0
; CHECK-NEXT:   %"m'ipl" = load double*, double** %"a'ipc", align 8
; CHECK-NEXT:   %ld = extractvalue { { i8*, i8* }, i8*, double } %truetape, 2
; CHECK-NEXT:   %m0diffeld = fmul fast double %differeturn, %ld
; CHECK-NEXT:   %m1diffeld = fmul fast double %differeturn, %ld
; CHECK-NEXT:   %1 = fadd fast double %m0diffeld, %m1diffeld
; CHECK-NEXT:   %2 = load double, double* %"m'ipl", align 8
; CHECK-NEXT:   %3 = fadd fast double %2, %1
; CHECK-NEXT:   store double %3, double* %"m'ipl", align 8
; CHECK-NEXT:   %4 = load double, double* %"m'ipl", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"m'ipl", align 8
; CHECK-NEXT:   call void @diffeset(double** undef, double** undef, { i8*, i8* } %tapeArg1)
; CHECK-NEXT:   tail call void @free(i8* nonnull %"malloccall'mi")
; CHECK-NEXT:   %5 = insertvalue { double } undef, double %4, 0
; CHECK-NEXT:   ret { double } %5
; CHECK-NEXT: }

