; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S -enzyme-zero-cache=0 | FileCheck -check-prefixes CHECK,UNDEF %s
; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S -enzyme-zero-cache=1 | FileCheck -check-prefixes CHECK,ZERO %s

declare dso_local double @__enzyme_autodiff(i8*, double)

define void @subsq(double * writeonly nocapture %out, double %x) {
entry:
  %mul = fmul double %x, %x
  store double %mul, double* %out, align 8
  ret void
}

define double @mid(double %x) {
  %r = alloca double, align 8
  call void @subsq(double* %r, double %x)
  %ld = load double, double* %r, align 8
  ret double %ld
}

define double @square(double %x) {
entry:
  %m = call double @mid(double %x)
  %mul = fmul double %m, %m
  ret double %mul
}

define double @dsquare(double %x) local_unnamed_addr {
entry:
  %call = tail call double @__enzyme_autodiff(i8* bitcast (double (double)* @square to i8*), double %x)
  ret double %call
}

; CHECK: define internal { double } @diffesquare(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %m = call fast double @augmented_mid(double %x)
; CHECK-NEXT:   %m0diffem = fmul fast double %differeturn, %m
; CHECK-NEXT:   %m1diffem = fmul fast double %differeturn, %m
; CHECK-NEXT:   %0 = fadd fast double %m0diffem, %m1diffem
; CHECK-NEXT:   %1 = call { double } @diffemid(double %x, double %0)
; CHECK-NEXT:   ret { double } %1
; CHECK-NEXT: }

; CHECK: define internal void @augmented_subsq(double* nocapture writeonly %out, double* nocapture %"out'", double %x)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %mul = fmul double %x, %x
; CHECK-NEXT:   store double %mul, double* %out, align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal double @augmented_mid(double %x)
; CHECK-NEXT:   %r = alloca double, i64 1, align 8
; UNDEF-NEXT:   call void @augmented_subsq(double* %r, double* undef, double %x)
; ZERO-NEXT:   call void @augmented_subsq(double* %r, double* null, double %x)
; CHECK-NEXT:   %ld = load double, double* %r, align 8
; CHECK-NEXT:   ret double %ld
; CHECK-NEXT: }

; CHECK: define internal { double } @diffemid(double %x, double %differeturn)
; CHECK-NEXT: invert:
; CHECK-NEXT:   %"r'ai" = alloca double, i64 1, align 8
; CHECK-NEXT:   %0 = bitcast double* %"r'ai" to i8*
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull dereferenceable(8) dereferenceable_or_null(8) %0, i8 0, i64 8, i1 false)
; CHECK-NEXT:   %1 = load double, double* %"r'ai", align 8
; CHECK-NEXT:   %2 = fadd fast double %1, %differeturn
; CHECK-NEXT:   store double %2, double* %"r'ai", align 8
; UNDEF-NEXT:   %3 = call { double } @diffesubsq(double* undef, double* %"r'ai", double %x)
; ZERO-NEXT:   %3 = call { double } @diffesubsq(double* null, double* %"r'ai", double %x)
; CHECK-NEXT:   ret { double } %3
; CHECK-NEXT: }

; CHECK: define internal { double } @diffesubsq(double* nocapture writeonly %out, double* nocapture %"out'", double %x)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = load double, double* %"out'", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"out'", align 8
; CHECK-NEXT:   %m0diffex = fmul fast double %0, %x
; CHECK-NEXT:   %m1diffex = fmul fast double %0, %x
; CHECK-NEXT:   %1 = fadd fast double %m0diffex, %m1diffex
; CHECK-NEXT:   %2 = insertvalue { double } undef, double %1, 0
; CHECK-NEXT:   ret { double } %2
; CHECK-NEXT: }
