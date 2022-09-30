; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

declare dso_local double @__enzyme_autodiff(i8*, double, double*, double*)

define void @subsq(double * writeonly nocapture %out, double %x) {
entry:
  %mul = fmul double %x, %x
  store double %mul, double* %out, align 8
  ret void
}

define double @square(double %x, double* %r) {
entry:
  call void @subsq(double* %r, double %x)
  %ld = load double, double* %r, align 8
  ret double %ld
}

define double @dsquare(double* %r, double* %dr, double %x) local_unnamed_addr {
entry:
  %call = tail call double @__enzyme_autodiff(i8* bitcast (double (double, double*)* @square to i8*), double %x, double* %r, double* %dr)
  ret double %call
}

; CHECK: define internal { double } @diffesquare(double %x, double* %r, double* %"r'", double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = load double, double* %"r'", align 8
; CHECK-NEXT:   %1 = fadd fast double %0, %differeturn
; CHECK-NEXT:   store double %1, double* %"r'", align 8
; CHECK-NEXT:   %2 = call { double } @diffesubsq(double* %r, double* %"r'", double %x)
; CHECK-NEXT:   ret { double } %2
; CHECK-NEXT: }
