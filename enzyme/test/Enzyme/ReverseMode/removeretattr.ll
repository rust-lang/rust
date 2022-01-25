; RUN: if [ %llvmver -ge 11 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -simplifycfg -instcombine -adce -S | FileCheck %s; fi

; this test should ensure that the alignment on the <2 x double> load is kept

define void @caller(double* %in_W, double* %in_Wp) {
entry:
  call void @__enzyme_autodiff(i8* bitcast (void (double*)* @matvec to i8*), double* nonnull %in_W, double* nonnull %in_Wp) #8
  ret void
}

declare void @__enzyme_autodiff(i8*, double*, double*)

define noalias noundef nonnull align 8 double* @cst(double* noalias %W) {
entry:
  ret double* %W
}

define internal void @matvec(double* noalias %W) {
entry:
  %ptr = call double* @cst(double* %W)
  %ld = load double, double* %ptr, align 8
  %mul = fmul double %ld, %ld
  store double %mul, double* %W
  ret void
}

; CHECK: define internal { double*, double* } @augmented_cst(double* noalias %W, double* %"W'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[fca0insert:.+]] = insertvalue { double*, double* } undef, double* %W, 0
; CHECK-NEXT:   %[[fca1insert:.+]] = insertvalue { double*, double* } %[[fca0insert:.+]], double* %"W'", 1
; CHECK-NEXT:   ret { double*, double* } %[[fca1insert:.+]]
; CHECK-NEXT: }

; CHECK: define internal void @diffecst(double* noalias %W, double* %"W'") 
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
