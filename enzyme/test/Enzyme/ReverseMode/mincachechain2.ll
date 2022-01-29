; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -simplifycfg -S | FileCheck %s
; ModuleID = 'inp.ll'

declare dso_local void @_Z17__enzyme_autodiffPvPdS0_i(i8*, double*, double*, i64*) local_unnamed_addr #4
define dso_local void @outer(double* %m, double* %m2, i64* %n) local_unnamed_addr #2 {
entry:
  call void @_Z17__enzyme_autodiffPvPdS0_i(i8* bitcast (double (double*, i64*)* @_Z10reduce_maxPdi to i8*), double* nonnull %m, double* nonnull %m2, i64* %n)
  ret void
}
; Function Attrs: nounwind uwtable
define dso_local double @_Z10reduce_maxPdi(double* %vec, i64* %v) #0 {
entry:
  %res = call double @pb(double* %vec, i64* %v)
  store i64 0, i64* %v, align 8
  ret double %res
}

define double @pb(double* %__x, i64* %v) {
entry:
  %n = load i64, i64* %v, align 8
  %ptr = getelementptr inbounds double, double* %__x, i64 %n
  %ptr2 = getelementptr inbounds double, double* %ptr, i64 1
  %ld = call double @loader(double* %ptr2)
  ret double %ld
}

define double @loader(double* %ptr) {
entry:
  %ld = load double, double* %ptr, align 8
  ret double %ld
}


; CHECK: define internal void @diffe_Z10reduce_maxPdi(double* %vec, double* %"vec'", i64* %v, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %res_augmented = call i64 @augmented_pb(double* %vec, double* %"vec'", i64* %v)
; CHECK-NEXT:   store i64 0, i64* %v, align 8
; CHECK-NEXT:   call void @diffepb(double* %vec, double* %"vec'", i64* %v, double %differeturn, i64 %res_augmented)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @augmented_loader(double* %ptr, double* %"ptr'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal i64 @augmented_pb(double* %__x, double* %"__x'", i64* %v)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %n = load i64, i64* %v, align 8
; CHECK-NEXT:   %"ptr'ipg" = getelementptr inbounds double, double* %"__x'", i64 %n
; CHECK-NEXT:   %ptr = getelementptr inbounds double, double* %__x, i64 %n
; CHECK-NEXT:   %"ptr2'ipg" = getelementptr inbounds double, double* %"ptr'ipg", i64 1
; CHECK-NEXT:   %ptr2 = getelementptr inbounds double, double* %ptr, i64 1
; CHECK-NEXT:   call void @augmented_loader(double* %ptr2, double* %"ptr2'ipg")
; CHECK-NEXT:   ret i64 %n
; CHECK-NEXT: }

; CHECK: define internal void @diffepb(double* %__x, double* %"__x'", i64* %v, double %differeturn, i64 %n)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"ptr'ipg" = getelementptr inbounds double, double* %"__x'", i64 %n
; CHECK-NEXT:   %ptr = getelementptr inbounds double, double* %__x, i64 %n
; CHECK-NEXT:   %"ptr2'ipg" = getelementptr inbounds double, double* %"ptr'ipg", i64 1
; CHECK-NEXT:   %ptr2 = getelementptr inbounds double, double* %ptr, i64 1
; CHECK-NEXT:   call void @diffeloader(double* %ptr2, double* %"ptr2'ipg", double %differeturn)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @diffeloader(double* %ptr, double* %"ptr'", double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = load double, double* %"ptr'", align 8
; CHECK-NEXT:   %1 = fadd fast double %0, %differeturn
; CHECK-NEXT:   store double %1, double* %"ptr'", align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
