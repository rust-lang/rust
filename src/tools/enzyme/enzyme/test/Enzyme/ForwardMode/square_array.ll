; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -early-cse -S | FileCheck %s
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(early-cse)" -enzyme-preopt=false -S | FileCheck %s

define { double, double } @squared(double %x) {
entry:
  %mul = fmul double %x, %x
  %mul2 = fmul double %mul, %x
  %.fca.0.insert = insertvalue { double, double } undef, double %mul, 0
  %.fca.1.insert = insertvalue { double, double } %.fca.0.insert, double %mul2, 1
  ret { double, double } %.fca.1.insert
}

define { double, double } @dsquared(double %x) {
entry:
  %call = call { double, double } (i8*, ...) @__enzyme_fwddiff(i8* bitcast ({ double, double } (double)* @squared to i8*), double %x, double 1.0)
  ret { double, double } %call
}

declare { double, double } @__enzyme_fwddiff(i8*, ...)



; CHECK: define internal {{(dso_local )?}}{ double, double } @fwddiffesquared(double %x, double %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %mul = fmul double %x, %x
; CHECK-NEXT:   %[[i0:.+]] = fmul fast double %"x'", %x
; CHECK-NEXT:   %[[i1:.+]] = fadd fast double %[[i0]], %[[i0]]
; CHECK-NEXT:   %[[i2:.+]] = fmul fast double %[[i1]], %x
; CHECK-NEXT:   %[[i3:.+]] = fmul fast double %"x'", %mul
; CHECK-NEXT:   %[[i4:.+]] = fadd fast double %[[i2]], %[[i3]]
; CHECK-NEXT:   %[[i5:.+]] = insertvalue { double, double } zeroinitializer, double %[[i1]], 0
; CHECK-NEXT:   %[[i6:.+]] = insertvalue { double, double } %[[i5]], double %[[i4]], 1
; CHECK-NEXT:   ret { double, double } %[[i6]]
; CHECK-NEXT: }
