; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -adce -S | FileCheck %s
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(adce)" -enzyme-preopt=false -S | FileCheck %s

declare void @_Z16__enzyme_fwddiff(...)

define void @_Z34testFwdDerivativesRosenbrockEnzymev(i8* %a, i8* %b) {
  call void (...) @_Z16__enzyme_fwddiff(double (i64*)* @f, metadata !"enzyme_dup", i8* %a, i8* %b)
  ret void
}

define double @f(i64* %i10) {
bb:
  %i13 = load i64, i64* %i10, align 8
  %i14 = sub i64 2, %i13
  %i15 = sdiv exact i64 %i14, 8
  %a5 = uitofp i64 %i15 to double
  ret double %a5
}

; CHECK: define internal double @fwddiffef(i64* %i10, i64* %"i10'")
; CHECK-NEXT: bb:
; CHECK-NEXT:   ret double 0.000000e+00
; CHECK-NEXT: }
