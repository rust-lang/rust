; RUN: if [ %llvmver -ge 13 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi

; Function Attrs: noinline nounwind readnone uwtable
define double @tester(double %x, i32 %y) {
entry:
  %0 = tail call fast double @llvm.powi.f64.i32(double %x, i32 %y)
  ret double %0
}

define double @test_derivative(double %x, i32 %y) {
entry:
  %0 = tail call double (double (double, i32)*, ...) @__enzyme_fwddiff(double (double, i32)* nonnull @tester, double %x, double 1.0, i32 %y)
  ret double %0
}

; Function Attrs: nounwind readnone speculatable
declare double @llvm.powi.f64.i32(double, i32)

; Function Attrs: nounwind
declare double @__enzyme_fwddiff(double (double, i32)*, ...)

; CHECK: define internal {{(dso_local )?}}double @fwddiffetester(double %x, double %"x'", i32 %y)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[ym1:.+]] = sub i32 %y, 1
; CHECK-NEXT:   %[[newpow:.+]] = call fast double @llvm.powi.f64.i32(double %x, i32 %[[ym1]])
; CHECK-DAG:    %[[sitofp:.+]] = sitofp i32 %y to double
; CHECK-DAG:    %[[newpowdret:.+]] = fmul fast double %"x'", %[[newpow]]
; CHECK-NEXT:   %[[dx:.+]] = fmul fast double %[[newpowdret]], %[[sitofp]]
; CHECK-NEXT:   ret double %[[dx:.+]]
; CHECK-NEXT: }

