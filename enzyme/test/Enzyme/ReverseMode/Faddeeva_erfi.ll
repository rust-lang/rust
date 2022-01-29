; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -instsimplify -simplifycfg -adce -S | FileCheck %s

declare { double, double } @Faddeeva_erfi({ double, double }, double)

define { double, double } @tester({ double, double } %in) {
entry:
  %call = call { double, double } @Faddeeva_erfi({ double, double } %in, double 0.000000e+00)
  ret { double, double } %call
}

define { double, double } @test_derivative({ double, double } %x) {
entry:
  %0 = tail call { double, double } ({ double, double } ({ double, double })*, ...) @__enzyme_autodiff({ double, double } ({ double, double })* nonnull @tester, metadata !"enzyme_out", { double, double } %x)
  ret { double, double } %0
}

; Function Attrs: nounwind
declare { double, double } @__enzyme_autodiff({ double, double } ({ double, double })*, ...)

; CHECK: define internal { { double, double } } @diffetester({ double, double } %in, { double, double } %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = extractvalue { double, double } %in, 0
; CHECK-NEXT:   %1 = extractvalue { double, double } %in, 1
; CHECK-DAG:    %[[a2:.+]] = fmul fast double %1, %1
; CHECK-DAG:    %[[a3:.+]] = fmul fast double %0, %0
; CHECK-NEXT:   %4 = fsub fast double %[[a3]], %[[a2]]
; CHECK-NEXT:   %5 = fmul fast double %0, %1
; CHECK-NEXT:   %6 = fadd fast double %5, %5
; CHECK-NEXT:   %[[i9:.+]] = call fast double @llvm.exp.f64(double %4)
; CHECK-NEXT:   %[[i10:.+]] = call fast double @llvm.cos.f64(double %6)
; CHECK-NEXT:   %[[i11:.+]] = fmul fast double %[[i9]], %[[i10]]
; CHECK-NEXT:   %[[i12:.+]] = call fast double @llvm.sin.f64(double %6)
; CHECK-NEXT:   %[[i13:.+]] = fmul fast double %[[i9]], %[[i12]]
; CHECK-NEXT:   %[[i14:.+]] = fmul fast double %[[i11]], 0x3FF20DD750429B6D
; CHECK-NEXT:   %[[i15:.+]] = fmul fast double %[[i13]], 0x3FF20DD750429B6D
; CHECK-NEXT:   %[[i16:.+]] = extractvalue { double, double } %differeturn, 0
; CHECK-NEXT:   %[[i17:.+]] = extractvalue { double, double } %differeturn, 1
; CHECK-DAG:   %[[i18:.+]] = fmul fast double %[[i15]], %[[i17]]
; CHECK-DAG:   %[[i19:.+]] = fmul fast double %[[i14]], %[[i16]]
; CHECK-NEXT:   %[[i20:.+]] = fsub fast double %[[i19]], %[[i18]]
; CHECK-DAG:   %[[i21:.+]] = fmul fast double %[[i14]], %[[i17]]
; CHECK-DAG:   %[[i22:.+]] = fmul fast double %[[i15]], %[[i16]]
; CHECK-NEXT:   %[[i23:.+]] = fadd fast double %[[i22]], %[[i21]]
; CHECK-NEXT:   %.fca.0.insert5 = insertvalue { double, double } undef, double %[[i20]], 0
; CHECK-NEXT:   %.fca.1.insert8 = insertvalue { double, double } %.fca.0.insert5, double %[[i23]], 1
; CHECK-NEXT:   %[[i24:.+]] = insertvalue { { double, double } } undef, { double, double } %.fca.1.insert8, 0
; CHECK-NEXT:   ret { { double, double } } %[[i24]]
; CHECK-NEXT: }
