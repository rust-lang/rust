; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -instsimplify -simplifycfg -adce -S | FileCheck %s

declare { double, double } @Faddeeva_erfc({ double, double }, double)

define { double, double } @tester({ double, double } %in) {
entry:
  %call = call { double, double } @Faddeeva_erfc({ double, double } %in, double 0.000000e+00)
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
; CHECK-NEXT:   %7 = {{(fsub fast double \-0.000000e\+00,|fneg fast double)}} %4
; CHECK-NEXT:   %8 = {{(fsub fast double \-0.000000e\+00,|fneg fast double)}} %6
; CHECK-NEXT:   %9 = call fast double @llvm.exp.f64(double %7)
; CHECK-NEXT:   %10 = call fast double @llvm.cos.f64(double %8)
; CHECK-NEXT:   %11 = fmul fast double %9, %10
; CHECK-NEXT:   %12 = call fast double @llvm.sin.f64(double %8)
; CHECK-NEXT:   %13 = fmul fast double %9, %12
; CHECK-NEXT:   %14 = fmul fast double %11, 0xBFF20DD750429B6D
; CHECK-NEXT:   %15 = fmul fast double %13, 0xBFF20DD750429B6D
; CHECK-NEXT:   %16 = extractvalue { double, double } %differeturn, 0
; CHECK-NEXT:   %17 = extractvalue { double, double } %differeturn, 1
; CHECK-DAG:    %[[a18:.+]] = fmul fast double %15, %17
; CHECK-DAG:    %[[a19:.+]] = fmul fast double %14, %16
; CHECK-NEXT:   %20 = fsub fast double %[[a19]], %[[a18]]
; CHECK-DAG:    %[[a21:.+]] = fmul fast double %14, %17
; CHECK-DAG:    %[[a22:.+]] = fmul fast double %15, %16
; CHECK-NEXT:   %23 = fadd fast double %[[a22]], %[[a21]]
; CHECK-NEXT:   %.fca.0.insert5 = insertvalue { double, double } undef, double %20, 0
; CHECK-NEXT:   %.fca.1.insert8 = insertvalue { double, double } %.fca.0.insert5, double %23, 1
; CHECK-NEXT:   %24 = insertvalue { { double, double } } undef, { double, double } %.fca.1.insert8, 0
; CHECK-NEXT:   ret { { double, double } } %24
; CHECK-NEXT: }
