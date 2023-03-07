; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -early-cse -instsimplify -S | FileCheck %s
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(early-cse,instsimplify)" -enzyme-preopt=false -S | FileCheck %s

declare { double, double } @Faddeeva_erfc({ double, double }, double)

define { double, double } @tester({ double, double } %in) {
entry:
  %call = call { double, double } @Faddeeva_erfc({ double, double } %in, double 0.000000e+00)
  ret { double, double } %call
}

define { double, double } @test_derivative({ double, double } %x) {
entry:
  %0 = tail call { double, double } ({ double, double } ({ double, double })*, ...) @__enzyme_fwddiff({ double, double } ({ double, double })* @tester, { double, double } %x, { double, double } { double 1.0, double 1.0 })
  ret { double, double } %0
}

; Function Attrs: nounwind
declare { double, double } @__enzyme_fwddiff({ double, double } ({ double, double })*, ...)


; CHECK: define internal { double, double } @fwddiffetester({ double, double } %in, { double, double } %"in'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = extractvalue { double, double } %in, 0
; CHECK-NEXT:   %1 = extractvalue { double, double } %in, 1
; CHECK-DAG:    %[[a2:.+]] = fmul fast double %0, %0
; CHECK-DAG:    %[[a3:.+]] = fmul fast double %1, %1
; CHECK-NEXT:   %4 = fsub fast double %[[a2]], %[[a3]]
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
; CHECK-NEXT:   %15 = insertvalue { double, double } undef, double %14, 0
; CHECK-NEXT:   %16 = fmul fast double %13, 0xBFF20DD750429B6D
; CHECK-NEXT:   %17 = insertvalue { double, double } %15, double %16, 1
; CHECK-NEXT:   %18 = extractvalue { double, double } %"in'", 0
; CHECK-NEXT:   %19 = extractvalue { double, double } %"in'", 1
; CHECK-DAG:    %[[a20:.+]] = fmul fast double %14, %18
; CHECK-DAG:    %[[a21:.+]] = fmul fast double %16, %19
; CHECK-NEXT:   %22 = fsub fast double %[[a20]], %[[a21]]
; CHECK-NEXT:   %23 = insertvalue { double, double } %17, double %22, 0
; CHECK-DAG:    %[[a24:.+]] = fmul fast double %16, %18
; CHECK-DAG:    %[[a25:.+]] = fmul fast double %14, %19
; CHECK-NEXT:   %26 = fadd fast double %[[a24]], %[[a25]]
; CHECK-NEXT:   %27 = insertvalue { double, double } %23, double %26, 1
; CHECK-NEXT:   ret { double, double } %27
; CHECK-NEXT: }
