; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -instsimplify -simplifycfg -adce -S | FileCheck %s

declare { double, double } @Faddeeva_erfi({ double, double }, double)

define { double, double } @tester({ double, double } %in) {
entry:
  %call = call { double, double } @Faddeeva_erfi({ double, double } %in, double 0.000000e+00)
  ret { double, double } %call
}

define { double, double } @test_derivative({ double, double } %x) {
entry:
  %0 = tail call { double, double } ({ double, double } ({ double, double })*, ...) @__enzyme_fwdsplit({ double, double } ({ double, double })* nonnull @tester, { double, double } %x, { double, double } { double 1.0, double 1.0 }, i8* null)
  ret { double, double } %0
}

; Function Attrs: nounwind
declare { double, double } @__enzyme_fwdsplit({ double, double } ({ double, double })*, ...)


; CHECK: define internal { double, double } @fwddiffetester({ double, double } %in, { double, double } %"in'", i8* %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   tail call void @free(i8* nonnull %tapeArg)
; CHECK-NEXT:   %0 = extractvalue { double, double } %in, 0
; CHECK-NEXT:   %1 = extractvalue { double, double } %in, 1
; CHECK-DAG:    %[[a2:.+]] = fmul fast double %0, %0
; CHECK-DAG:    %[[a3:.+]] = fmul fast double %1, %1
; CHECK-NEXT:   %4 = fsub fast double %[[a2]], %[[a3]]
; CHECK-DAG:    %[[a5:.+]] = fmul fast double %0, {{(%1|%5)}}
; CHECK-DAG:    %[[a6:.+]] = fadd fast double {{(%5|%1)}}, {{(%5|%1)}}
; CHECK-NEXT:   %7 = call fast double @llvm.exp.f64(double %4)
; CHECK-NEXT:   %8 = call fast double @llvm.cos.f64(double %6)
; CHECK-NEXT:   %9 = fmul fast double %7, %8
; CHECK-NEXT:   %10 = call fast double @llvm.sin.f64(double %6)
; CHECK-NEXT:   %11 = fmul fast double %7, %10
; CHECK-NEXT:   %12 = fmul fast double %9, 0x3FF20DD750429B6D
; CHECK-NEXT:   %13 = insertvalue { double, double } undef, double %12, 0
; CHECK-NEXT:   %14 = fmul fast double %11, 0x3FF20DD750429B6D
; CHECK-NEXT:   %15 = insertvalue { double, double } %13, double %14, 1
; CHECK-NEXT:   %16 = extractvalue { double, double } %"in'", 0
; CHECK-NEXT:   %17 = extractvalue { double, double } %"in'", 1
; CHECK-NEXT:   %18 = fmul fast double %14, %17
; CHECK-NEXT:   %19 = fmul fast double %12, %16
; CHECK-NEXT:   %20 = fsub fast double %19, %18
; CHECK-NEXT:   %21 = insertvalue { double, double } %15, double %20, 0
; CHECK-NEXT:   %22 = fmul fast double %12, %17
; CHECK-NEXT:   %23 = fmul fast double %14, %16
; CHECK-NEXT:   %24 = fadd fast double %23, %22
; CHECK-NEXT:   %25 = insertvalue { double, double } %21, double %24, 1
; CHECK-NEXT:   ret { double, double } %25
; CHECK-NEXT: }
