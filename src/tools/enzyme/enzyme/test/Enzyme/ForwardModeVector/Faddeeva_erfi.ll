; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -instsimplify -simplifycfg -adce -S | FileCheck %s

%struct.Gradients = type { { double, double }, { double, double }, { double, double } }

declare %struct.Gradients @__enzyme_fwddiff({ double, double } ({ double, double })*, ...)

declare { double, double } @Faddeeva_erfi({ double, double }, double)

define { double, double } @tester({ double, double } %in) {
entry:
  %call = call { double, double } @Faddeeva_erfi({ double, double } %in, double 0.000000e+00)
  ret { double, double } %call
}

define %struct.Gradients @test_derivative({ double, double } %x) {
entry:
  %0 = tail call %struct.Gradients ({ double, double } ({ double, double })*, ...) @__enzyme_fwddiff({ double, double } ({ double, double })* nonnull @tester,  metadata !"enzyme_width", i64 3, { double, double } %x, { double, double } { double 1.0, double 1.0 }, { double, double } { double 1.0, double 1.0 }, { double, double } { double 1.0, double 1.0 })
  ret %struct.Gradients %0
}


; CHECK: define internal [3 x { double, double }] @fwddiffe3tester({ double, double } %in, [3 x { double, double }] %"in'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = extractvalue { double, double } %in, 0
; CHECK-NEXT:   %1 = extractvalue { double, double } %in, 1
; CHECK-DAG:    %[[a2:.+]] = fmul fast double %1, %1
; CHECK-DAG:    %[[a3:.+]] = fmul fast double %0, %0
; CHECK-NEXT:   %4 = fsub fast double %[[a3]], %[[a2]]
; CHECK-NEXT:   %5 = fmul fast double %0, %1
; CHECK-NEXT:   %6 = fadd fast double %5, %5
; CHECK-NEXT:   %7 = call fast double @llvm.exp.f64(double %4)
; CHECK-NEXT:   %8 = call fast double @llvm.cos.f64(double %6)
; CHECK-NEXT:   %9 = fmul fast double %7, %8
; CHECK-NEXT:   %10 = call fast double @llvm.sin.f64(double %6)
; CHECK-NEXT:   %11 = fmul fast double %7, %10
; CHECK-NEXT:   %12 = fmul fast double %9, 0x3FF20DD750429B6D
; CHECK-NEXT:   %13 = insertvalue { double, double } undef, double %12, 0
; CHECK-NEXT:   %14 = fmul fast double %11, 0x3FF20DD750429B6D
; CHECK-NEXT:   %15 = insertvalue { double, double } %13, double %14, 1
; CHECK-NEXT:   %16 = extractvalue [3 x { double, double }] %"in'", 0
; CHECK-NEXT:   %17 = extractvalue { double, double } %16, 0
; CHECK-NEXT:   %18 = extractvalue { double, double } %16, 1
; CHECK-DAG:    %[[a19:.+]] = fmul fast double %14, %18
; CHECK-DAG:    %[[a20:.+]] = fmul fast double %12, %17
; CHECK-NEXT:   %21 = fsub fast double %[[a20]], %[[a19]]
; CHECK-NEXT:   %22 = insertvalue { double, double } %15, double %21, 0
; CHECK-DAG:    %[[a23:.+]] = fmul fast double %12, %18
; CHECK-DAG:    %[[a24:.+]] = fmul fast double %14, %17
; CHECK-NEXT:   %25 = fadd fast double %[[a24]], %[[a23]]
; CHECK-NEXT:   %26 = insertvalue { double, double } %22, double %25, 1
; CHECK-NEXT:   %27 = insertvalue [3 x { double, double }] undef, { double, double } %26, 0
; CHECK-NEXT:   %28 = extractvalue [3 x { double, double }] %"in'", 1
; CHECK-NEXT:   %29 = extractvalue { double, double } %28, 0
; CHECK-NEXT:   %30 = extractvalue { double, double } %28, 1
; CHECK-DAG:    %[[a31:.+]] = fmul fast double %14, %30
; CHECK-DAG:    %[[a32:.+]] = fmul fast double %12, %29
; CHECK-NEXT:   %33 = fsub fast double %[[a32]], %[[a31]]
; CHECK-NEXT:   %34 = insertvalue { double, double } %15, double %33, 0
; CHECK-DAG:    %[[a35:.+]] = fmul fast double %12, %30
; CHECK-DAG:    %[[a36:.+]] = fmul fast double %14, %29
; CHECK-NEXT:   %37 = fadd fast double %[[a36]], %[[a35]]
; CHECK-NEXT:   %38 = insertvalue { double, double } %34, double %37, 1
; CHECK-NEXT:   %39 = insertvalue [3 x { double, double }] %27, { double, double } %38, 1
; CHECK-NEXT:   %40 = extractvalue [3 x { double, double }] %"in'", 2
; CHECK-NEXT:   %41 = extractvalue { double, double } %40, 0
; CHECK-NEXT:   %42 = extractvalue { double, double } %40, 1
; CHECK-DAG:    %[[a43:.+]] = fmul fast double %14, %42
; CHECK-DAG:    %[[a44:.+]] = fmul fast double %12, %41
; CHECK-NEXT:   %45 = fsub fast double %[[a44]], %[[a43]]
; CHECK-NEXT:   %46 = insertvalue { double, double } %15, double %45, 0
; CHECK-DAG:    %[[a47:.+]] = fmul fast double %12, %42
; CHECK-DAG:    %[[a48:.+]] = fmul fast double %14, %41
; CHECK-NEXT:   %49 = fadd fast double %[[a48]], %[[a47]]
; CHECK-NEXT:   %50 = insertvalue { double, double } %46, double %49, 1
; CHECK-NEXT:   %51 = insertvalue [3 x { double, double }] %39, { double, double } %50, 2
; CHECK-NEXT:   ret [3 x { double, double }] %51
; CHECK-NEXT: }