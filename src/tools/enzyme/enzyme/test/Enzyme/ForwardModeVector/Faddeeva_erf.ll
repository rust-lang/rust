; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -instsimplify -simplifycfg -adce -S | FileCheck %s

%struct.Gradients = type { { double, double }, { double, double }, { double, double } }

declare %struct.Gradients @__enzyme_fwddiff({ double, double } ({ double, double })*, ...)

declare { double, double } @Faddeeva_erf({ double, double }, double)

define { double, double } @tester({ double, double } %in) {
entry:
  %call = call { double, double } @Faddeeva_erf({ double, double } %in, double 0.000000e+00)
  ret { double, double } %call
}

define %struct.Gradients @test_derivative({ double, double } %x) {
entry:
  %0 = tail call %struct.Gradients ({ double, double } ({ double, double })*, ...) @__enzyme_fwddiff({ double, double } ({ double, double })*  @tester,  metadata !"enzyme_width", i64 3, { double, double } %x, { double, double } { double 1.0, double 1.0 }, { double, double } { double 1.0, double 1.0 }, { double, double } { double 1.0, double 1.0 })
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
; CHECK-NEXT:   %7 = {{(fsub fast double \-0.000000e\+00,|fneg fast double)}} %4
; CHECK-NEXT:   %8 = {{(fsub fast double \-0.000000e\+00,|fneg fast double)}} %6
; CHECK-NEXT:   %9 = call fast double @llvm.exp.f64(double %7)
; CHECK-NEXT:   %10 = call fast double @llvm.cos.f64(double %8)
; CHECK-NEXT:   %11 = fmul fast double %9, %10
; CHECK-NEXT:   %12 = call fast double @llvm.sin.f64(double %8)
; CHECK-NEXT:   %13 = fmul fast double %9, %12
; CHECK-NEXT:   %14 = fmul fast double %11, 0x3FF20DD750429B6D
; CHECK-NEXT:   %15 = insertvalue { double, double } undef, double %14, 0
; CHECK-NEXT:   %16 = fmul fast double %13, 0x3FF20DD750429B6D
; CHECK-NEXT:   %17 = insertvalue { double, double } %15, double %16, 1
; CHECK-NEXT:   %18 = extractvalue [3 x { double, double }] %"in'", 0
; CHECK-NEXT:   %19 = extractvalue { double, double } %18, 0
; CHECK-NEXT:   %20 = extractvalue { double, double } %18, 1
; CHECK-DAG:    %[[a21:.+]] = fmul fast double %16, %20
; CHECK-DAG:    %[[a22:.+]] = fmul fast double %14, %19
; CHECK-NEXT:   %23 = fsub fast double %[[a22]], %[[a21]]
; CHECK-NEXT:   %24 = insertvalue { double, double } %17, double %23, 0
; CHECK-DAG:    %[[a25:.+]] = fmul fast double %14, %20
; CHECK-DAG:    %[[a26:.+]] = fmul fast double %16, %19
; CHECK-NEXT:   %27 = fadd fast double %[[a26]], %[[a25]]
; CHECK-NEXT:   %28 = insertvalue { double, double } %24, double %27, 1
; CHECK-NEXT:   %29 = insertvalue [3 x { double, double }] undef, { double, double } %28, 0
; CHECK-NEXT:   %30 = extractvalue [3 x { double, double }] %"in'", 1
; CHECK-NEXT:   %31 = extractvalue { double, double } %30, 0
; CHECK-NEXT:   %32 = extractvalue { double, double } %30, 1
; CHECK-DAG:    %[[a33:.+]] = fmul fast double %16, %32
; CHECK-DAG:    %[[a34:.+]] = fmul fast double %14, %31
; CHECK-NEXT:   %35 = fsub fast double %[[a34]], %[[a33]]
; CHECK-NEXT:   %36 = insertvalue { double, double } %17, double %35, 0
; CHECK-DAG:    %[[a37:.+]] = fmul fast double %14, %32
; CHECK-DAG:    %[[a38:.+]] = fmul fast double %16, %31
; CHECK-NEXT:   %39 = fadd fast double %[[a38]], %[[a37]]
; CHECK-NEXT:   %40 = insertvalue { double, double } %36, double %39, 1
; CHECK-NEXT:   %41 = insertvalue [3 x { double, double }] %29, { double, double } %40, 1
; CHECK-NEXT:   %42 = extractvalue [3 x { double, double }] %"in'", 2
; CHECK-NEXT:   %43 = extractvalue { double, double } %42, 0
; CHECK-NEXT:   %44 = extractvalue { double, double } %42, 1
; CHECK-DAG:    %[[a45:.+]] = fmul fast double %16, %44
; CHECK-DAG:    %[[a46:.+]] = fmul fast double %14, %43
; CHECK-NEXT:   %47 = fsub fast double %[[a46]], %[[a45]]
; CHECK-NEXT:   %48 = insertvalue { double, double } %17, double %47, 0
; CHECK-DAG:    %[[a49:.+]] = fmul fast double %14, %44
; CHECK-DAG:    %[[a50:.+]] = fmul fast double %16, %43
; CHECK-NEXT:   %51 = fadd fast double %[[a50]], %[[a49]]
; CHECK-NEXT:   %52 = insertvalue { double, double } %48, double %51, 1
; CHECK-NEXT:   %53 = insertvalue [3 x { double, double }] %41, { double, double } %52, 2
; CHECK-NEXT:   ret [3 x { double, double }] %53
; CHECK-NEXT: }