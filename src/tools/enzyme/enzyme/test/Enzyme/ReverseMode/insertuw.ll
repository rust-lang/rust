; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -S | FileCheck %s

define void @tester(double* %in0, double* %in1, i1 %c) {
entry:
  br i1 %c, label %trueb, label %exit

trueb:
  %pre_x0 = load double, double* %in0
  store double 0.000000e+00, double* %in0
  %x0 = insertvalue {double, double, double*} undef, double %pre_x0, 0

  %pre_x1 = load double, double* %in1
  store double 0.000000e+00, double* %in1
  %x1 = insertvalue {double, double, double*} %x0, double %pre_x1, 1

  %out1 = insertvalue {double, double, double*} %x1, double* %in0, 2
  
  %post_x0 = extractvalue {double, double, double*} %out1, 0
  %post_x1 = extractvalue {double, double, double*} %x1, 1
  
  %mul0 = fmul double %post_x0, %post_x1
  store double %mul0, double* %in0   
  
  br label %exit

exit:
  ret void
}

define void @test_derivative(double* %x, double* %dx, double* %y, double* %dy) {
entry:
  tail call void (...) @__enzyme_autodiff(void (double*, double*, i1)* nonnull @tester, double* %x, double* %dx, double* %y, double* %dy, i1 true)
  ret void
}

; Function Attrs: nounwind
declare void @__enzyme_autodiff(...)

; CHECK: define internal void @diffetester(double* %in0, double* %"in0'", double* %in1, double* %"in1'", i1 %c)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"x1'de" = alloca { double, double, double* }
; CHECK-NEXT:   store { double, double, double* } zeroinitializer, { double, double, double* }* %"x1'de"
; CHECK-NEXT:   %"out1'de" = alloca { double, double, double* }
; CHECK-NEXT:   store { double, double, double* } zeroinitializer, { double, double, double* }* %"out1'de"
; CHECK-NEXT:   %"x0'de" = alloca { double, double, double* }
; CHECK-NEXT:   store { double, double, double* } zeroinitializer, { double, double, double* }* %"x0'de"
; CHECK-NEXT:   br i1 %c, label %trueb, label %exit

; CHECK: trueb:                                            ; preds = %entry
; CHECK-NEXT:   %pre_x0 = load double, double* %in0
; CHECK-NEXT:   store double 0.000000e+00, double* %in0
; CHECK-NEXT:   %x0 = insertvalue { double, double, double* } undef, double %pre_x0, 0
; CHECK-NEXT:   %pre_x1 = load double, double* %in1
; CHECK-NEXT:   store double 0.000000e+00, double* %in1
; CHECK-NEXT:   %x1 = insertvalue { double, double, double* } %x0, double %pre_x1, 1
; CHECK-NEXT:   %out1 = insertvalue { double, double, double* } %x1, double* %in0, 2
; CHECK-NEXT:   %post_x0 = extractvalue { double, double, double* } %out1, 0
; CHECK-NEXT:   %post_x1 = extractvalue { double, double, double* } %x1, 1
; CHECK-NEXT:   %mul0 = fmul double %post_x0, %post_x1
; CHECK-NEXT:   store double %mul0, double* %in0
; CHECK-NEXT:   br label %exit

; CHECK: exit:                                             ; preds = %trueb, %entry
; CHECK-NEXT:   %x1_cache.0 = phi { double, double, double* } [ %x1, %trueb ], [ undef, %entry ]
; CHECK-NEXT:   br label %invertexit

; CHECK: invertentry:                                      ; preds = %invertexit, %inverttrueb
; CHECK-NEXT:   ret void

; CHECK: inverttrueb:                                      ; preds = %invertexit
; CHECK-NEXT:   %0 = load double, double* %"in0'"
; CHECK-NEXT:   store double 0.000000e+00, double* %"in0'"
; CHECK-NEXT:   %1 = fadd fast double 0.000000e+00, %0
; CHECK-NEXT:   %post_x1_unwrap = extractvalue { double, double, double* } %x1_cache.0, 1
; CHECK-NEXT:   %m0diffepost_x0 = fmul fast double %1, %post_x1_unwrap
; CHECK-NEXT:   %out1_unwrap = insertvalue { double, double, double* } %x1_cache.0, double* %in0, 2
; CHECK-NEXT:   %post_x0_unwrap = extractvalue { double, double, double* } %out1_unwrap, 0
; CHECK-NEXT:   %m1diffepost_x1 = fmul fast double %1, %post_x0_unwrap
; CHECK-NEXT:   %2 = fadd fast double 0.000000e+00, %m0diffepost_x0
; CHECK-NEXT:   %3 = fadd fast double 0.000000e+00, %m1diffepost_x1
; CHECK-NEXT:   %4 = getelementptr inbounds { double, double, double* }, { double, double, double* }* %"x1'de", i32 0, i32 1
; CHECK-NEXT:   %5 = load double, double* %4
; CHECK-NEXT:   %6 = fadd fast double %5, %3
; CHECK-NEXT:   store double %6, double* %4
; CHECK-NEXT:   %7 = getelementptr inbounds { double, double, double* }, { double, double, double* }* %"out1'de", i32 0, i32 0
; CHECK-NEXT:   %8 = load double, double* %7
; CHECK-NEXT:   %9 = fadd fast double %8, %2
; CHECK-NEXT:   store double %9, double* %7
; CHECK-NEXT:   %10 = load { double, double, double* }, { double, double, double* }* %"out1'de"
; CHECK-NEXT:   %11 = insertvalue { double, double, double* } %10, double* null, 2
; CHECK-NEXT:   %12 = load { double, double, double* }, { double, double, double* }* %"x1'de"
; CHECK-NEXT:   %13 = extractvalue { double, double, double* } %10, 0
; CHECK-NEXT:   %14 = getelementptr inbounds { double, double, double* }, { double, double, double* }* %"x1'de", i32 0, i32 0
; CHECK-NEXT:   %15 = load double, double* %14
; CHECK-NEXT:   %16 = fadd fast double %15, %13
; CHECK-NEXT:   store double %16, double* %14
; CHECK-NEXT:   %17 = extractvalue { double, double, double* } %10, 1
; CHECK-NEXT:   %18 = getelementptr inbounds { double, double, double* }, { double, double, double* }* %"x1'de", i32 0, i32 1
; CHECK-NEXT:   %19 = load double, double* %18
; CHECK-NEXT:   %20 = fadd fast double %19, %17
; CHECK-NEXT:   store double %20, double* %18
; CHECK-NEXT:   store { double, double, double* } zeroinitializer, { double, double, double* }* %"out1'de"
; CHECK-NEXT:   %21 = load { double, double, double* }, { double, double, double* }* %"x1'de"
; CHECK-NEXT:   %22 = extractvalue { double, double, double* } %21, 1
; CHECK-NEXT:   %23 = fadd fast double 0.000000e+00, %22
; CHECK-NEXT:   %24 = load { double, double, double* }, { double, double, double* }* %"x1'de"
; CHECK-NEXT:   %25 = insertvalue { double, double, double* } %24, double 0.000000e+00, 1
; CHECK-NEXT:   %26 = load { double, double, double* }, { double, double, double* }* %"x0'de"
; CHECK-NEXT:   %27 = extractvalue { double, double, double* } %24, 0
; CHECK-NEXT:   %28 = getelementptr inbounds { double, double, double* }, { double, double, double* }* %"x0'de", i32 0, i32 0
; CHECK-NEXT:   %29 = load double, double* %28
; CHECK-NEXT:   %30 = fadd fast double %29, %27
; CHECK-NEXT:   store double %30, double* %28
; CHECK-NEXT:   %31 = getelementptr inbounds { double, double, double* }, { double, double, double* }* %"x0'de", i32 0, i32 1
; CHECK-NEXT:   %32 = load double, double* %31
; CHECK-NEXT:   %33 = fadd fast double %32, 0.000000e+00
; CHECK-NEXT:   store double %33, double* %31
; CHECK-NEXT:   store { double, double, double* } zeroinitializer, { double, double, double* }* %"x1'de"
; CHECK-NEXT:   store double 0.000000e+00, double* %"in1'"
; CHECK-NEXT:   %34 = load double, double* %"in1'"
; CHECK-NEXT:   %35 = fadd fast double %34, %23
; CHECK-NEXT:   store double %35, double* %"in1'"
; CHECK-NEXT:   %36 = load { double, double, double* }, { double, double, double* }* %"x0'de"
; CHECK-NEXT:   %37 = extractvalue { double, double, double* } %36, 0
; CHECK-NEXT:   %38 = fadd fast double 0.000000e+00, %37
; CHECK-NEXT:   store { double, double, double* } zeroinitializer, { double, double, double* }* %"x0'de"
; CHECK-NEXT:   store double 0.000000e+00, double* %"in0'"
; CHECK-NEXT:   %39 = load double, double* %"in0'"
; CHECK-NEXT:   %40 = fadd fast double %39, %38
; CHECK-NEXT:   store double %40, double* %"in0'"
; CHECK-NEXT:   br label %invertentry

; CHECK: invertexit:                                       ; preds = %exit
; CHECK-NEXT:   br i1 %c, label %inverttrueb, label %invertentry
; CHECK-NEXT: }
