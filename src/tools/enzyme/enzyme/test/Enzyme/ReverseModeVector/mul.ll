; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -early-cse -simplifycfg -S | FileCheck %s

%struct.Gradients = type { [2 x double], [2 x double] }

; Function Attrs: nounwind
declare %struct.Gradients @__enzyme_autodiff(double (double, double)*, ...)

; Function Attrs: noinline nounwind readnone uwtable
define double @tester(double %x, double %y) {
entry:
  %0 = fmul fast double %x, %y
  ret double %0
}

define %struct.Gradients @test_derivative(double %x, double %y) {
entry:
  %0 = tail call %struct.Gradients (double (double, double)*, ...) @__enzyme_autodiff(double (double, double)* nonnull @tester, metadata !"enzyme_width", i64 2, double %x, double %y)
  ret %struct.Gradients %0
}


; CHECK: define internal { [2 x double], [2 x double] } @diffe2tester(double %x, double %y, [2 x double] %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"x'de" = alloca [2 x double]
; CHECK-NEXT:   store [2 x double] zeroinitializer, [2 x double]* %"x'de"
; CHECK-NEXT:   %"y'de" = alloca [2 x double]
; CHECK-NEXT:   store [2 x double] zeroinitializer, [2 x double]* %"y'de"
; CHECK-NEXT:   %0 = extractvalue [2 x double] %differeturn, 0
; CHECK-NEXT:   %m0diffex = fmul fast double %0, %y
; CHECK-NEXT:   %1 = insertvalue [2 x double] undef, double %m0diffex, 0
; CHECK-NEXT:   %2 = extractvalue [2 x double] %differeturn, 1
; CHECK-NEXT:   %m0diffex1 = fmul fast double %2, %y
; CHECK-NEXT:   %m1diffey = fmul fast double %0, %x
; CHECK-NEXT:   %[[i4:.+]] = insertvalue [2 x double] undef, double %m1diffey, 0
; CHECK-NEXT:   %m1diffey2 = fmul fast double %2, %x
; CHECK-NEXT:   %[[i6:.+]] = getelementptr inbounds [2 x double], [2 x double]* %"x'de", i32 0, i32 0
; CHECK-NEXT:   %[[i7:.+]] = load double, double* %[[i6]]
; CHECK-NEXT:   %[[i8:.+]] = fadd fast double %[[i7]], %m0diffex
; CHECK-NEXT:   store double %[[i8]], double* %[[i6]]
; CHECK-NEXT:   %[[i9:.+]] = getelementptr inbounds [2 x double], [2 x double]* %"x'de", i32 0, i32 1
; CHECK-NEXT:   %[[i10:.+]] = load double, double* %[[i9]]
; CHECK-NEXT:   %[[i11:.+]] = fadd fast double %[[i10]], %m0diffex1
; CHECK-NEXT:   store double %[[i11]], double* %[[i9]]
; CHECK-NEXT:   %[[i12:.+]] = getelementptr inbounds [2 x double], [2 x double]* %"y'de", i32 0, i32 0
; CHECK-NEXT:   %[[i13:.+]] = load double, double* %[[i12]]
; CHECK-NEXT:   %[[i14:.+]] = fadd fast double %[[i13]], %m1diffey
; CHECK-NEXT:   store double %[[i14]], double* %[[i12]]
; CHECK-NEXT:   %[[i15:.+]] = getelementptr inbounds [2 x double], [2 x double]* %"y'de", i32 0, i32 1
; CHECK-NEXT:   %[[i16:.+]] = load double, double* %[[i15]]
; CHECK-NEXT:   %[[i17:.+]] = fadd fast double %[[i16]], %m1diffey2
; CHECK-NEXT:   store double %[[i17]], double* %[[i15]]
; CHECK-NEXT:   %[[i18:.+]] = load [2 x double], [2 x double]* %"x'de"
; CHECK-NEXT:   %[[i19:.+]] = load [2 x double], [2 x double]* %"y'de"
; CHECK-NEXT:   %[[i20:.+]] = insertvalue { [2 x double], [2 x double] } undef, [2 x double] %[[i18]], 0
; CHECK-NEXT:   %[[i21:.+]] = insertvalue { [2 x double], [2 x double] } %[[i20]], [2 x double] %[[i19]], 1
; CHECK-NEXT:   ret { [2 x double], [2 x double] } %[[i21]]
; CHECK-NEXT: }
