; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -inline -mem2reg -instsimplify -gvn -dse -dse -S | FileCheck %s

define void @addOneMem(double* nocapture %x) {
entry:
  %0 = load double, double* %x
  %add = fadd double %0, 1.000000e+00
  store double %add, double* %x
  ret void
}

define void @test_derivative(double* %x, double* %xp1, double* %xp2, double* %xp3) {
entry:
  call void (void (double*)*, ...) @__enzyme_autodiff(void (double*)* nonnull @addOneMem, metadata !"enzyme_width", i64 3, double* %x, double* %xp1, double* %xp2, double* %xp3)
  ret void
}

declare void @__enzyme_autodiff(void (double*)*, ...)

; CHECK: define void @test_derivative(double* %x, double* %xp1, double* %xp2, double* %xp3)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"add'de.i" = alloca [3 x double]
; CHECK-NEXT:   %"'de.i" = alloca [3 x double]
; CHECK-NEXT:   %0 = bitcast [3 x double]* %"add'de.i" to i8*
; CHECK-NEXT:   call void @llvm.lifetime.start.p0i8(i64 24, i8* %0)
; CHECK-NEXT:   %1 = bitcast [3 x double]* %"'de.i" to i8*
; CHECK-NEXT:   call void @llvm.lifetime.start.p0i8(i64 24, i8* %1)
; CHECK-NEXT:   store [3 x double] zeroinitializer, [3 x double]* %"add'de.i"
; CHECK-NEXT:   store [3 x double] zeroinitializer, [3 x double]* %"'de.i"
; CHECK-NEXT:   %2 = load double, double* %x
; CHECK-NEXT:   %add.i = fadd double %2, 1.000000e+00
; CHECK-NEXT:   store double %add.i, double* %x
; CHECK-NEXT:   %3 = load double, double* %xp1
; CHECK-NEXT:   %4 = load double, double* %xp2
; CHECK-NEXT:   %5 = load double, double* %xp3
; CHECK-NEXT:   store double 0.000000e+00, double* %xp1
; CHECK-NEXT:   store double 0.000000e+00, double* %xp2
; CHECK-NEXT:   store double 0.000000e+00, double* %xp3
; CHECK-NEXT:   %6 = getelementptr inbounds [3 x double], [3 x double]* %"add'de.i", i32 0, i32 0
; CHECK-NEXT:   %7 = load double, double* %6
; CHECK-NEXT:   %8 = fadd fast double %7, %3
; CHECK-NEXT:   store double %8, double* %6
; CHECK-NEXT:   %9 = getelementptr inbounds [3 x double], [3 x double]* %"add'de.i", i32 0, i32 1
; CHECK-NEXT:   %10 = load double, double* %9
; CHECK-NEXT:   %11 = fadd fast double %10, %4
; CHECK-NEXT:   store double %11, double* %9
; CHECK-NEXT:   %12 = getelementptr inbounds [3 x double], [3 x double]* %"add'de.i", i32 0, i32 2
; CHECK-NEXT:   %13 = load double, double* %12
; CHECK-NEXT:   %14 = fadd fast double %13, %5
; CHECK-NEXT:   store double %14, double* %12
; CHECK-NEXT:   %15 = load [3 x double], [3 x double]* %"add'de.i"
; CHECK-NEXT:   %16 = extractvalue [3 x double] %15, 0
; CHECK-NEXT:   %17 = getelementptr inbounds [3 x double], [3 x double]* %"'de.i", i32 0, i32 0
; CHECK-NEXT:   %18 = load double, double* %17
; CHECK-NEXT:   %19 = fadd fast double %18, %16
; CHECK-NEXT:   store double %19, double* %17
; CHECK-NEXT:   %20 = extractvalue [3 x double] %15, 1
; CHECK-NEXT:   %21 = getelementptr inbounds [3 x double], [3 x double]* %"'de.i", i32 0, i32 1
; CHECK-NEXT:   %22 = load double, double* %21
; CHECK-NEXT:   %23 = fadd fast double %22, %20
; CHECK-NEXT:   store double %23, double* %21
; CHECK-NEXT:   %24 = extractvalue [3 x double] %15, 2
; CHECK-NEXT:   %25 = getelementptr inbounds [3 x double], [3 x double]* %"'de.i", i32 0, i32 2
; CHECK-NEXT:   %26 = load double, double* %25
; CHECK-NEXT:   %27 = fadd fast double %26, %24
; CHECK-NEXT:   store double %27, double* %25
; CHECK-NEXT:   %28 = load [3 x double], [3 x double]* %"'de.i"
; CHECK-NEXT:   %[[i31:.+]] = extractvalue [3 x double] %28, 0
; CHECK-NEXT:   %[[i29:.+]] = load double, double* %xp1
; CHECK-NEXT:   %[[i32:.+]] = fadd fast double %[[i29]], %[[i31]]
; CHECK-NEXT:   store double %[[i32]], double* %xp1
; CHECK-NEXT:   %[[i33:.+]] = extractvalue [3 x double] %28, 1
; CHECK-NEXT:   %[[i30:.+]] = load double, double* %xp2
; CHECK-NEXT:   %[[i34:.+]] = fadd fast double %[[i30]], %[[i33]]
; CHECK-NEXT:   store double %[[i34]], double* %xp2
; CHECK-NEXT:   %[[i35:.+]] = extractvalue [3 x double] %28, 2
; CHECK-NEXT:   %[[i36:.+]] = load double, double* %xp3
; CHECK-NEXT:   %[[i37:.+]] = fadd fast double %[[i36]], %[[i35]]
; CHECK-NEXT:   store double %[[i37]], double* %xp3
; CHECK-NEXT:   call void @llvm.lifetime.end.p0i8(i64 24, i8* %0)
; CHECK-NEXT:   call void @llvm.lifetime.end.p0i8(i64 24, i8* %1)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
