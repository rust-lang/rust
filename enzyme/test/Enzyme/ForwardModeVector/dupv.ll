; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -simplifycfg -S | FileCheck %s


@enzyme_width = external global i32, align 4
@enzyme_dupv = external global i32, align 4

define void @square(double* nocapture readonly %x, double* nocapture %out) {
entry:
  %0 = load double, double* %x, align 8
  %mul = fmul double %0, %0
  store double %mul, double* %out, align 8
  ret void
}

define void @dsquare(double* %x, double* %dx, double* %out, double* %dout) {
entry:
  %0 = load i32, i32* @enzyme_width, align 4
  %1 = load i32, i32* @enzyme_dupv, align 4
  call void (i8*, ...) @__enzyme_fwddiff(i8* bitcast (void (double*, double*)* @square to i8*), i32 %0, i32 3, i32 %1, i64 16, double* %x, double* %dx, i32 %1, i64 16, double* %out, double* %dout)
  ret void
}

declare void @__enzyme_fwddiff(i8*, ...)


; CHECK: define void @dsquare(double* %x, double* %dx, double* %out, double* %dout)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = load i32, i32* @enzyme_width, align 4
; CHECK-NEXT:   %1 = load i32, i32* @enzyme_dupv, align 4
; CHECK-NEXT:   %2 = bitcast double* %dx to i8*
; CHECK-NEXT:   %3 = getelementptr i8, i8* %2, i64 0
; CHECK-NEXT:   %4 = bitcast i8* %3 to double*
; CHECK-NEXT:   %5 = insertvalue [3 x double*] undef, double* %4, 0
; CHECK-NEXT:   %6 = bitcast double* %dx to i8*
; CHECK-NEXT:   %7 = getelementptr i8, i8* %6, i64 16
; CHECK-NEXT:   %8 = bitcast i8* %7 to double*
; CHECK-NEXT:   %9 = insertvalue [3 x double*] %5, double* %8, 1
; CHECK-NEXT:   %10 = bitcast double* %dx to i8*
; CHECK-NEXT:   %11 = getelementptr i8, i8* %10, i64 32
; CHECK-NEXT:   %12 = bitcast i8* %11 to double*
; CHECK-NEXT:   %13 = insertvalue [3 x double*] %9, double* %12, 2
; CHECK-NEXT:   %14 = bitcast double* %dout to i8*
; CHECK-NEXT:   %15 = getelementptr i8, i8* %14, i64 0
; CHECK-NEXT:   %16 = bitcast i8* %15 to double*
; CHECK-NEXT:   %17 = insertvalue [3 x double*] undef, double* %16, 0
; CHECK-NEXT:   %18 = bitcast double* %dout to i8*
; CHECK-NEXT:   %19 = getelementptr i8, i8* %18, i64 16
; CHECK-NEXT:   %20 = bitcast i8* %19 to double*
; CHECK-NEXT:   %21 = insertvalue [3 x double*] %17, double* %20, 1
; CHECK-NEXT:   %22 = bitcast double* %dout to i8*
; CHECK-NEXT:   %23 = getelementptr i8, i8* %22, i64 32
; CHECK-NEXT:   %24 = bitcast i8* %23 to double*
; CHECK-NEXT:   %25 = insertvalue [3 x double*] %21, double* %24, 2
; CHECK-NEXT:   call void @fwddiffe3square(double* %x, [3 x double*] %13, double* %out, [3 x double*] %25)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @fwddiffe3square(double* nocapture readonly %x, [3 x double*] %"x'", double* nocapture %out, [3 x double*] %"out'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = load double, double* %x, align 8
; CHECK-NEXT:   %1 = extractvalue [3 x double*] %"x'", 0
; CHECK-NEXT:   %2 = load double, double* %1, align 8
; CHECK-NEXT:   %3 = insertvalue [3 x double] undef, double %2, 0
; CHECK-NEXT:   %4 = extractvalue [3 x double*] %"x'", 1
; CHECK-NEXT:   %5 = load double, double* %4, align 8
; CHECK-NEXT:   %6 = insertvalue [3 x double] %3, double %5, 1
; CHECK-NEXT:   %7 = extractvalue [3 x double*] %"x'", 2
; CHECK-NEXT:   %8 = load double, double* %7, align 8
; CHECK-NEXT:   %9 = insertvalue [3 x double] %6, double %8, 2
; CHECK-NEXT:   %mul = fmul double %0, %0
; CHECK-NEXT:   %10 = extractvalue [3 x double] %9, 0
; CHECK-NEXT:   %11 = extractvalue [3 x double] %9, 0
; CHECK-NEXT:   %12 = fmul fast double %10, %0
; CHECK-NEXT:   %13 = fmul fast double %11, %0
; CHECK-NEXT:   %14 = fadd fast double %12, %13
; CHECK-NEXT:   %15 = insertvalue [3 x double] undef, double %14, 0
; CHECK-NEXT:   %16 = extractvalue [3 x double] %9, 1
; CHECK-NEXT:   %17 = extractvalue [3 x double] %9, 1
; CHECK-NEXT:   %18 = fmul fast double %16, %0
; CHECK-NEXT:   %19 = fmul fast double %17, %0
; CHECK-NEXT:   %20 = fadd fast double %18, %19
; CHECK-NEXT:   %21 = insertvalue [3 x double] %15, double %20, 1
; CHECK-NEXT:   %22 = extractvalue [3 x double] %9, 2
; CHECK-NEXT:   %23 = extractvalue [3 x double] %9, 2
; CHECK-NEXT:   %24 = fmul fast double %22, %0
; CHECK-NEXT:   %25 = fmul fast double %23, %0
; CHECK-NEXT:   %26 = fadd fast double %24, %25
; CHECK-NEXT:   %27 = insertvalue [3 x double] %21, double %26, 2
; CHECK-NEXT:   store double %mul, double* %out, align 8
; CHECK-NEXT:   %28 = extractvalue [3 x double*] %"out'", 0
; CHECK-NEXT:   %29 = extractvalue [3 x double] %27, 0
; CHECK-NEXT:   store double %29, double* %28, align 8
; CHECK-NEXT:   %30 = extractvalue [3 x double*] %"out'", 1
; CHECK-NEXT:   %31 = extractvalue [3 x double] %27, 1
; CHECK-NEXT:   store double %31, double* %30, align 8
; CHECK-NEXT:   %32 = extractvalue [3 x double*] %"out'", 2
; CHECK-NEXT:   %33 = extractvalue [3 x double] %27, 2
; CHECK-NEXT:   store double %33, double* %32, align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
