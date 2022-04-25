; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -simplifycfg -S | FileCheck %s

@enzyme_width = external global i32, align 4
@enzyme_dupv = external global i32, align 4
@enzyme_dupnoneedv = external global i32, align 4


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
   %2 = load i32, i32* @enzyme_dupnoneedv, align 4
   call void (i8*, ...) @__enzyme_fwddiff(i8* bitcast (void (double*, double*)* @square to i8*), i32 %0, i32 3, i32 %1, i64 16, double* %x, double* %dx, i32 %1, i64 16, i32 %2, i32 16, double* %out, double* %dout)
   ret void
 }

 declare void @__enzyme_fwddiff(i8*, ...)

; CHECK: define void @dsquare(double* %x, double* %dx, double* %out, double* %dout)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = load i32, i32* @enzyme_width
; CHECK-NEXT:   %1 = load i32, i32* @enzyme_dupv
; CHECK-NEXT:   %2 = load i32, i32* @enzyme_dupnoneedv
; CHECK-NEXT:   %3 = bitcast double* %dx to i8*
; CHECK-NEXT:   %4 = getelementptr i8, i8* %3, i64 0
; CHECK-NEXT:   %5 = bitcast i8* %4 to double*
; CHECK-NEXT:   %6 = insertvalue [3 x double*] undef, double* %5, 0
; CHECK-NEXT:   %7 = bitcast double* %dx to i8*
; CHECK-NEXT:   %8 = getelementptr i8, i8* %7, i64 16
; CHECK-NEXT:   %9 = bitcast i8* %8 to double*
; CHECK-NEXT:   %10 = insertvalue [3 x double*] %6, double* %9, 1
; CHECK-NEXT:   %11 = bitcast double* %dx to i8*
; CHECK-NEXT:   %12 = getelementptr i8, i8* %11, i64 32
; CHECK-NEXT:   %13 = bitcast i8* %12 to double*
; CHECK-NEXT:   %14 = insertvalue [3 x double*] %10, double* %13, 2
; CHECK-NEXT:   %15 = bitcast double* %dout to i8*
; CHECK-NEXT:   %16 = getelementptr i8, i8* %15, i32 0
; CHECK-NEXT:   %17 = bitcast i8* %16 to double*
; CHECK-NEXT:   %18 = insertvalue [3 x double*] undef, double* %17, 0
; CHECK-NEXT:   %19 = bitcast double* %dout to i8*
; CHECK-NEXT:   %20 = getelementptr i8, i8* %19, i32 16
; CHECK-NEXT:   %21 = bitcast i8* %20 to double*
; CHECK-NEXT:   %22 = insertvalue [3 x double*] %18, double* %21, 1
; CHECK-NEXT:   %23 = bitcast double* %dout to i8*
; CHECK-NEXT:   %24 = getelementptr i8, i8* %23, i32 32
; CHECK-NEXT:   %25 = bitcast i8* %24 to double*
; CHECK-NEXT:   %26 = insertvalue [3 x double*] %22, double* %25, 2
; CHECK-NEXT:   call void @fwddiffe3square(double* %x, [3 x double*] %14, double* %out, [3 x double*] %26)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @fwddiffe3square(double* nocapture readonly %x, [3 x double*] %"x'", double* nocapture %out, [3 x double*] %"out'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = load double, double* %x
; CHECK-NEXT:   %1 = extractvalue [3 x double*] %"x'", 0
; CHECK-NEXT:   %2 = load double, double* %1
; CHECK-NEXT:   %3 = insertvalue [3 x double] undef, double %2, 0
; CHECK-NEXT:   %4 = extractvalue [3 x double*] %"x'", 1
; CHECK-NEXT:   %5 = load double, double* %4
; CHECK-NEXT:   %6 = insertvalue [3 x double] %3, double %5, 1
; CHECK-NEXT:   %7 = extractvalue [3 x double*] %"x'", 2
; CHECK-NEXT:   %8 = load double, double* %7
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
; CHECK-NEXT:   store double %mul, double* %out
; CHECK-NEXT:   %28 = extractvalue [3 x double*] %"out'", 0
; CHECK-NEXT:   %29 = extractvalue [3 x double] %27, 0
; CHECK-NEXT:   store double %29, double* %28
; CHECK-NEXT:   %30 = extractvalue [3 x double*] %"out'", 1
; CHECK-NEXT:   %31 = extractvalue [3 x double] %27, 1
; CHECK-NEXT:   store double %31, double* %30,
; CHECK-NEXT:   %32 = extractvalue [3 x double*] %"out'", 2
; CHECK-NEXT:   %33 = extractvalue [3 x double] %27, 2
; CHECK-NEXT:   store double %33, double* %32
; CHECK-NEXT:   ret void
; CHECK-NEXT: }