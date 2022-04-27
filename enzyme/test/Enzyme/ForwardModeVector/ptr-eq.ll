; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -S | FileCheck %s

declare void @__enzyme_fwddiff(void (double*,double*)*, ...)
declare void @free(i8*)

define void @f(double* %x, double* %y) {
entry:
  %val = load double, double* %x
  store double %val, double* %y
  %ptr = bitcast double* %x to i8*
  call void @free(i8* %ptr)
  ret void
}

define void @df(double* %x, double* %xp, double* %y, double* %dy) {
entry:
  tail call void (void (double*,double*)*, ...) @__enzyme_fwddiff(void (double*,double*)* nonnull @f, metadata !"enzyme_width", i64 3, double* %x, double* %xp, double* %xp, double* %xp, double* %y, double* %dy, double* %dy, double* %dy)
  ret void
}


; CHECK: define internal void @fwddiffe3f(double* %x, [3 x double*] %"x'", double* %y, [3 x double*] %"y'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %val = load double, double* %x
; CHECK-NEXT:   %0 = extractvalue [3 x double*] %"x'", 0
; CHECK-NEXT:   %1 = load double, double* %0
; CHECK-NEXT:   %2 = insertvalue [3 x double] undef, double %1, 0
; CHECK-NEXT:   %3 = extractvalue [3 x double*] %"x'", 1
; CHECK-NEXT:   %4 = load double, double* %3
; CHECK-NEXT:   %5 = insertvalue [3 x double] %2, double %4, 1
; CHECK-NEXT:   %6 = extractvalue [3 x double*] %"x'", 2
; CHECK-NEXT:   %7 = load double, double* %6
; CHECK-NEXT:   %8 = insertvalue [3 x double] %5, double %7, 2
; CHECK-NEXT:   store double %val, double* %y
; CHECK-NEXT:   %9 = extractvalue [3 x double*] %"y'", 0
; CHECK-NEXT:   %10 = extractvalue [3 x double] %8, 0
; CHECK-NEXT:   store double %10, double* %9
; CHECK-NEXT:   %11 = extractvalue [3 x double*] %"y'", 1
; CHECK-NEXT:   %12 = extractvalue [3 x double] %8, 1
; CHECK-NEXT:   store double %12, double* %11
; CHECK-NEXT:   %13 = extractvalue [3 x double*] %"y'", 2
; CHECK-NEXT:   %14 = extractvalue [3 x double] %8, 2
; CHECK-NEXT:   store double %14, double* %13
; CHECK-NEXT:   %15 = extractvalue [3 x double*] %"x'", 0
; CHECK-NEXT:   %"ptr'ipc" = bitcast double* %15 to i8*
; CHECK-NEXT:   %16 = insertvalue [3 x i8*] undef, i8* %"ptr'ipc", 0
; CHECK-NEXT:   %17 = extractvalue [3 x double*] %"x'", 1
; CHECK-NEXT:   %"ptr'ipc1" = bitcast double* %17 to i8*
; CHECK-NEXT:   %18 = insertvalue [3 x i8*] %16, i8* %"ptr'ipc1", 1
; CHECK-NEXT:   %19 = extractvalue [3 x double*] %"x'", 2
; CHECK-NEXT:   %"ptr'ipc2" = bitcast double* %19 to i8*
; CHECK-NEXT:   %20 = insertvalue [3 x i8*] %18, i8* %"ptr'ipc2", 2
; CHECK-NEXT:   %ptr = bitcast double* %x to i8*
; CHECK-NEXT:   call void @free(i8* %ptr)
; CHECK-NEXT:   %21 = extractvalue [3 x i8*] %20, 0
; CHECK-NEXT:   %22 = extractvalue [3 x i8*] %20, 1
; CHECK-NEXT:   %23 = extractvalue [3 x i8*] %20, 2
; CHECK-NEXT:   %24 = icmp ne i8* %ptr, %21
; CHECK-NEXT:   br i1 %24, label %free0.i, label %__enzyme_checked_free_3.exit

; CHECK: free0.i:                                          ; preds = %entry
; CHECK-NEXT:   call void @free(i8* %21)
; CHECK-NEXT:   %25 = icmp ne i8* %21, %22
; CHECK-NEXT:   %26 = icmp ne i8* %22, %23
; CHECK-NEXT:   %27 = and i1 %26, %25
; CHECK-NEXT:   br i1 %27, label %free1.i, label %__enzyme_checked_free_3.exit

; CHECK: free1.i:                                          ; preds = %free0.i
; CHECK-NEXT:   call void @free(i8* %22)
; CHECK-NEXT:   call void @free(i8* %23)
; CHECK-NEXT:   br label %__enzyme_checked_free_3.exit

; CHECK: __enzyme_checked_free_3.exit:                     ; preds = %entry, %free0.i, %free1.i
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @__enzyme_checked_free_3(i8* nocapture{{( %0)?}}, i8* nocapture{{( %1)?}}, i8* nocapture{{( %2)?}}, i8* nocapture{{( %3)?}})
; CHECK-NEXT: entry:
; CHECK-NEXT:   %4 = icmp ne i8* %0, %1
; CHECK-NEXT:   br i1 %4, label %free0, label %end

; CHECK: free0:                                            ; preds = %entry
; CHECK-NEXT:   call void @free(i8* %1)
; CHECK-NEXT:   %5 = icmp ne i8* %1, %2
; CHECK-NEXT:   %6 = icmp ne i8* %2, %3
; CHECK-NEXT:   %7 = and i1 %6, %5
; CHECK-NEXT:   br i1 %7, label %free1, label %end

; CHECK: end:                                              ; preds = %free1, %free0, %entry
; CHECK-NEXT:   ret void

; CHECK: free1:                                            ; preds = %free0
; CHECK-NEXT:   call void @free(i8* %2)
; CHECK-NEXT:   call void @free(i8* %3)
; CHECK-NEXT:   br label %end
; CHECK-NEXT: }