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
  tail call void (void (double*,double*)*, ...) @__enzyme_fwddiff(void (double*,double*)* nonnull @f, double* %x, double* %xp, double* %y, double* %dy)
  ret void
}


; CHECK: define internal void @fwddiffef(double* %x, double* %"x'", double* %y, double* %"y'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %val = load double, double* %x
; CHECK-NEXT:   %0 = load double, double* %"x'"
; CHECK-NEXT:   store double %val, double* %y
; CHECK-NEXT:   store double %0, double* %"y'"
; CHECK-NEXT:   %"ptr'ipc" = bitcast double* %"x'" to i8*
; CHECK-NEXT:   %ptr = bitcast double* %x to i8*
; CHECK-NEXT:   call void @free(i8* %ptr)
; CHECK-NEXT:   call void @__enzyme_checked_free_1(i8* %ptr, i8* %"ptr'ipc")
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @__enzyme_checked_free_1(i8* nocapture{{( %0)?}}, i8* nocapture{{( %1)?}})
; CHECK-NEXT: entry:
; CHECK-NEXT:   %2 = icmp ne i8* %0, %1
; CHECK-NEXT:   br i1 %2, label %free0, label %end

; CHECK: free0:                                            ; preds = %entry
; CHECK-NEXT:   call void @free(i8* %1)
; CHECK-NEXT:   br label %end

; CHECK: end:                                              ; preds = %free0, %entry
; CHECK-NEXT:   ret void
; CHECK-NEXT: }