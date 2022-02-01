; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -early-cse -instsimplify -simplifycfg -S | FileCheck %s

declare double @llvm.pow.f64(double, double)

declare dso_local double @__enzyme_autodiff(i8*, double, i64)

@.str = private unnamed_addr constant [10 x i8] c"result=%f\00", align 1

declare dso_local i32 @printf(i8*, ...)

define void @main() {
entry:
  %call = tail call double @__enzyme_autodiff(i8* bitcast (double (double, i64)* @julia_euroad_1769 to i8*), double 0.5, i64 3)
  %p = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str, i64 0, i64 0), double %call)
  ret void
}

define double @julia_euroad_1769(double %arg, i64 %i5) {
bb:
  switch i64 %i5, label %bb9 [
    i64 12, label %bb12
    i64 7, label %bb7
  ]

bb7:                                              ; preds = %bb4
  %i7 = fmul double %arg, %arg
  br label %bb13

bb9:                                              ; preds = %bb4
  %ti5 = uitofp i64 %i5 to double
  %i9 = call double @llvm.pow.f64(double %arg, double %ti5)
  br label %bb13

bb12:                                             ; preds = %bb4
  br label %bb13

bb13:                                             ; preds = %bb12, %bb9, %bb8, %bb7, %bb4
  %i14 = phi double [ %i7, %bb7 ], [ %i9, %bb9 ], [ %arg, %bb12 ]
  ret double %i14
}

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}

; CHECK: define internal { double } @diffejulia_euroad_1769(double %arg, i64 %i5, double %differeturn)
; CHECK-NEXT: bb:
; CHECK-NEXT:   %0 = icmp eq i64 7, %i5
; CHECK-NEXT:   %1 = icmp eq i64 12, %i5
; CHECK-NEXT:   %2 = or i1 %1, %0
; CHECK-NEXT:   %3 = select {{(fast )?}}i1 %1, double %differeturn, double 0.000000e+00
; CHECK-NEXT:   %4 = select {{(fast )?}}i1 %2, double 0.000000e+00, double %differeturn
; CHECK-NEXT:   %5 = select {{(fast )?}}i1 %0, double %differeturn, double 0.000000e+00
; CHECK-NEXT:   switch i64 %i5, label %invertbb9 [
; CHECK-NEXT:     i64 12, label %invertbb
; CHECK-NEXT:     i64 7, label %invertbb7
; CHECK-NEXT:   ]

; CHECK: invertbb:                                         ; preds = %bb, %invertbb9, %invertbb7
; CHECK-NEXT:   %"arg'de.0" = phi double [ %13, %invertbb9 ], [ %8, %invertbb7 ], [ %3, %bb ]
; CHECK-NEXT:   %6 = insertvalue { double } undef, double %"arg'de.0", 0
; CHECK-NEXT:   ret { double } %6

; CHECK: invertbb7:                                        ; preds = %bb
; CHECK-NEXT:   %m0diffearg = fmul fast double %5, %arg
; CHECK-NEXT:   %7 = fadd fast double %3, %m0diffearg
; CHECK-NEXT:   %8 = fadd fast double %7, %m0diffearg
; CHECK-NEXT:   br label %invertbb

; CHECK: invertbb9:                                        ; preds = %bb
; CHECK-NEXT:   %ti5_unwrap = uitofp i64 %i5 to double
; CHECK-NEXT:   %9 = fsub fast double %ti5_unwrap, 1.000000e+00
; CHECK-NEXT:   %10 = call fast double @llvm.pow.f64(double %arg, double %9)
; CHECK-NEXT:   %11 = fmul fast double %4, %10
; CHECK-NEXT:   %12 = fmul fast double %11, %ti5_unwrap
; CHECK-NEXT:   %13 = fadd fast double %3, %12
; CHECK-NEXT:   br label %invertbb
; CHECK-NEXT: }
