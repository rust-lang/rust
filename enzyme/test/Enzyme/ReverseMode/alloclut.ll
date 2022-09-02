; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -adce -loop-deletion -correlated-propagation -simplifycfg -adce -S | FileCheck %s

; Function Attrs: inlinehint nounwind uwtable
define double @f(double %x, i32* %z) {
entry:
  %a = alloca [2 x double], align 32
  br label %loop1

loop1:
  %k = phi i64 [ 0, %entry ], [ %k1, %cleanup ]
  %sum = phi double [ 0.000000e+00, %entry ], [ %sumj, %cleanup ]
  %k1 = add nuw nsw i64 %k, 1
  %g0 = getelementptr [2 x double], [2 x double]* %a, i64 0, i32 0
  %fp = uitofp i64 %k to double
  store double %x, double* %g0, align 8
  %g1 = getelementptr [2 x double], [2 x double]* %a, i64 0, i32 1
  %m1 = fadd double %x, %fp
  store double %m1, double* %g1, align 8
  br label %loop2

loop2:
  %j = phi i64 [ %j1, %loop2 ], [ 0, %loop1 ]
  %sumj = phi double [ %add, %loop2 ], [ 0.000000e+00, %loop1 ]
  %idx = add i64 %k, %j
  %zgep = getelementptr inbounds i32, i32* %z, i64 %idx
  %lu = load i32, i32* %zgep, align 4
  %which = getelementptr [2 x double], [2 x double]* %a, i64 0, i32 %lu
  %val = load double, double* %which, align 8
  %val2 = fmul double %val, %val
  %add = fadd double %val2, %sumj
  %j1 = add nuw nsw i64 %j, 1
  %exit2 = icmp eq i64 %j1, 400
  br i1 %exit2, label %cleanup, label %loop2

cleanup:
  %exit1 = icmp eq i64 %k1, 4
  br i1 %exit1, label %exit, label %loop1

exit:
  ret double %add
}

define dso_local void @dsum(double %x, i32* %LUT) {
entry:
  %0 = tail call double (double (double, i32*)*, ...) @__enzyme_autodiff(double (double, i32*)* nonnull @f, double %x, metadata !"enzyme_const", i32* %LUT)
  ret void
}

declare double @__enzyme_autodiff(double (double, i32*)*, ...)

; CHECK: define internal { double } @diffef(double %x, i32* %z, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"a'ipa" = alloca [2 x double]
; CHECK-NEXT:   store [2 x double] zeroinitializer, [2 x double]* %"a'ipa"
; CHECK-NEXT:   %a = alloca [2 x double]
; CHECK-NEXT:   %malloccall = tail call noalias nonnull dereferenceable(64) dereferenceable_or_null(64) i8* @malloc(i64 64)
; CHECK-NEXT:   %val_malloccache = bitcast i8* %malloccall to [2 x double]*
; CHECK-NEXT:   br label %loop1

; CHECK: loop1:                                            ; preds = %loop1, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %loop1 ], [ 0, %entry ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %g0 = getelementptr [2 x double], [2 x double]* %a, i64 0, i32 0
; CHECK-NEXT:   %fp = uitofp i64 %iv to double
; CHECK-NEXT:   store double %x, double* %g0, align 8
; CHECK-NEXT:   %g1 = getelementptr [2 x double], [2 x double]* %a, i64 0, i32 1
; CHECK-NEXT:   %m1 = fadd double %x, %fp
; CHECK-NEXT:   store double %m1, double* %g1, align 8
; CHECK-NEXT:   %0 = getelementptr inbounds [2 x double], [2 x double]* %val_malloccache, i64 %iv
; CHECK-NEXT:   %1 = load [2 x double], [2 x double]* %a, align 32
; CHECK-NEXT:   store [2 x double] %1, [2 x double]* %0, align 16
; CHECK-NEXT:   %exit1 = icmp eq i64 %iv.next, 4
; CHECK-NEXT:   br i1 %exit1, label %mergeinvertloop1_exit, label %loop1

; CHECK: invertentry:                                      ; preds = %invertloop1
; CHECK-NEXT:   %2 = insertvalue { double } undef, double %6, 0
; CHECK-NEXT:   tail call void @free(i8* nonnull %malloccall)
; CHECK-NEXT:   ret { double } %2

; CHECK: invertloop1:                                      ; preds = %invertloop2
; CHECK-NEXT:   %3 = load double, double* %"g1'ipg_unwrap"
; CHECK-NEXT:   store double 0.000000e+00, double* %"g1'ipg_unwrap"
; CHECK-NEXT:   %4 = fadd fast double %"x'de.0", %3
; CHECK-NEXT:   %"g0'ipg_unwrap" = getelementptr [2 x double], [2 x double]* %"a'ipa", i64 0, i32 0
; CHECK-NEXT:   %5 = load double, double* %"g0'ipg_unwrap"
; CHECK-NEXT:   store double 0.000000e+00, double* %"g0'ipg_unwrap"
; CHECK-NEXT:   %6 = fadd fast double %4, %5
; CHECK-NEXT:   %7 = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %7, label %invertentry, label %incinvertloop1

; CHECK: incinvertloop1:                                   ; preds = %invertloop1
; CHECK-NEXT:   %8 = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertcleanup

; CHECK: invertloop2:                                      ; preds = %invertcleanup, %incinvertloop2
; CHECK-NEXT:   %"iv1'ac.0" = phi i64 [ 399, %invertcleanup ], [ %16, %incinvertloop2 ]
; CHECK-NEXT:   %9 = getelementptr inbounds [2 x double], [2 x double]* %val_malloccache, i64 %"iv'ac.0"
; CHECK-NEXT:   %idx_unwrap = add i64 %"iv'ac.0", %"iv1'ac.0"
; CHECK-NEXT:   %zgep_unwrap = getelementptr inbounds i32, i32* %z, i64 %idx_unwrap
; CHECK-NEXT:   %lu_unwrap = load i32, i32* %zgep_unwrap, align 4, !invariant.group !
; CHECK-NEXT:   %10 = getelementptr inbounds [2 x double], [2 x double]* %9, i64 0, i32 %lu_unwrap
; CHECK-NEXT:   %11 = load double, double* %10, align 8, !invariant.group !
; CHECK-NEXT:   %m0diffeval = fmul fast double %"add'de.1", %11
; CHECK-NEXT:   %m1diffeval = fmul fast double %"add'de.1", %11
; CHECK-NEXT:   %12 = fadd fast double %m0diffeval, %m1diffeval
; CHECK-NEXT:   %"which'ipg_unwrap" = getelementptr [2 x double], [2 x double]* %"a'ipa", i64 0, i32 %lu_unwrap
; CHECK-NEXT:   %13 = load double, double* %"which'ipg_unwrap"
; CHECK-NEXT:   %14 = fadd fast double %13, %12
; CHECK-NEXT:   store double %14, double* %"which'ipg_unwrap"
; CHECK-NEXT:   %15 = icmp eq i64 %"iv1'ac.0", 0
; CHECK-NEXT:   br i1 %15, label %invertloop1, label %incinvertloop2

; CHECK: incinvertloop2:                                   ; preds = %invertloop2
; CHECK-NEXT:   %16 = add nsw i64 %"iv1'ac.0", -1
; CHECK-NEXT:   br label %invertloop2

; CHECK: invertcleanup:                                    ; preds = %mergeinvertloop1_exit, %incinvertloop1
; CHECK-NEXT:   %"x'de.0" = phi double [ 0.000000e+00, %mergeinvertloop1_exit ], [ %6, %incinvertloop1 ]
; CHECK-NEXT:   %"add'de.1" = phi double [ %differeturn, %mergeinvertloop1_exit ], [ 0.000000e+00, %incinvertloop1 ]
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ 3, %mergeinvertloop1_exit ], [ %8, %incinvertloop1 ]
; CHECK-NEXT:   br label %invertloop2

; CHECK: mergeinvertloop1_exit:                            ; preds = %loop1
; CHECK-NEXT:   %"g1'ipg_unwrap" = getelementptr [2 x double], [2 x double]* %"a'ipa", i64 0, i32 1
; CHECK-NEXT:   br label %invertcleanup
; CHECK-NEXT: }
