; RUN: opt < %s %loadEnzyme -enzyme -enzyme_preopt=false -mem2reg -instsimplify -adce -loop-deletion -correlated-propagation -simplifycfg -S | FileCheck %s

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local void @f(double* %x, double* %z, i64* %rows) {
entry:
  br label %loop1

loop1:
  %a17 = phi i64 [ 4, %entry ], [ %.pre, %prel1 ]
  %k = phi i64 [ 0, %entry ], [ %k1, %prel1 ]
  %X1 = getelementptr inbounds double, double* %x, i64 %k
  %L1 = load double, double* %X1
  br label %loop2

loop2:
  %j = phi i64 [ %j1, %loop2 ], [ 0, %loop1 ]
  %X2 = getelementptr inbounds double, double* %x, i64 %j
  %L2 = load double, double* %X2
  %why = fmul fast double %L1, %L2
  %tostore = getelementptr inbounds double, double* %z, i64 %a17
  store double %why, double* %tostore
  %j1 = add nuw nsw i64 %j, 1
  %exit2 = icmp eq i64 %j1, 4
  br i1 %exit2, label %cleanup, label %loop2

cleanup:
  %k1 = add nuw nsw i64 %k, 1
  %.pre = load i64, i64* %rows
  %exit1 = icmp eq i64 %k1, 4
  br i1 %exit1, label %exit, label %prel1

prel1:
  br label %loop1

exit:
  ret void
}

define dso_local void @dsum(double* %x, double* %xp, double* %z, double* %zp, i64* %n, i64* %np) {
entry:
  %0 = tail call double (void (double*, double*, i64*)*, ...) @__enzyme_autodiff(void (double*, double*, i64*)* nonnull @f, double* %x, double* %xp, double* %z, double* %zp, metadata !"diffe_const", i64* %n)
  ret void
}

declare double @__enzyme_autodiff(void (double*, double*, i64*)*, ...)

; CHECK: define internal {} @diffef(double* %x, double* %"x'", double* %z, double* %"z'", i64* %rows) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %malloccall = tail call noalias nonnull i8* @malloc(i64 32)
; CHECK-NEXT:   %L1_malloccache = bitcast i8* %malloccall to double*
; CHECK-NEXT:   %malloccall3 = tail call noalias nonnull i8* @malloc(i64 32)
; CHECK-NEXT:   %a17_malloccache = bitcast i8* %malloccall3 to i64*
; CHECK-NEXT:   %malloccall5 = tail call noalias nonnull i8* @malloc(i64 128)
; CHECK-NEXT:   %L2_malloccache = bitcast i8* %malloccall5 to double*
; CHECK-NEXT:   br label %loop1

; CHECK: loop1:                                            ; preds = %cleanup, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %cleanup ], [ 0, %entry ]
; CHECK-NEXT:   %a17 = phi i64 [ 4, %entry ], [ %.pre, %cleanup ]
; CHECK-NEXT:   %0 = getelementptr i64, i64* %a17_malloccache, i64 %iv
; CHECK-NEXT:   store i64 %a17, i64* %0
; CHECK-NEXT:   %iv.next = add nuw i64 %iv, 1
; CHECK-NEXT:   %X1 = getelementptr inbounds double, double* %x, i64 %iv
; CHECK-NEXT:   %L1 = load double, double* %X1
; CHECK-NEXT:   %1 = getelementptr double, double* %L1_malloccache, i64 %iv
; CHECK-NEXT:   store double %L1, double* %1
; CHECK-NEXT:   br label %loop2

; CHECK: loop2:                                            ; preds = %loop2, %loop1
; CHECK-NEXT:   %iv1 = phi i64 [ %iv.next2, %loop2 ], [ 0, %loop1 ]
; CHECK-NEXT:   %iv.next2 = add nuw i64 %iv1, 1
; CHECK-NEXT:   %X2 = getelementptr inbounds double, double* %x, i64 %iv1
; CHECK-NEXT:   %L2 = load double, double* %X2
; CHECK-NEXT:   %2 = mul nuw i64 %iv1, 4
; CHECK-NEXT:   %3 = add nuw i64 %iv, %2
; CHECK-NEXT:   %4 = getelementptr double, double* %L2_malloccache, i64 %3
; CHECK-NEXT:   store double %L2, double* %4
; CHECK-NEXT:   %why = fmul fast double %L1, %L2
; CHECK-NEXT:   %tostore = getelementptr inbounds double, double* %z, i64 %a17
; CHECK-NEXT:   store double %why, double* %tostore
; CHECK-NEXT:   %exit2 = icmp eq i64 %iv.next2, 4
; CHECK-NEXT:   br i1 %exit2, label %cleanup, label %loop2

; CHECK: cleanup:                                          ; preds = %loop2
; CHECK-NEXT:   %.pre = load i64, i64* %rows
; CHECK-NEXT:   %exit1 = icmp eq i64 %iv.next, 4
; CHECK-NEXT:   br i1 %exit1, label %invertcleanup, label %loop1

; CHECK: invertentry:                                      ; preds = %invertloop1
; CHECK-NEXT:   tail call void @free(i8* nonnull %malloccall5)
; CHECK-NEXT:   tail call void @free(i8* nonnull %malloccall3)
; CHECK-NEXT:   tail call void @free(i8* nonnull %malloccall)
; CHECK-NEXT:   ret {} undef

; CHECK: invertloop1:                                      ; preds = %invertloop2
; CHECK-NEXT:   %"X1'ipg" = getelementptr double, double* %"x'", i64 %"iv'ac.0"
; CHECK-NEXT:   %5 = load double, double* %"X1'ipg"
; CHECK-NEXT:   %6 = fadd fast double %5, %18
; CHECK-NEXT:   store double %6, double* %"X1'ipg"
; CHECK-NEXT:   %7 = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %7, label %invertentry, label %incinvertloop1

; CHECK: incinvertloop1:                                   ; preds = %invertloop1
; CHECK-NEXT:   %8 = sub nuw nsw i64 %"iv'ac.0", 1
; CHECK-NEXT:   br label %invertcleanup

; CHECK: invertloop2:                                      ; preds = %invertcleanup, %incinvertloop2
; CHECK-NEXT:   %"L1'de.0" = phi double [ 0.000000e+00, %invertcleanup ], [ %18, %incinvertloop2 ]
; CHECK-NEXT:   %"iv1'ac.0" = phi i64 [ 3, %invertcleanup ], [ %22, %incinvertloop2 ]
; CHECK-NEXT:   %9 = getelementptr i64, i64* %a17_malloccache, i64 %"iv'ac.0"
; CHECK-NEXT:   %10 = load i64, i64* %9, !invariant.load !0
; CHECK-NEXT:   %"tostore'ipg" = getelementptr double, double* %"z'", i64 %10
; CHECK-NEXT:   %11 = load double, double* %"tostore'ipg"
; CHECK-NEXT:   %"tostore'ipg4" = getelementptr double, double* %"z'", i64 %10
; CHECK-NEXT:   store double 0.000000e+00, double* %"tostore'ipg4"
; CHECK-NEXT:   %12 = mul nuw i64 %"iv1'ac.0", 4
; CHECK-NEXT:   %13 = add nuw i64 %"iv'ac.0", %12
; CHECK-NEXT:   %14 = getelementptr double, double* %L2_malloccache, i64 %13
; CHECK-NEXT:   %15 = load double, double* %14, !invariant.load !0
; CHECK-NEXT:   %m0diffeL1 = fmul fast double %11, %15
; CHECK-NEXT:   %16 = getelementptr double, double* %L1_malloccache, i64 %"iv'ac.0"
; CHECK-NEXT:   %17 = load double, double* %16, !invariant.load !0
; CHECK-NEXT:   %m1diffeL2 = fmul fast double %11, %17
; CHECK-NEXT:   %18 = fadd fast double %"L1'de.0", %m0diffeL1
; CHECK-NEXT:   %"X2'ipg" = getelementptr double, double* %"x'", i64 %"iv1'ac.0"
; CHECK-NEXT:   %19 = load double, double* %"X2'ipg"
; CHECK-NEXT:   %20 = fadd fast double %19, %m1diffeL2
; CHECK-NEXT:   store double %20, double* %"X2'ipg"
; CHECK-NEXT:   %21 = icmp eq i64 %"iv1'ac.0", 0
; CHECK-NEXT:   br i1 %21, label %invertloop1, label %incinvertloop2

; CHECK: incinvertloop2:                                   ; preds = %invertloop2
; CHECK-NEXT:   %22 = sub nuw nsw i64 %"iv1'ac.0", 1
; CHECK-NEXT:   br label %invertloop2

; CHECK: invertcleanup:                                    ; preds = %cleanup, %incinvertloop1
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %8, %incinvertloop1 ], [ 3, %cleanup ]
; CHECK-NEXT:   br label %invertloop2
; CHECK-NEXT: }
