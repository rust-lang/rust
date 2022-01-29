; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -adce -loop-deletion -correlated-propagation -simplifycfg -S | FileCheck %s

; This requires the memcpy optimization to run

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
  %0 = tail call double (void (double*, double*, i64*)*, ...) @__enzyme_autodiff(void (double*, double*, i64*)* nonnull @f, double* %x, double* %xp, double* %z, double* %zp, metadata !"enzyme_const", i64* %n)
  ret void
}

declare double @__enzyme_autodiff(void (double*, double*, i64*)*, ...)

; CHECK: define internal void @diffef(double* %x, double* %"x'", double* %z, double* %"z'", i64* %rows)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[malloccall3:.+]] = tail call noalias nonnull dereferenceable(32) dereferenceable_or_null(32) i8* @malloc(i64 32)
; CHECK-NEXT:   %a17_malloccache = bitcast i8* %[[malloccall3]] to i64*
; CHECK-NEXT:   %[[malloccall4:.+]] = tail call noalias nonnull dereferenceable(128) dereferenceable_or_null(128) i8* @malloc(i64 128)
; CHECK-NEXT:   %L2_malloccache = bitcast i8* %[[malloccall4]] to double*
; CHECK-NEXT:   %[[malloccall:.+]] = tail call noalias nonnull dereferenceable(32) dereferenceable_or_null(32) i8* @malloc(i64 32)
; CHECK-NEXT:   %L1_malloccache = bitcast i8* %[[malloccall]] to double*
; CHECK-NEXT:   br label %loop1
; CHECK: loop1:                                            ; preds = %cleanup, %entry
; CHECK-NEXT:   %iv = phi i64 [ 0, %entry ], [ %iv.next, %cleanup ]
; CHECK-NEXT:   %a17 = phi i64 [ 4, %entry ], [ %.pre, %cleanup ]
; CHECK-NEXT:   %[[gepiv:.+]] = getelementptr inbounds i64, i64* %a17_malloccache, i64 %iv
; CHECK-NEXT:   store i64 %a17, i64* %[[gepiv]], align 8, !invariant.group !0
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %X1 = getelementptr inbounds double, double* %x, i64 %iv
; CHECK-NEXT:   %L1 = load double, double* %X1
; CHECK-NEXT:   %[[gepL1:.+]] = getelementptr inbounds double, double* %L1_malloccache, i64 %iv
; CHECK-NEXT:   store double %L1, double* %[[gepL1]], align 8, !invariant.group !1
; CHECK-NEXT:   br label %loop2

; CHECK: loop2:                                            ; preds = %loop2, %loop1
; CHECK-NEXT:   %iv1 = phi i64 [ %iv.next2, %loop2 ], [ 0, %loop1 ]
; CHECK-NEXT:   %iv.next2 = add nuw nsw i64 %iv1, 1
; CHECK-NEXT:   %X2 = getelementptr inbounds double, double* %x, i64 %iv1
; CHECK-NEXT:   %L2 = load double, double* %X2
; CHECK-NEXT:   %why = fmul fast double %L1, %L2
; CHECK-NEXT:   %tostore = getelementptr inbounds double, double* %z, i64 %a17
; CHECK-NEXT:   store double %why, double* %tostore
; CHECK-NEXT:   %2 = mul nuw nsw i64 %iv, 4
; CHECK-NEXT:   %3 = add nuw nsw i64 %iv1, %2
; CHECK-NEXT:   %4 = getelementptr inbounds double, double* %L2_malloccache, i64 %3
; CHECK-NEXT:   store double %L2, double* %4, align 8, !invariant.group !2
; CHECK-NEXT:   %exit2 = icmp eq i64 %iv.next2, 4
; CHECK-NEXT:   br i1 %exit2, label %cleanup, label %loop2

; CHECK: cleanup:                                          ; preds = %loop2
; CHECK-NEXT:   %.pre = load i64, i64* %rows
; CHECK-NEXT:   %exit1 = icmp eq i64 %iv.next, 4
; CHECK-NEXT:   br i1 %exit1, label %invertcleanup, label %loop1

; CHECK: invertentry:                                      ; preds = %invertloop1
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[malloccall3]])
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[malloccall4]])
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[malloccall]])
; CHECK-NEXT:   ret void

; CHECK: invertloop1:                                      ; preds = %invertloop2
; CHECK-NEXT:   %[[X1ipg:.+]] = getelementptr inbounds double, double* %"x'", i64 %"iv'ac.0"
; CHECK-NEXT:   %5 = load double, double* %[[X1ipg]]
; CHECK-NEXT:   %6 = fadd fast double %5, %18
; CHECK-NEXT:   store double %6, double* %[[X1ipg]]
; CHECK-NEXT:   %7 = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %7, label %invertentry, label %incinvertloop1

; CHECK: incinvertloop1:                                   ; preds = %invertloop1
; CHECK-NEXT:   %8 = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertcleanup

; CHECK: invertloop2:                                      ; preds = %invertcleanup, %incinvertloop2
; CHECK-NEXT:   %"L1'de.0" = phi double [ 0.000000e+00, %invertcleanup ], [ %18, %incinvertloop2 ]
; CHECK-NEXT:   %"iv1'ac.0" = phi i64 [ 3, %invertcleanup ], [ %22, %incinvertloop2 ]
; CHECK-NEXT:   %9 = getelementptr inbounds i64, i64* %a17_malloccache, i64 %"iv'ac.0"
; CHECK-NEXT:   %10 = load i64, i64* %9, align 8, !invariant.group !0
; CHECK-NEXT:   %[[tostoreipg:.+]] = getelementptr inbounds double, double* %"z'", i64 %10
; CHECK-NEXT:   %11 = load double, double* %[[tostoreipg]]
; CHECK-NEXT:   store double 0.000000e+00, double* %[[tostoreipg]]
; CHECK-NEXT:   %12 = mul nuw nsw i64 %"iv'ac.0", 4
; CHECK-NEXT:   %13 = add nuw nsw i64 %"iv1'ac.0", %12
; CHECK-NEXT:   %14 = getelementptr inbounds double, double* %L2_malloccache, i64 %13
; CHECK-NEXT:   %15 = load double, double* %14, align 8, !invariant.group !2
; CHECK-NEXT:   %m0diffeL1 = fmul fast double %11, %15
; CHECK-NEXT:   %16 = getelementptr inbounds double, double* %L1_malloccache, i64 %"iv'ac.0"
; CHECK-NEXT:   %17 = load double, double* %16, align 8, !invariant.group !1
; CHECK-NEXT:   %m1diffeL2 = fmul fast double %11, %17
; CHECK-NEXT:   %18 = fadd fast double %"L1'de.0", %m0diffeL1
; CHECK-NEXT:   %[[X2ipg:.+]] = getelementptr inbounds double, double* %"x'", i64 %"iv1'ac.0"
; CHECK-NEXT:   %19 = load double, double* %[[X2ipg]]
; CHECK-NEXT:   %20 = fadd fast double %19, %m1diffeL2
; CHECK-NEXT:   store double %20, double* %[[X2ipg]]
; CHECK-NEXT:   %21 = icmp eq i64 %"iv1'ac.0", 0
; CHECK-NEXT:   br i1 %21, label %invertloop1, label %incinvertloop2

; CHECK: incinvertloop2:                                   ; preds = %invertloop2
; CHECK-NEXT:   %22 = add nsw i64 %"iv1'ac.0", -1
; CHECK-NEXT:   br label %invertloop2

; CHECK: invertcleanup:                                    ; preds = %cleanup, %incinvertloop1
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %8, %incinvertloop1 ], [ 3, %cleanup ]
; CHECK-NEXT:   br label %invertloop2
; CHECK-NEXT: }
