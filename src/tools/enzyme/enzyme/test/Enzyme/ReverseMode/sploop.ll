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
  %L1 = load double, double* %X1, align 8
  br label %loop2

loop2:
  %j = phi i64 [ %j1, %loop2 ], [ 0, %loop1 ]
  %X2 = getelementptr inbounds double, double* %x, i64 %j
  %L2 = load double, double* %X2, align 8
  %why = fmul fast double %L1, %L2
  %tostore = getelementptr inbounds double, double* %z, i64 %a17
  store double %why, double* %tostore, align 8
  %j1 = add nuw nsw i64 %j, 1
  %exit2 = icmp eq i64 %j1, 4
  br i1 %exit2, label %cleanup, label %loop2

cleanup:
  %k1 = add nuw nsw i64 %k, 1
  %.pre = load i64, i64* %rows, align 8
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
; CHECK-NEXT:   %.pre_malloccache = bitcast i8* %[[malloccall3]] to i64*
; CHECK-NEXT:   %[[malloccall4:.+]] = tail call noalias nonnull dereferenceable(128) dereferenceable_or_null(128) i8* @malloc(i64 128)
; CHECK-NEXT:   %L2_malloccache = bitcast i8* %[[malloccall4]] to double*
; CHECK-NEXT:   %[[malloccall:.+]] = tail call noalias nonnull dereferenceable(32) dereferenceable_or_null(32) i8* @malloc(i64 32)
; CHECK-NEXT:   %L1_malloccache = bitcast i8* %[[malloccall]] to double*
; CHECK-NEXT:   br label %loop1
; CHECK: loop1:                                            ; preds = %cleanup, %entry
; CHECK-NEXT:   %iv = phi i64 [ 0, %entry ], [ %iv.next, %cleanup ]
; CHECK-NEXT:   %a17 = phi i64 [ 4, %entry ], [ %.pre, %cleanup ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %X1 = getelementptr inbounds double, double* %x, i64 %iv
; CHECK-NEXT:   %L1 = load double, double* %X1
; CHECK-NEXT:   %[[gepL1:.+]] = getelementptr inbounds double, double* %L1_malloccache, i64 %iv
; CHECK-NEXT:   store double %L1, double* %[[gepL1]], align 8, !invariant.group ![[g0:[0-9]+]]
; CHECK-NEXT:   br label %loop2

; CHECK: loop2:                                            ; preds = %loop2, %loop1
; CHECK-NEXT:   %iv1 = phi i64 [ %iv.next2, %loop2 ], [ 0, %loop1 ]
; CHECK-NEXT:   %iv.next2 = add nuw nsw i64 %iv1, 1
; CHECK-NEXT:   %X2 = getelementptr inbounds double, double* %x, i64 %iv1
; CHECK-NEXT:   %L2 = load double, double* %X2
; CHECK-NEXT:   %why = fmul fast double %L1, %L2
; CHECK-NEXT:   %tostore = getelementptr inbounds double, double* %z, i64 %a17
; CHECK-NEXT:   store double %why, double* %tostore
; CHECK-NEXT:   %[[i2:.+]] = mul nuw nsw i64 %iv, 4
; CHECK-NEXT:   %[[i3:.+]] = add nuw nsw i64 %iv1, %[[i2]]
; CHECK-NEXT:   %[[i4:.+]] = getelementptr inbounds double, double* %L2_malloccache, i64 %[[i3]]
; CHECK-NEXT:   store double %L2, double* %[[i4]], align 8, !invariant.group ![[g1:[0-9]+]]
; CHECK-NEXT:   %exit2 = icmp eq i64 %iv.next2, 4
; CHECK-NEXT:   br i1 %exit2, label %cleanup, label %loop2

; CHECK: cleanup:                                          ; preds = %loop2
; CHECK-NEXT:   %.pre = load i64, i64* %rows
; CHECK-NEXT:   %[[gepiv:.+]] = getelementptr inbounds i64, i64* %.pre_malloccache, i64 %iv
; CHECK-NEXT:   store i64 %.pre, i64* %[[gepiv]], align 8, !invariant.group ![[g2:[0-9]+]]
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
; CHECK-NEXT:   %6 = fadd fast double %5, %21
; CHECK-NEXT:   store double %6, double* %[[X1ipg]]
; CHECK-NEXT:   %7 = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %7, label %invertentry, label %incinvertloop1

; CHECK: incinvertloop1:                                   ; preds = %invertloop1
; CHECK-NEXT:   %8 = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertcleanup

; CHECK: invertloop2:                                      ; preds = %invertcleanup, %incinvertloop2
; CHECK-NEXT:   %"L1'de.0" = phi double [ 0.000000e+00, %invertcleanup ], [ %21, %incinvertloop2 ]
; CHECK-NEXT:   %"iv1'ac.0" = phi i64 [ 3, %invertcleanup ], [ %25, %incinvertloop2 ]
; CHECK-NEXT:   %9 = icmp ne i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %9, label %invertloop2_phirc, label %invertloop2_phimerge

; CHECK: invertloop2_phirc:                                ; preds = %invertloop2
; CHECK-NEXT:   %10 = sub nuw i64 %"iv'ac.0", 1
; CHECK-NEXT:   %11 = getelementptr inbounds i64, i64* %.pre_malloccache, i64 %10
; CHECK-NEXT:   %12 = load i64, i64* %11, align 8, !invariant.group ![[g2]]
; CHECK-NEXT:   br label %invertloop2_phimerge

; CHECK: invertloop2_phimerge:                             ; preds = %invertloop2, %invertloop2_phirc
; CHECK-NEXT:   %13 = phi i64 [ %12, %invertloop2_phirc ], [ 4, %invertloop2 ]
; CHECK-NEXT:   %"tostore'ipg_unwrap" = getelementptr inbounds double, double* %"z'", i64 %13
; CHECK-NEXT:   %14 = load double, double* %"tostore'ipg_unwrap", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"tostore'ipg_unwrap", align 8
; CHECK-NEXT:   %15 = mul nuw nsw i64 %"iv'ac.0", 4
; CHECK-NEXT:   %16 = add nuw nsw i64 %"iv1'ac.0", %15
; CHECK-NEXT:   %17 = getelementptr inbounds double, double* %L2_malloccache, i64 %16
; CHECK-NEXT:   %18 = load double, double* %17, align 8, !invariant.group ![[g1]]
; CHECK-NEXT:   %m0diffeL1 = fmul fast double %14, %18
; CHECK-NEXT:   %19 = getelementptr inbounds double, double* %L1_malloccache, i64 %"iv'ac.0"
; CHECK-NEXT:   %20 = load double, double* %19, align 8, !invariant.group ![[g0]]
; CHECK-NEXT:   %m1diffeL2 = fmul fast double %14, %20
; CHECK-NEXT:   %21 = fadd fast double %"L1'de.0", %m0diffeL1
; CHECK-NEXT:   %"X2'ipg_unwrap" = getelementptr inbounds double, double* %"x'", i64 %"iv1'ac.0"
; CHECK-NEXT:   %22 = load double, double* %"X2'ipg_unwrap", align 8
; CHECK-NEXT:   %23 = fadd fast double %22, %m1diffeL2
; CHECK-NEXT:   store double %23, double* %"X2'ipg_unwrap", align 8
; CHECK-NEXT:   %24 = icmp eq i64 %"iv1'ac.0", 0
; CHECK-NEXT:   br i1 %24, label %invertloop1, label %incinvertloop2

; CHECK: incinvertloop2:                                   ; preds = %invertloop2_phimerge
; CHECK-NEXT:   %25 = add nsw i64 %"iv1'ac.0", -1
; CHECK-NEXT:   br label %invertloop2

; CHECK: invertcleanup:                                    ; preds = %cleanup, %incinvertloop1
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %8, %incinvertloop1 ], [ 3, %cleanup ]
; CHECK-NEXT:   br label %invertloop2
; CHECK-NEXT: }
