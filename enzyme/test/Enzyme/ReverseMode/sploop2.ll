; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -adce -loop-deletion -correlated-propagation -simplifycfg -S | FileCheck %s

; This requires the memcpy optimization to run

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local void @f(double* noalias %x, double* noalias %z, i64* %rows) {
entry:
  %.pre = load i64, i64* %rows
  br label %loop1

loop1:
  %a17 = phi i64 [ 4, %entry ], [ %.pre, %cleanup ]
  %k = phi i64 [ 0, %entry ], [ %k1, %cleanup ]
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
  %exit1 = icmp eq i64 %k1, 4
  br i1 %exit1, label %exit, label %loop1

exit:
  store double 1.000000e+00, double* %x
  ret void
}

define dso_local void @dsum(double* %x, double* %xp, double* %z, double* %zp, i64* %n, i64* %np) {
entry:
  %0 = tail call double (void (double*, double*, i64*)*, ...) @__enzyme_autodiff(void (double*, double*, i64*)* nonnull @f, double* %x, double* %xp, double* %z, double* %zp, metadata !"enzyme_const", i64* %n)
  ret void
}

declare double @__enzyme_autodiff(void (double*, double*, i64*)*, ...)

; CHECK: define internal void @diffef(double* noalias %x, double* %"x'", double* noalias %z, double* %"z'", i64* %rows)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %.pre = load i64, i64* %rows
; CHECK-NEXT:   %[[malloccall4:.+]] = tail call noalias nonnull dereferenceable(32) dereferenceable_or_null(32) i8* @malloc(i64 32)
; CHECK-NEXT:   %L2_malloccache = bitcast i8* %[[malloccall4]] to double*
; CHECK-NEXT:   %[[xptr:.+]] = bitcast double* %x to i8*
; CHECK-NEXT:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 %[[malloccall4]], i8* nonnull {{(align 8 )?}}%[[xptr:.+]], i64 32, i1 false)
; CHECK-NEXT:   %[[malloccall:.+]] = tail call noalias nonnull dereferenceable(32) dereferenceable_or_null(32) i8* @malloc(i64 32)
; CHECK-NEXT:   %L1_malloccache = bitcast i8* %[[malloccall]] to double*
; CHECK-NEXT:   %[[xptr2:.+]] = bitcast double* %x to i8*
; CHECK-NEXT:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 %[[malloccall]], i8* nonnull {{(align 8 )?}}%[[xptr2:.+]], i64 32, i1 false)
; CHECK-NEXT:   br label %loop1

; CHECK: loop1:                                            ; preds = %cleanup, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %cleanup ], [ 0, %entry ]
; CHECK-NEXT:   %a17 = phi i64 [ 4, %entry ], [ %.pre, %cleanup ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %X1 = getelementptr inbounds double, double* %x, i64 %iv
; CHECK-NEXT:   %L1 = load double, double* %X1
; CHECK-NEXT:   br label %loop2

; CHECK: loop2:                                            ; preds = %loop2, %loop1
; CHECK-NEXT:   %iv1 = phi i64 [ %iv.next2, %loop2 ], [ 0, %loop1 ]
; CHECK-NEXT:   %iv.next2 = add nuw nsw i64 %iv1, 1
; CHECK-NEXT:   %X2 = getelementptr inbounds double, double* %x, i64 %iv1
; CHECK-NEXT:   %L2 = load double, double* %X2
; CHECK-NEXT:   %why = fmul fast double %L1, %L2
; CHECK-NEXT:   %tostore = getelementptr inbounds double, double* %z, i64 %a17
; CHECK-NEXT:   store double %why, double* %tostore
; CHECK-NEXT:   %exit2 = icmp eq i64 %iv.next2, 4
; CHECK-NEXT:   br i1 %exit2, label %cleanup, label %loop2

; CHECK: cleanup:                                          ; preds = %loop2
; CHECK-NEXT:   %exit1 = icmp eq i64 %iv.next, 4
; CHECK-NEXT:   br i1 %exit1, label %exit, label %loop1

; CHECK: exit:                                             ; preds = %cleanup
; CHECK-NEXT:   store double 1.000000e+00, double* %x
; CHECK-NEXT:   store double 0.000000e+00, double* %"x'"
; CHECK-NEXT:   br label %invertcleanup

; CHECK: invertentry:                                      ; preds = %invertloop1
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[malloccall4]])
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[malloccall]])
; CHECK-NEXT:   ret void

; CHECK: invertloop1:                                      ; preds = %invertloop2
; CHECK-NEXT:   %[[X1ipg:.+]] = getelementptr inbounds double, double* %"x'", i64 %"iv'ac.0"
; CHECK-NEXT:   %[[a5:.+]] = load double, double* %[[X1ipg]]
; CHECK-NEXT:   %[[a6:.+]] = fadd fast double %[[a5]], %[[a18:.+]]
; CHECK-NEXT:   store double %[[a6]], double* %[[X1ipg]]
; CHECK-NEXT:   %[[a7:.+]] = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %[[a7]], label %invertentry, label %incinvertloop1

; CHECK: incinvertloop1:                                   ; preds = %invertloop1
; CHECK-NEXT:   %[[a8:.+]] = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertcleanup

; CHECK: invertloop2:                                      ; preds = %invertcleanup, %incinvertloop2
; CHECK-NEXT:   %"L1'de.0" = phi double [ 0.000000e+00, %invertcleanup ], [ %[[a18]], %incinvertloop2 ]
; CHECK-NEXT:   %"iv1'ac.0" = phi i64 [ 3, %invertcleanup ], [ %[[a22:.+]], %incinvertloop2 ]
; CHECK-NEXT:   %[[pcmp:.+]] = icmp ne i64 %"iv'ac.0", 0
; CHECK-NEXT:   %[[re_a17:.+]] = select i1 %[[pcmp]], i64 %.pre, i64 4
; CHECK-NEXT:   %[[tostoreipg:.+]] = getelementptr inbounds double, double* %"z'", i64 %[[re_a17]]
; CHECK-NEXT:   %[[a11:.+]] = load double, double* %[[tostoreipg]]
; CHECK-NEXT:   store double 0.000000e+00, double* %[[tostoreipg]]
; CHECK-NEXT:   %[[rgep:.+]] = getelementptr inbounds double, double* %L2_malloccache, i64 %"iv1'ac.0"
; CHECK-NEXT:   %[[a15:.+]] = load double, double* %[[rgep]], align 8, !invariant.group !
; CHECK-NEXT:   %m0diffeL1 = fmul fast double %[[a11]], %[[a15]]
; CHECK-NEXT:   %[[a16:.+]] = getelementptr inbounds double, double* %L1_malloccache, i64 %"iv'ac.0"
; CHECK-NEXT:   %[[a17:.+]] = load double, double* %[[a16]], align 8, !invariant.group !
; CHECK-NEXT:   %m1diffeL2 = fmul fast double %[[a11]], %[[a17]]
; CHECK-NEXT:   %[[a18]] = fadd fast double %"L1'de.0", %m0diffeL1
; CHECK-NEXT:   %[[X2ipg:.+]] = getelementptr inbounds double, double* %"x'", i64 %"iv1'ac.0"
; CHECK-NEXT:   %[[a19:.+]] = load double, double* %[[X2ipg]]
; CHECK-NEXT:   %[[a20:.+]] = fadd fast double %[[a19]], %m1diffeL2
; CHECK-NEXT:   store double %[[a20]], double* %[[X2ipg]]
; CHECK-NEXT:   %[[a21:.+]] = icmp eq i64 %"iv1'ac.0", 0
; CHECK-NEXT:   br i1 %[[a21]], label %invertloop1, label %incinvertloop2

; CHECK: incinvertloop2:                                   ; preds = %invertloop2
; CHECK-NEXT:   %[[a22]] = add nsw i64 %"iv1'ac.0", -1
; CHECK-NEXT:   br label %invertloop2

; CHECK: invertcleanup:
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ 3, %exit ], [ %[[a8]], %incinvertloop1 ]
; CHECK-NEXT:   br label %invertloop2
; CHECK-NEXT: }
