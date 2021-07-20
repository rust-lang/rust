; For some weird reason this needs preopt on to cause a failure before the fix was introduced
; using the post preopt code and preopt off did not result in the failure
; RUN: if [ %llvmver -eq 7 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=true -mem2reg -sroa -simplifycfg -adce -early-cse -S | FileCheck %s; fi
source_filename = "/mnt/pci4/wmdata/Enzyme2/enzyme/test/Integration/ReverseMode/mycos.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare dso_local double @__enzyme_autodiff(i8*, double)

define dso_local double @ddd_mysin2(double %x) {
entry:
  %xp = alloca double, align 8
  store double %x, double* %xp
  %e83 = call double @mid(double %x, double* %xp, double* %xp)
  store double 0.000000e+00, double* %xp
  ret double %e83
}

define dso_local double @dddd_mysin2(double %x) {
entry:
  %call = call double @__enzyme_autodiff(i8* bitcast (double (double)* @ddd_mysin2 to i8*), double %x) #7
  ret double %call
}

define internal double @mid(double %x, double* %"'de16", double* %xp) {
entry:
  br label %loop

loop:                             ; preds = %invertincinvertfor.cond, %entry
  %tmp22 = phi i64 [ 0, %entry ], [ %tmp32, %loop ]
  %tmp24 = load double, double* %xp, align 8
  %tmp32 = add nsw i64 %tmp22, 1
  store double 3.000000e+00, double* %"'de16", align 8
  %tmp41 = load double, double* %xp, align 8
  %tmp42 = fadd fast double %tmp41, %tmp24
  store double %tmp42, double* %xp, align 8
  %tmp27 = icmp eq i64 %tmp22, 16
  br i1 %tmp27, label %exit, label %loop

exit:                   ; preds = %invertinvertfor.cond
  %tmp60 = fmul fast double %tmp24, %x
  ret double %tmp60
}

define internal double @pp_mid(double %x, double* %"'de16", double* %xp) {
entry:
  %tmp24.pre = load double, double* %xp, align 8
  br label %loop

loop:                                             ; preds = %loop, %entry
  %tiv = phi i64 [ %tiv.next, %loop ], [ 0, %entry ]
  %tmp24 = phi double [ %tmp24.pre, %entry ], [ %tmp42, %loop ]
  %tiv.next = add nuw nsw i64 %tiv, 1
  %tmp32 = add nsw i64 %tiv, 1
  store double 3.000000e+00, double* %"'de16", align 8
  %tmp41 = load double, double* %xp, align 8
  %tmp42 = fadd fast double %tmp41, %tmp24
  store double %tmp42, double* %xp, align 8
  %tmp27 = icmp eq i64 %tiv, 16
  br i1 %tmp27, label %exit, label %loop

exit:                                             ; preds = %loop
  %tmp60 = fmul fast double %tmp24, %x
  ret double %tmp60
}

; CHECK: define internal { double } @diffemid(double %x, double* %"'de16", double* %"'de16'", double* %xp, double* %"xp'", double %differeturn, double* %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   br label %loop

; CHECK: loop:                                             ; preds = %loop, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %loop ], [ 0, %entry ]
; CHECK-DAG:   %0 = getelementptr inbounds double, double* %tapeArg, i64 %iv
; CHECK-DAG:   %tmp24 = load double, double* %0, align 8, !invariant.group !1
; CHECK-DAG:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %tmp27 = icmp eq i64 %iv, 16
; CHECK-NEXT:   br i1 %tmp27, label %invertexit, label %loop

; CHECK: invertentry:                                      ; preds = %invertloop
; CHECK-DAG:   %[[a0:.+]] = load double, double* %"xp'", align 8
; CHECK-DAG:   %[[a1:.+]] = fadd fast double %[[a0]], %[[a3:.+]]
; CHECK-DAG:   store double %[[a1]], double* %"xp'", align 8
; CHECK-DAG:   %[[a2:.+]] = insertvalue { double } undef, double %m1diffex, 0
; CHECK-DAG:   %4 = bitcast double* %tapeArg to i8*
; CHECK-DAG:   tail call void @free(i8* nonnull %4)
; CHECK-DAG:   ret { double } %[[a2]]

; CHECK: invertloop:                                       ; preds = %invertexit, %incinvertloop
; CHECK-NEXT:   %"tmp24'de.0" = phi double [ %m0diffetmp24, %invertexit ], [ 0.000000e+00, %incinvertloop ]
; CHECK-NEXT:   %"tmp42'de.0" = phi double [ 0.000000e+00, %invertexit ], [ %9, %incinvertloop ]
; CHECK-NEXT:   %"tmp24.pre'de.0" = phi double [ 0.000000e+00, %invertexit ], [ %[[a3]], %incinvertloop ]
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ 16, %invertexit ], [ %[[a7:.+]], %incinvertloop ]
; CHECK-NEXT:   %5 = load double, double* %"xp'", align 8
; CHECK-NEXT:   %6 = fadd fast double %"tmp42'de.0", %5
; CHECK-NEXT:   %7 = fadd fast double %"tmp24'de.0", %6
; CHECK-NEXT:   store double %6, double* %"xp'", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"'de16'", align 8
; CHECK-NEXT:   %8 = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   %9 = select{{( fast)?}} i1 %8, double 0.000000e+00, double %7
; CHECK-NEXT:   %10 = fadd fast double %"tmp24.pre'de.0", %7
; CHECK-NEXT:   %[[a3]] = select{{( fast)?}} i1 %8, double %10, double %"tmp24.pre'de.0"
; CHECK-NEXT:   br i1 %8, label %invertentry, label %incinvertloop

; CHECK: incinvertloop:                                    ; preds = %invertloop
; CHECK-NEXT:   %[[a7]] = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertloop

; CHECK: invertexit:                                       ; preds = %loop
; CHECK-NEXT:   %m0diffetmp24 = fmul fast double %differeturn, %x
; CHECK-NEXT:   %m1diffex = fmul fast double %differeturn, %tmp24
; CHECK-NEXT:   br label %invertloop
; CHECK-NEXT: }
