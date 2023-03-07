; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -simplifycfg -adce -early-cse -S | FileCheck %s

source_filename = "/mnt/pci4/wmdata/Enzyme2/enzyme/test/Integration/ReverseMode/mycos.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare dso_local double @__enzyme_autodiff(i8*, double)

define dso_local double @ddd_mysin2(double %x) {
entry:
  %xp = alloca double, align 8
  store double %x, double* %xp
  %e83 = call double @mid(double %x, double* %xp)
  store double 0.000000e+00, double* %xp
  ret double %e83
}

define dso_local double @dddd_mysin2(double %x) {
entry:
  %call = call double @__enzyme_autodiff(i8* bitcast (double (double)* @ddd_mysin2 to i8*), double %x) #7
  ret double %call
}

define internal double @mid(double %x, double* %xp) {
entry:
  %tmp41 = load double, double* %xp, align 8
  br label %loop

loop:                                             ; preds = %loop, %entry
  %tiv = phi i64 [ %tiv.next, %loop ], [ 0, %entry ]
  %tmp24 = phi double [ 0.000000e+00, %entry ], [ %tmp42, %loop ]
  %tiv.next = add nuw nsw i64 %tiv, 1
  %tmp32 = add nsw i64 %tiv, 1
  %tmp42 = fadd fast double %tmp41, %tmp24
  %tmp27 = icmp eq i64 %tiv, 16
  br i1 %tmp27, label %exit, label %loop

exit:                                             ; preds = %loop
  %tmp60 = fmul fast double %tmp24, %x
  ret double %tmp60
}

; CHECK: define internal { double } @diffemid(double %x, double* %xp, double* %"xp'", double %differeturn, double* %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   br label %loop

; CHECK: loop:                                             ; preds = %loop, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %loop ], [ 0, %entry ]
; CHECK-NEXT:   %0 = getelementptr inbounds double, double* %tapeArg, i64 %iv
; CHECK-NEXT:   %tmp24 = load double, double* %0, align 8, !invariant.group !
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %tmp27 = icmp eq i64 %iv, 16
; CHECK-NEXT:   br i1 %tmp27, label %invertexit, label %loop

; CHECK: invertentry:                                      ; preds = %invertloop
; CHECK-NEXT:   %[[a0:.+]] = load double, double* %"xp'", align 8
; CHECK-NEXT:   %[[a1:.+]] = fadd fast double %[[a0]], %[[a3:.+]]
; CHECK-NEXT:   store double %[[a1]], double* %"xp'", align 8
; CHECK-NEXT:   %[[a2:.+]] = insertvalue { double } undef, double %m1diffex, 0
; CHECK-NEXT:   %4 = bitcast double* %tapeArg to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %4)
; CHECK-NEXT:   ret { double } %[[a2]]

; CHECK: invertloop:                                       ; preds = %invertexit, %incinvertloop
; CHECK-NEXT:   %"tmp24'de.0" = phi double [ %m0diffetmp24, %invertexit ], [ 0.000000e+00, %incinvertloop ]
; CHECK-NEXT:   %"tmp42'de.0" = phi double [ 0.000000e+00, %invertexit ], [ %[[a6:.+]], %incinvertloop ]
; CHECK-NEXT:   %"tmp41'de.0" = phi double [ 0.000000e+00, %invertexit ], [ %[[a3]], %incinvertloop ]
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ 16, %invertexit ], [ %[[a7:.+]], %incinvertloop ]
; CHECK-NEXT:   %[[a3]] = fadd fast double %"tmp41'de.0", %"tmp42'de.0"
; CHECK-NEXT:   %[[a4:.+]] = fadd fast double %"tmp24'de.0", %"tmp42'de.0"
; CHECK-NEXT:   %[[a5:.+]] = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   %[[a6]] = select{{( fast)?}} i1 %[[a5]], double 0.000000e+00, double %[[a4]]
; CHECK-NEXT:   br i1 %[[a5]], label %invertentry, label %incinvertloop

; CHECK: incinvertloop:                                    ; preds = %invertloop
; CHECK-NEXT:   %[[a7]] = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertloop

; CHECK: invertexit:                                       ; preds = %loop
; CHECK-NEXT:   %m0diffetmp24 = fmul fast double %differeturn, %x
; CHECK-NEXT:   %m1diffex = fmul fast double %differeturn, %tmp24
; CHECK-NEXT:   br label %invertloop
; CHECK-NEXT: }

