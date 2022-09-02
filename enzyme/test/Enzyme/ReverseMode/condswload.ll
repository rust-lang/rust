; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -early-cse -simplifycfg -instsimplify -correlated-propagation -simplifycfg -adce -S | FileCheck %s

declare double @__enzyme_autodiff(i8*, ...)

; Function Attrs: norecurse nounwind uwtable
define double @alldiv(double* %a, i1 %cmp, i32 %val) {
entry:
  br i1 %cmp, label %mid, label %fin

mid:
  switch i32 %val, label %bdef [ i32 17, label %b1
                                 i32 42, label %b2 ]

b1: 
  %g1 = getelementptr inbounds double, double* %a, i32 32
  %l1 = load double, double* %g1, align 8
  br label %end

b2: 
  %g2 = getelementptr inbounds double, double* %a, i32 64
  %l2 = load double, double* %g2, align 8
  br label %end

bdef: 
  %g3 = getelementptr inbounds double, double* %a, i32 128
  %l3 = load double, double* %g3, align 8
  br label %end

end:
  %p = phi double [ %l1, %b1 ], [ %l2, %b2 ], [ %l3, %bdef ]
  %sq = fmul double %p, %p
  br label %fin

fin:
  %res = phi double [ 0.000000e+00, %entry ], [ %sq, %end ]
  ret double %res
}

define void @main(double* %a, double* %da, i1 %N, i32 %N2) {
entry:
  %call = call double (i8*, ...) @__enzyme_autodiff(i8* bitcast (double (double*, i1, i32)* @alldiv to i8*), double* nonnull %a, double* nonnull %da, i1 %N, i32 %N2)
  ret void
}

; CHECK: define internal void @diffealldiv(double* %a, double* %"a'", i1 %cmp, i32 %val, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = select {{(fast )?}}i1 %cmp, double %differeturn, double 0.000000e+00
; CHECK-NEXT:   br i1 %cmp, label %invertend, label %invertentry

; CHECK: invertentry:
; CHECK-NEXT:   ret void

; CHECK: invertb1:                                         ; preds = %invertend_phimerge
; CHECK-NEXT:   %"g1'ipg_unwrap" = getelementptr inbounds double, double* %"a'", i32 32
; CHECK-NEXT:   %1 = load double, double* %"g1'ipg_unwrap", align 8
; CHECK-NEXT:   %2 = fadd fast double %1, %14
; CHECK-NEXT:   store double %2, double* %"g1'ipg_unwrap", align 8
; CHECK-NEXT:   br label %invertentry

; CHECK: invertb2:                                         ; preds = %invertend_phimerge
; CHECK-NEXT:   %"g2'ipg_unwrap" = getelementptr inbounds double, double* %"a'", i32 64
; CHECK-NEXT:   %3 = load double, double* %"g2'ipg_unwrap", align 8
; CHECK-NEXT:   %4 = fadd fast double %3, %13
; CHECK-NEXT:   store double %4, double* %"g2'ipg_unwrap", align 8
; CHECK-NEXT:   br label %invertentry

; CHECK: invertbdef:                                       ; preds = %invertend_phimerge
; CHECK-NEXT:   %"g3'ipg_unwrap" = getelementptr inbounds double, double* %"a'", i32 128
; CHECK-NEXT:   %5 = load double, double* %"g3'ipg_unwrap", align 8
; CHECK-NEXT:   %6 = fadd fast double %5, %12
; CHECK-NEXT:   store double %6, double* %"g3'ipg_unwrap", align 8
; CHECK-NEXT:   br label %invertentry

; CHECK: invertend:                                        ; preds = %entry
; CHECK-NEXT:  switch i32 %val, label %invertend_phirc [
; CHECK-NEXT:    i32 17, label %invertend_phirc1
; CHECK-NEXT:    i32 42, label %invertend_phirc2
; CHECK-NEXT:  ]

; CHECK: invertend_phirc:                                  ; preds = %invertend
; CHECK-NEXT:   %g3_unwrap = getelementptr inbounds double, double* %a, i32 128
; CHECK-NEXT:   %l3_unwrap = load double, double* %g3_unwrap, align 8, !invariant.group !
; CHECK-NEXT:   br label %invertend_phimerge

; CHECK: invertend_phirc1:                                 ; preds = %invertend
; CHECK-NEXT:   %g1_unwrap = getelementptr inbounds double, double* %a, i32 32
; CHECK-NEXT:   %l1_unwrap = load double, double* %g1_unwrap, align 8, !invariant.group !
; CHECK-NEXT:   br label %invertend_phimerge

; CHECK: invertend_phirc2:                                 ; preds = %invertend
; CHECK-NEXT:   %g2_unwrap = getelementptr inbounds double, double* %a, i32 64
; CHECK-NEXT:   %l2_unwrap = load double, double* %g2_unwrap, align 8, !invariant.group !
; CHECK-NEXT:   br label %invertend_phimerge

; CHECK: invertend_phimerge: 
; CHECK-NEXT:   %7 = phi {{(fast )?}}double [ %l3_unwrap, %invertend_phirc ], [ %l1_unwrap, %invertend_phirc1 ], [ %l2_unwrap, %invertend_phirc2 ]
; CHECK-NEXT:   %m0diffep = fmul fast double %0, %7
; CHECK-NEXT:   %8 = fadd fast double %m0diffep, %m0diffep
; CHECK-NEXT:   %9 = icmp eq i32 17, %val
; CHECK-NEXT:   %10 = icmp eq i32 42, %val
; CHECK-NEXT:   %11 = or i1 %9, %10
; CHECK-NEXT:   %12 = select {{(fast )?}}i1 %11, double 0.000000e+00, double %8
; CHECK-NEXT:   %13 = select {{(fast )?}}i1 %10, double %8, double 0.000000e+00
; CHECK-NEXT:   %14 = select {{(fast )?}}i1 %9, double %8, double 0.000000e+00
; CHECK-NEXT:   switch i32 %val, label %invertbdef [
; CHECK-NEXT:     i32 17, label %invertb1
; CHECK-NEXT:     i32 42, label %invertb2
; CHECK-NEXT:   ]

