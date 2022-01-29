; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -early-cse -instcombine -simplifycfg -adce -S | FileCheck %s

define double @caller(double* %data, i64* %a4) {
entry:
  %res = call double @badfunc(double* %data, i64* %a4)
  %res2 = call double @identity(double %res)
  store i64 0, i64* %a4
  store double 0.000000e+00, double* %data
  ret double %res2
}

define double @identity(double %res) {
entry:
  ret double %res
}

define double @badfunc(double* %data, i64* %a4) {
entry:
  %a5 = load i64, i64* %a4
  br label %loop1

loop1:
  %i = phi i64 [ 0, %entry ], [ %next1, %exit ]
  %res1 = phi double [ 0.000000e+00, %entry ], [ %add, %exit ]
  %next1 = add nuw nsw i64 %i, 1
  %a19 = load i64, i64* %a4
  store i64 %next1, i64* %a4
  br label %loop2

loop2:
  %k = phi i64 [ %nextk, %loop2 ], [ 0, %loop1 ]
  %res = phi double [ %add, %loop2 ], [ %res1, %loop1 ]
  %nextk = add nuw nsw i64 %k, 1
  %gepk3 = getelementptr double, double* %data, i64 %k
  %datak = load double, double* %gepk3
  %add = fadd fast double %datak, %res
  %exitcond3 = icmp eq i64 %nextk, %a19
  br i1 %exitcond3, label %exit, label %loop2

exit:
  store double %add, double* %data
  %exitcond25 = icmp eq i64 %next1, %a5
  br i1 %exitcond25, label %returner, label %loop1

returner:
  ret double %add
}

define dso_local void @derivative(double* %this, double* %dthis, i64* %xpr) {
  %call11 = call fast double @__enzyme_autodiff(i8* bitcast (double (double*, i64*)* @caller to i8*), double* %this, double* %dthis, i64* %xpr)
  ret void
}

declare dso_local double @__enzyme_autodiff(i8*, double*, double*, i64*)

; CHECK: define internal void @diffecaller(double* %data, double* %"data'", i64* %a4, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %res_augmented = call { { i64, i64* }, double } @augmented_badfunc(double* %data, double* %"data'", i64* %a4)
; CHECK-NEXT:   %res = extractvalue { { i64, i64* }, double } %res_augmented, 1
; CHECK-NEXT:   call void @augmented_identity(double %res)
; CHECK-NEXT:   store i64 0, i64* %a4, align 4
; CHECK-NEXT:   store double 0.000000e+00, double* %data, align 8
; CHECK-NEXT:   %[[resev:.+]] = extractvalue { { i64, i64* }, double } %res_augmented, 0
; CHECK-NEXT:   store double 0.000000e+00, double* %"data'", align 8
; CHECK-NEXT:   %[[identr:.+]] = call { double } @diffeidentity(double %res, double %differeturn)
; CHECK-NEXT:   %[[iev:.+]] = extractvalue { double } %[[identr]], 0
; CHECK-NEXT:   call void @diffebadfunc(double* {{(nonnull )?}}%data, double* {{(nonnull )?}}%"data'", i64* {{(nonnull )?}}%a4, double %[[iev]], { i64, i64* } %[[resev]])
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @augmented_identity(double %res)
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal { double } @diffeidentity(double %res, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[drt:.+]] = insertvalue { double } undef, double %differeturn, 0
; CHECK-NEXT:   ret { double } %[[drt]]
; CHECK-NEXT: }

; CHECK: define internal { { i64, i64* }, double } @augmented_badfunc(double* %data, double* %"data'", i64* %a4)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %a5 = load i64, i64* %a4, align 4
; CHECK-NEXT:   %mallocsize = shl nuw nsw i64 %a5, 3
; CHECK-NEXT:   %malloccall = tail call noalias nonnull i8* @malloc(i64 %mallocsize)
; CHECK-NEXT:   %a19_malloccache = bitcast i8* %malloccall to i64*
; CHECK-NEXT:   br label %loop1

; CHECK: loop1:                                            ; preds = %exit, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %exit ], [ 0, %entry ]
; CHECK-NEXT:   %res1 = phi double [ %add, %exit ], [ 0.000000e+00, %entry ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %a19 = load i64, i64* %a4, align 4
; CHECK-NEXT:   %0 = getelementptr inbounds i64, i64* %a19_malloccache, i64 %iv
; CHECK-NEXT:   store i64 %a19, i64* %0, align 8, !invariant.group !0
; CHECK-NEXT:   store i64 %iv.next, i64* %a4, align 4
; CHECK-NEXT:   br label %loop2

; CHECK: loop2:                                            ; preds = %loop2, %loop1
; CHECK-NEXT:   %iv1 = phi i64 [ %iv.next2, %loop2 ], [ 0, %loop1 ]
; CHECK-NEXT:   %res = phi double [ %add, %loop2 ], [ %res1, %loop1 ]
; CHECK-NEXT:   %iv.next2 = add nuw nsw i64 %iv1, 1
; CHECK-NEXT:   %gepk3 = getelementptr double, double* %data, i64 %iv1
; CHECK-NEXT:   %datak = load double, double* %gepk3, align 8
; CHECK-NEXT:   %add = fadd fast double %datak, %res
; CHECK-NEXT:   %exitcond3 = icmp eq i64 %iv.next2, %a19
; CHECK-NEXT:   br i1 %exitcond3, label %exit, label %loop2

; CHECK: exit:                                             ; preds = %loop2
; CHECK-NEXT:   store double %add, double* %data, align 8
; CHECK-NEXT:   %exitcond25 = icmp eq i64 %iv.next, %a5
; CHECK-NEXT:   br i1 %exitcond25, label %returner, label %loop1

; CHECK: returner:                                         ; preds = %exit
; CHECK-NEXT:   %.fca.0.0.insert = insertvalue { { i64, i64* }, double } undef, i64 %a5, 0, 0
; CHECK-NEXT:   %.fca.0.1.insert = insertvalue { { i64, i64* }, double } %.fca.0.0.insert, i64* %a19_malloccache, 0, 1
; CHECK-NEXT:   %.fca.1.insert = insertvalue { { i64, i64* }, double } %.fca.0.1.insert, double %add, 1
; CHECK-NEXT:   ret { { i64, i64* }, double } %.fca.1.insert
; CHECK-NEXT: }

; CHECK: define internal void @diffebadfunc(double* %data, double* %"data'", i64* %a4, double %differeturn, { i64, i64* } %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[a19cache:.+]] = extractvalue { i64, i64* } %tapeArg, 1
; CHECK-NEXT:   %a5 = extractvalue { i64, i64* } %tapeArg, 0
; CHECK-NEXT:   br label %loop1

; CHECK: loop1:                                            ; preds = %exit, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %exit ], [ 0, %entry ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %1 = getelementptr inbounds i64, i64* %[[a19cache]], i64 %iv
; CHECK-NEXT:   %a19 = load i64, i64* %1, align 8, !invariant.group !1
; CHECK-NEXT:   br label %loop2

; CHECK: loop2:                                            ; preds = %loop2, %loop1
; CHECK-NEXT:   %iv1 = phi i64 [ %iv.next2, %loop2 ], [ 0, %loop1 ]
; CHECK-NEXT:   %iv.next2 = add nuw nsw i64 %iv1, 1
; CHECK-NEXT:   %exitcond3 = icmp eq i64 %iv.next2, %a19
; CHECK-NEXT:   br i1 %exitcond3, label %exit, label %loop2

; CHECK: exit:                                             ; preds = %loop2
; CHECK-NEXT:   %exitcond25 = icmp eq i64 %iv.next, %a5
; CHECK-NEXT:   br i1 %exitcond25, label %invertexit, label %loop1

; CHECK: invertentry:                                      ; preds = %invertloop1
; CHECK-NEXT:   %[[free0:.+]] = bitcast i64* %[[a19cache]] to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[free0]])
; CHECK-NEXT:   ret void

; CHECK: invertloop1:                                      ; preds = %invertloop2
; CHECK-NEXT:   %[[l1eq:.+]] = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %[[l1eq]], label %invertentry, label %incinvertloop1

; CHECK: incinvertloop1:                                   ; preds = %invertloop1
; CHECK-NEXT:   %[[p6:.+]] = fadd fast double %[[dadd:.+]], %[[dres:.+]]
; CHECK-NEXT:   br label %invertexit

; CHECK: invertloop2:
; CHECK-NEXT:   %"res1'de.0" = phi double [ 0.000000e+00, %invertexit ], [ %[[dres]], %invertloop2 ]
; CHECK-NEXT:   %"add'de.0" = phi double [ %[[add1p:.+]], %invertexit ], [ %[[dadd]], %invertloop2 ]
; CHECK-NEXT:   %"iv1'ac.0.in" = phi i64 [ %[[iv1p:.+]], %invertexit ], [ %"iv1'ac.0", %invertloop2 ]
; CHECK-NEXT:   %"iv1'ac.0" = add i64 %"iv1'ac.0.in", -1
; CHECK-NEXT:   %[[gepk3ipg:.+]] = getelementptr double, double* %"data'", i64 %"iv1'ac.0"
; CHECK-NEXT:   %[[linv:.+]] = load double, double* %[[gepk3ipg]], align 8
; CHECK-NEXT:   %[[tostore:.+]] = fadd fast double %[[linv]], %"add'de.0"
; CHECK-NEXT:   store double %[[tostore]], double* %[[gepk3ipg]], align 8
; CHECK-NEXT:   %[[eq:.+]] = icmp eq i64 %"iv1'ac.0", 0
; CHECK-NEXT:   %[[dadd]] = select{{( fast)?}} i1 %[[eq]], double 0.000000e+00, double %"add'de.0"
; CHECK-NEXT:   %[[fdadd:.+]] = fadd fast double %"res1'de.0", %"add'de.0"
; CHECK-NEXT:   %[[dres]] = select{{( fast)?}} i1 %[[eq]], double %[[fdadd]], double %"res1'de.0"
; CHECK-NEXT:   br i1 %[[eq]], label %invertloop1, label %invertloop2

; CHECK: invertexit:                                       ; preds = %exit, %incinvertloop1
; CHECK-NEXT:   %"add'de.1" = phi double [ %[[p6]], %incinvertloop1 ], [ %differeturn, %exit ]
; CHECK-NEXT:   %"iv'ac.0.in" = phi i64 [ %"iv'ac.0", %incinvertloop1 ], [ %a5, %exit ]
; CHECK-NEXT:   %"iv'ac.0" = add i64 %"iv'ac.0.in", -1
; CHECK-NEXT:   %[[datap:.+]] = load double, double* %"data'", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"data'", align 8
; CHECK-NEXT:   %[[mci:.+]] = getelementptr inbounds i64, i64* %[[a19cache]], i64 %"iv'ac.0"
; CHECK-NEXT:   %[[iv1p]] = load i64, i64* %[[mci]], align 8
; CHECK-NEXT:   %[[add1p]] = fadd fast double %"add'de.1", %[[datap]]
; CHECK-NEXT:   br label %invertloop2
; CHECK-NEXT: }
