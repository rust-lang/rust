; RUN: if [ %llvmver -lt 14 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -early-cse -instcombine -simplifycfg -adce -S | FileCheck %s -check-prefixes LLVM13,SHARED; fi
; RUN: if [ %llvmver -ge 14 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -early-cse -instcombine -simplifycfg -adce -S | FileCheck %s -check-prefixes LLVM14,SHARED; fi


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

; SHARED: define internal void @diffecaller(double* %data, double* %"data'", i64* %a4, double %differeturn)
; SHARED-NEXT: entry:
; SHARED-NEXT:   %res_augmented = call { { i64, i64* }, double } @augmented_badfunc(double* %data, double* %"data'", i64* %a4)
; SHARED-NEXT:   %res = extractvalue { { i64, i64* }, double } %res_augmented, 1
; SHARED-NEXT:   call void @augmented_identity(double %res)
; SHARED-NEXT:   store i64 0, i64* %a4, align 4
; SHARED-NEXT:   store double 0.000000e+00, double* %data, align 8
; SHARED-NEXT:   %[[resev:.+]] = extractvalue { { i64, i64* }, double } %res_augmented, 0
; SHARED-NEXT:   store double 0.000000e+00, double* %"data'", align 8
; SHARED-NEXT:   %[[identr:.+]] = call { double } @diffeidentity(double %res, double %differeturn)
; SHARED-NEXT:   %[[iev:.+]] = extractvalue { double } %[[identr]], 0
; SHARED-NEXT:   call void @diffebadfunc(double* {{(nonnull )?}}%data, double* {{(nonnull )?}}%"data'", i64* {{(nonnull )?}}%a4, double %[[iev]], { i64, i64* } %[[resev]])
; SHARED-NEXT:   ret void
; SHARED-NEXT: }

; SHARED: define internal void @augmented_identity(double %res)
; SHARED-NEXT: entry:
; SHARED-NEXT:   ret void
; SHARED-NEXT: }

; SHARED: define internal { double } @diffeidentity(double %res, double %differeturn)
; SHARED-NEXT: entry:
; SHARED-NEXT:   %[[drt:.+]] = insertvalue { double } undef, double %differeturn, 0
; SHARED-NEXT:   ret { double } %[[drt]]
; SHARED-NEXT: }

; SHARED: define internal { { i64, i64* }, double } @augmented_badfunc(double* %data, double* %"data'", i64* %a4)
; SHARED-NEXT: entry:
; SHARED-NEXT:   %a5 = load i64, i64* %a4, align 4
; SHARED-NEXT:   %mallocsize = shl nuw nsw i64 %a5, 3
; SHARED-NEXT:   %malloccall = tail call noalias nonnull i8* @malloc(i64 %mallocsize)
; SHARED-NEXT:   %a19_malloccache = bitcast i8* %malloccall to i64*
; SHARED-NEXT:   br label %loop1

; SHARED: loop1:                                            ; preds = %exit, %entry
; SHARED-NEXT:   %iv = phi i64 [ %iv.next, %exit ], [ 0, %entry ]
; SHARED-NEXT:   %res1 = phi double [ %add, %exit ], [ 0.000000e+00, %entry ]
; SHARED-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; SHARED-NEXT:   %a19 = load i64, i64* %a4, align 4
; SHARED-NEXT:   %0 = getelementptr inbounds i64, i64* %a19_malloccache, i64 %iv
; SHARED-NEXT:   store i64 %a19, i64* %0, align 8, !invariant.group !
; SHARED-NEXT:   store i64 %iv.next, i64* %a4, align 4
; SHARED-NEXT:   br label %loop2

; SHARED: loop2:                                            ; preds = %loop2, %loop1
; SHARED-NEXT:   %iv1 = phi i64 [ %iv.next2, %loop2 ], [ 0, %loop1 ]
; SHARED-NEXT:   %res = phi double [ %add, %loop2 ], [ %res1, %loop1 ]
; SHARED-NEXT:   %iv.next2 = add nuw nsw i64 %iv1, 1
; SHARED-NEXT:   %gepk3 = getelementptr double, double* %data, i64 %iv1
; SHARED-NEXT:   %datak = load double, double* %gepk3, align 8
; SHARED-NEXT:   %add = fadd fast double %datak, %res
; SHARED-NEXT:   %exitcond3 = icmp eq i64 %iv.next2, %a19
; SHARED-NEXT:   br i1 %exitcond3, label %exit, label %loop2

; SHARED: exit:                                             ; preds = %loop2
; SHARED-NEXT:   store double %add, double* %data, align 8
; SHARED-NEXT:   %exitcond25 = icmp eq i64 %iv.next, %a5
; SHARED-NEXT:   br i1 %exitcond25, label %returner, label %loop1

; SHARED: returner:                                         ; preds = %exit
; SHARED-NEXT:   %.fca.0.0.insert = insertvalue { { i64, i64* }, double } {{(undef|poison)}}, i64 %a5, 0, 0
; SHARED-NEXT:   %.fca.0.1.insert = insertvalue { { i64, i64* }, double } %.fca.0.0.insert, i64* %a19_malloccache, 0, 1
; SHARED-NEXT:   %.fca.1.insert = insertvalue { { i64, i64* }, double } %.fca.0.1.insert, double %add, 1
; SHARED-NEXT:   ret { { i64, i64* }, double } %.fca.1.insert
; SHARED-NEXT: }

; SHARED: define internal void @diffebadfunc(double* %data, double* %"data'", i64* %a4, double %differeturn, { i64, i64* } %tapeArg)
; SHARED-NEXT: entry:
; SHARED-NEXT:   %[[a19cache:.+]] = extractvalue { i64, i64* } %tapeArg, 1
; SHARED-NEXT:   %a5 = extractvalue { i64, i64* } %tapeArg, 0
; SHARED-NEXT:   br label %loop1

; SHARED: loop1:                                            ; preds = %exit, %entry
; SHARED-NEXT:   %iv = phi i64 [ %iv.next, %exit ], [ 0, %entry ]
; SHARED-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; SHARED-NEXT:   %1 = getelementptr inbounds i64, i64* %[[a19cache]], i64 %iv
; SHARED-NEXT:   %a19 = load i64, i64* %1, align 8, !invariant.group !1
; SHARED-NEXT:   br label %loop2

; SHARED: loop2:                                            ; preds = %loop2, %loop1
; SHARED-NEXT:   %iv1 = phi i64 [ %iv.next2, %loop2 ], [ 0, %loop1 ]
; SHARED-NEXT:   %iv.next2 = add nuw nsw i64 %iv1, 1
; SHARED-NEXT:   %exitcond3 = icmp eq i64 %iv.next2, %a19
; SHARED-NEXT:   br i1 %exitcond3, label %exit, label %loop2

; SHARED: exit:                                             ; preds = %loop2
; SHARED-NEXT:   %exitcond25 = icmp eq i64 %iv.next, %a5
; SHARED-NEXT:   br i1 %exitcond25, label %invertexit, label %loop1

; SHARED: invertentry:                                      ; preds = %invertloop1
; SHARED-NEXT:   %[[free0:.+]] = bitcast i64* %[[a19cache]] to i8*
; SHARED-NEXT:   tail call void @free(i8* nonnull %[[free0]])
; SHARED-NEXT:   ret void

; SHARED: invertloop1:                                      ; preds = %invertloop2
; SHARED-NEXT:   %[[l1eq:.+]] = icmp eq i64 %"iv'ac.0", 0
; SHARED-NEXT:   br i1 %[[l1eq]], label %invertentry, label %incinvertloop1

; SHARED: incinvertloop1:                                   ; preds = %invertloop1
; SHARED-NEXT:   %[[p6:.+]] = fadd fast double %[[dadd:.+]], %[[dres:.+]]
; SHARED-NEXT:   br label %invertexit

; SHARED: invertloop2:
; SHARED-NEXT:   %"res1'de.0" = phi double [ 0.000000e+00, %invertexit ], [ %[[dres]], %invertloop2 ]
; SHARED-NEXT:   %"add'de.0" = phi double [ %[[add1p:.+]], %invertexit ], [ %[[dadd]], %invertloop2 ]
; SHARED-NEXT:   %"iv1'ac.0.in" = phi i64 [ %[[iv1p:.+]], %invertexit ], [ %"iv1'ac.0", %invertloop2 ]
; SHARED-NEXT:   %"iv1'ac.0" = add i64 %"iv1'ac.0.in", -1
; SHARED-NEXT:   %[[gepk3ipg:.+]] = getelementptr double, double* %"data'", i64 %"iv1'ac.0"
; SHARED-NEXT:   %[[linv:.+]] = load double, double* %[[gepk3ipg]], align 8
; SHARED-NEXT:   %[[tostore:.+]] = fadd fast double %[[linv]], %"add'de.0"
; SHARED-NEXT:   store double %[[tostore]], double* %[[gepk3ipg]], align 8
; SHARED-NEXT:   %[[eq:.+]] = icmp eq i64 %"iv1'ac.0", 0
; SHARED-NEXT:   %[[dadd]] = select{{( fast)?}} i1 %[[eq]], double 0.000000e+00, double %"add'de.0"
; LLVM13-NEXT:   %[[fdadd:.+]] = fadd fast double %"res1'de.0", %"add'de.0"
; LLVM14-NEXT:   %[[fdadd:.+]] = select {{(fast )?}}i1 %[[eq]], double %"add'de.0", double {{\-?}}0.000000e+00
; LLVM13-NEXT:   %[[dres]] = select{{( fast)?}} i1 %[[eq]], double %[[fdadd]], double %"res1'de.0"
; LLVM14-NEXT:   %[[dres]] = fadd fast double %"res1'de.0", %[[fdadd]]
; SHARED-NEXT:   br i1 %[[eq]], label %invertloop1, label %invertloop2

; SHARED: invertexit:                                       ; preds = %exit, %incinvertloop1
; SHARED-NEXT:   %"add'de.1" = phi double [ %[[p6]], %incinvertloop1 ], [ %differeturn, %exit ]
; SHARED-NEXT:   %"iv'ac.0.in" = phi i64 [ %"iv'ac.0", %incinvertloop1 ], [ %a5, %exit ]
; SHARED-NEXT:   %"iv'ac.0" = add i64 %"iv'ac.0.in", -1
; SHARED-NEXT:   %[[datap:.+]] = load double, double* %"data'", align 8
; SHARED-NEXT:   store double 0.000000e+00, double* %"data'", align 8
; SHARED-NEXT:   %[[mci:.+]] = getelementptr inbounds i64, i64* %[[a19cache]], i64 %"iv'ac.0"
; SHARED-NEXT:   %[[iv1p]] = load i64, i64* %[[mci]], align 8
; SHARED-NEXT:   %[[add1p]] = fadd fast double %"add'de.1", %[[datap]]
; SHARED-NEXT:   br label %invertloop2
; SHARED-NEXT: }
