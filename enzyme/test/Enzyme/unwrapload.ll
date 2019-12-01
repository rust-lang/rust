; RUN: %opt < %s %loadEnzyme -enzyme -enzyme_preopt=false -mem2reg -sroa -early-cse -instcombine -simplifycfg -S | FileCheck %s

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
  %gepk3 = getelementptr inbounds double, double* %data, i64 %k
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

; CHECK: define internal {} @diffecaller(double* %data, double* %"data'", i64* %a4, double %differeturn) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %res_augmented = call { { i64, i64*, double** }, double } @augmented_badfunc(double* %data, double* %"data'", i64* %a4)
; CHECK-NEXT:   %0 = extractvalue { { i64, i64*, double** }, double } %res_augmented, 1
; CHECK-NEXT:   %res2_augmented = call { {} } @augmented_identity(double %0)
; CHECK-NEXT:   store i64 0, i64* %a4, align 4
; CHECK-NEXT:   store double 0.000000e+00, double* %data, align 8
; CHECK-NEXT:   %1 = extractvalue { { i64, i64*, double** }, double } %res_augmented, 0
; CHECK-NEXT:   store double 0.000000e+00, double* %"data'", align 8
; CHECK-NEXT:   %2 = call { double } @diffeidentity(double %0, double %differeturn, {} undef)
; CHECK-NEXT:   %3 = extractvalue { double } %2, 0
; CHECK-NEXT:   %4 = call {} @diffebadfunc(double* %data, double* %"data'", i64* %a4, double %3, { i64, i64*, double** } %1)
; CHECK-NEXT:   ret {} undef
; CHECK-NEXT: }

; CHECK: define internal { {} } @augmented_identity(double %res) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret { {} } undef
; CHECK-NEXT: }

; CHECK: define internal { double } @diffeidentity(double %res, double %differeturn, {} %tapeArg) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = insertvalue { double } undef, double %differeturn, 0
; CHECK-NEXT:   ret { double } %0
; CHECK-NEXT: }

; CHECK: define internal { { i64, i64*, double** }, double } @augmented_badfunc(double* %data, double* %"data'", i64* %a4) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %a5 = load i64, i64* %a4, align 4
; CHECK-NEXT:   %mallocsize = shl i64 %a5, 3
; CHECK-NEXT:   %malloccall = tail call noalias nonnull i8* @malloc(i64 %mallocsize)
; CHECK-NEXT:   %a19_malloccache = bitcast i8* %malloccall to i64*
; CHECK-NEXT:   %malloccall4 = tail call noalias nonnull i8* @malloc(i64 %mallocsize)
; CHECK-NEXT:   %datak_malloccache = bitcast i8* %malloccall4 to double**
; CHECK-NEXT:   br label %loop1

; CHECK: loop1:                                            ; preds = %exit, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %exit ], [ 0, %entry ]
; CHECK-NEXT:   %res1 = phi double [ %add, %exit ], [ 0.000000e+00, %entry ]
; CHECK-NEXT:   %iv.next = add nuw i64 %iv, 1
; CHECK-NEXT:   %a19 = load i64, i64* %a4, align 4
; CHECK-NEXT:   %0 = getelementptr i64, i64* %a19_malloccache, i64 %iv
; CHECK-NEXT:   store i64 %a19, i64* %0, align 8, !invariant.group !0
; CHECK-NEXT:   store i64 %iv.next, i64* %a4, align 4
; CHECK-NEXT:   %1 = getelementptr double*, double** %datak_malloccache, i64 %iv
; CHECK-NEXT:   %mallocsize5 = shl i64 %a19, 3
; CHECK-NEXT:   %malloccall6 = tail call noalias nonnull i8* @malloc(i64 %mallocsize5)
; CHECK-NEXT:   %2 = bitcast double** %1 to i8**
; CHECK-NEXT:   store i8* %malloccall6, i8** %2, align 8
; CHECK-NEXT:   br label %loop2

; CHECK: loop2:                                            ; preds = %loop2, %loop1
; CHECK-NEXT:   %iv1 = phi i64 [ %iv.next2, %loop2 ], [ 0, %loop1 ]
; CHECK-NEXT:   %res = phi double [ %add, %loop2 ], [ %res1, %loop1 ]
; CHECK-NEXT:   %iv.next2 = add nuw i64 %iv1, 1
; CHECK-NEXT:   %gepk3 = getelementptr inbounds double, double* %data, i64 %iv1
; CHECK-NEXT:   %datak = load double, double* %gepk3, align 8
; CHECK-NEXT:   %3 = load double*, double** %1, align 8, !dereferenceable !1, !invariant.group !2
; CHECK-NEXT:   %4 = getelementptr double, double* %3, i64 %iv1
; CHECK-NEXT:   store double %datak, double* %4, align 8, !invariant.group !3
; CHECK-NEXT:   %add = fadd fast double %datak, %res
; CHECK-NEXT:   %exitcond3 = icmp eq i64 %iv.next2, %a19
; CHECK-NEXT:   br i1 %exitcond3, label %exit, label %loop2

; CHECK: exit:                                             ; preds = %loop2
; CHECK-NEXT:   store double %add, double* %data, align 8
; CHECK-NEXT:   %exitcond25 = icmp eq i64 %iv.next, %a5
; CHECK-NEXT:   br i1 %exitcond25, label %returner, label %loop1

; CHECK: returner:                                         ; preds = %exit
; CHECK-NEXT:   %.fca.0.0.insert = insertvalue { { i64, i64*, double** }, double } undef, i64 %a5, 0, 0
; CHECK-NEXT:   %.fca.0.1.insert = insertvalue { { i64, i64*, double** }, double } %.fca.0.0.insert, i64* %a19_malloccache, 0, 1
; CHECK-NEXT:   %.fca.0.2.insert = insertvalue { { i64, i64*, double** }, double } %.fca.0.1.insert, double** %datak_malloccache, 0, 2
; CHECK-NEXT:   %.fca.1.insert = insertvalue { { i64, i64*, double** }, double } %.fca.0.2.insert, double %add, 1
; CHECK-NEXT:   ret { { i64, i64*, double** }, double } %.fca.1.insert
; CHECK-NEXT: }

; CHECK: define internal {} @diffebadfunc(double* %data, double* %"data'", i64* %a4, double %differeturn, { i64, i64*, double** } %tapeArg) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = extractvalue { i64, i64*, double** } %tapeArg, 1
; CHECK-NEXT:   %1 = extractvalue { i64, i64*, double** } %tapeArg, 2
; CHECK-NEXT:   %a5 = extractvalue { i64, i64*, double** } %tapeArg, 0
; CHECK-NEXT:   %mallocsize = shl i64 %a5, 3
; CHECK-NEXT:   %malloccall = tail call noalias nonnull i8* @malloc(i64 %mallocsize)
; CHECK-NEXT:   %_malloccache = bitcast i8* %malloccall to i64*
; CHECK-NEXT:   br label %loop1

; CHECK: loop1:                                            ; preds = %exit, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %exit ], [ 0, %entry ]
; CHECK-NEXT:   %res1 = phi double [ %add, %exit ], [ 0.000000e+00, %entry ]
; CHECK-NEXT:   %iv.next = add nuw i64 %iv, 1
; CHECK-NEXT:   %2 = getelementptr i64, i64* %0, i64 %iv
; CHECK-NEXT:   %a19 = load i64, i64* %2, align 8, !invariant.group !4
; CHECK-NEXT:   %3 = add i64 %a19, -1
; CHECK-NEXT:   %4 = getelementptr i64, i64* %_malloccache, i64 %iv
; CHECK-NEXT:   store i64 %3, i64* %4, align 8, !invariant.group !5
; CHECK-NEXT:   br label %loop2

; CHECK: loop2:                                            ; preds = %loop2, %loop1
; CHECK-NEXT:   %iv1 = phi i64 [ %iv.next2, %loop2 ], [ 0, %loop1 ]
; CHECK-NEXT:   %res = phi double [ %add, %loop2 ], [ %res1, %loop1 ]
; CHECK-NEXT:   %iv.next2 = add nuw i64 %iv1, 1
; CHECK-NEXT:   %5 = getelementptr double*, double** %1, i64 %iv
; CHECK-NEXT:   %6 = load double*, double** %5, align 8, !dereferenceable !1
; CHECK-NEXT:   %7 = getelementptr double, double* %6, i64 %iv1
; CHECK-NEXT:   %datak = load double, double* %7, align 8, !invariant.group !6
; CHECK-NEXT:   %add = fadd fast double %datak, %res
; CHECK-NEXT:   %exitcond3 = icmp eq i64 %iv.next2, %a19
; CHECK-NEXT:   br i1 %exitcond3, label %exit, label %loop2

; CHECK: exit:                                             ; preds = %loop2
; CHECK-NEXT:   %exitcond25 = icmp eq i64 %iv.next, %a5
; CHECK-NEXT:   br i1 %exitcond25, label %invertexit, label %loop1

; CHECK: invertentry:                                      ; preds = %invertloop1
; CHECK-NEXT:   tail call void @free(i8* nonnull %malloccall)
; CHECK-NEXT:   %8 = bitcast double** %1 to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %8)
; CHECK-NEXT:   %9 = bitcast i64* %0 to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %9)
; CHECK-NEXT:   ret {} undef

; CHECK: invertloop1:                                      ; preds = %invertloop2
; CHECK-NEXT:   %_unwrap = getelementptr double*, double** %1, i64 %"iv'ac.0"
; CHECK-NEXT:   %10 = bitcast double** %_unwrap to i8**
; CHECK-NEXT:   %11 = load i8*, i8** %10, align 8, !dereferenceable !1
; CHECK-NEXT:   tail call void @free(i8* nonnull %11)
; CHECK-NEXT:   %12 = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %12, label %invertentry, label %incinvertloop1

; CHECK: incinvertloop1:                                   ; preds = %invertloop1
; CHECK-NEXT:   %13 = fadd fast double %17, %19
; CHECK-NEXT:   br label %invertexit

; CHECK: invertloop2:                                      ; preds = %invertexit, %incinvertloop2
; CHECK-NEXT:   %"res1'de.0" = phi double [ 0.000000e+00, %invertexit ], [ %19, %incinvertloop2 ]
; CHECK-NEXT:   %"add'de.0" = phi double [ %24, %invertexit ], [ %17, %incinvertloop2 ]
; CHECK-NEXT:   %"iv1'ac.0" = phi i64 [ %23, %invertexit ], [ %20, %incinvertloop2 ]
; CHECK-NEXT:   %"gepk3'ipg" = getelementptr double, double* %"data'", i64 %"iv1'ac.0"
; CHECK-NEXT:   %14 = load double, double* %"gepk3'ipg", align 8
; CHECK-NEXT:   %15 = fadd fast double %14, %"add'de.0"
; CHECK-NEXT:   store double %15, double* %"gepk3'ipg", align 8
; CHECK-NEXT:   %16 = icmp eq i64 %"iv1'ac.0", 0
; CHECK-NEXT:   %17 = select i1 %16, double 0.000000e+00, double %"add'de.0"
; CHECK-NEXT:   %18 = fadd fast double %"res1'de.0", %"add'de.0"
; CHECK-NEXT:   %19 = select i1 %16, double %18, double %"res1'de.0"
; CHECK-NEXT:   br i1 %16, label %invertloop1, label %incinvertloop2

; CHECK: incinvertloop2:                                   ; preds = %invertloop2
; CHECK-NEXT:   %20 = add nsw i64 %"iv1'ac.0", -1
; CHECK-NEXT:   br label %invertloop2

; CHECK: invertexit:                                       ; preds = %exit, %incinvertloop1
; CHECK-NEXT:   %"add'de.1" = phi double [ %13, %incinvertloop1 ], [ %differeturn, %exit ]
; CHECK-NEXT:   %"iv'ac.0.in" = phi i64 [ %"iv'ac.0", %incinvertloop1 ], [ %a5, %exit ]
; CHECK-NEXT:   %"iv'ac.0" = add i64 %"iv'ac.0.in", -1
; CHECK-NEXT:   %21 = load double, double* %"data'", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"data'", align 8
; CHECK-NEXT:   %22 = getelementptr i64, i64* %_malloccache, i64 %"iv'ac.0"
; CHECK-NEXT:   %23 = load i64, i64* %22, align 8, !invariant.group !5
; CHECK-NEXT:   %24 = fadd fast double %"add'de.1", %21
; CHECK-NEXT:   br label %invertloop2
; CHECK-NEXT: }
