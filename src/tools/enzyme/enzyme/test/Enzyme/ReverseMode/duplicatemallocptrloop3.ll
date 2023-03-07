; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -simplifycfg -instsimplify -adce -S | FileCheck %s

@.str = private unnamed_addr constant [28 x i8] c"original =%f derivative=%f\0A\00", align 1

define dso_local double* @f(double** nocapture readonly %a0) local_unnamed_addr #0 {
entry:
  %a2 = load double*, double** %a0, align 8
  ret double* %a2
}

define dso_local double @malloced(double* noalias %out, double* noalias %a0) local_unnamed_addr #1 {
entry: 
  %p2 = call noalias i8* @malloc(i32 8)
  br label %loop

loop:  
  %a9 = phi i32 [ 0, %entry ], [ %a14, %loop ]
  %p3 = bitcast i8* %p2 to double**
  %a10 = getelementptr inbounds double, double* %a0, i32 %a9
  store double* %a10, double** %p3, align 8
  %a4 = call double* @f(double** nonnull %p3)
  %r = load double, double* %a4
  %m2 = fmul double %r, %r
  %a13 = getelementptr inbounds double, double* %out, i32 %a9
  store double %m2, double* %a13, align 8
  %a14 = add nuw nsw i32 %a9, 1
  %a15 = icmp eq i32 %a14, 10
  br i1 %a15, label %exit, label %loop

exit:                                                ; preds = %8
  call void @free(i8* %p2)
  ret double 0.000000e+00
}

declare dso_local noalias i8* @malloc(i32) local_unnamed_addr #2

declare dso_local void @free(i8* nocapture) local_unnamed_addr #3

define dso_local void @derivative(double* %o, double* %do, double* %a0, double* %a1) local_unnamed_addr #4 {
entry:
  call void (i8*, ...) @__enzyme_autodiff(i8* bitcast (double (double*, double*)* @malloced to i8*), double* %o, double* %do, double* %a0, double* %a1)
  ret void
}

declare dso_local void @__enzyme_autodiff(i8*, ...)

attributes #0 = { noinline norecurse nounwind readonly }
attributes #1 = { nounwind }
attributes #2 = { inaccessiblememonly nounwind }
attributes #3 = { inaccessiblemem_or_argmemonly nounwind }
attributes #4 = { nounwind }
attributes #6 = { nounwind readonly }
attributes #7 = { nounwind }
attributes #9 = { nounwind }


; CHECK: define internal void @diffemalloced(double* noalias %out, double* %"out'", double* noalias %a0, double* %"a0'", double %differeturn) 
; CHECK-NEXT: entry:
; CHECK-NEXT:   %p2 = call noalias nonnull dereferenceable(8) dereferenceable_or_null(8) i8* @malloc(i32 8)
; CHECK-NEXT:   %"p2'mi" = call noalias nonnull dereferenceable(8) dereferenceable_or_null(8) i8* @malloc(i32 8)
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull dereferenceable(8) dereferenceable_or_null(8) %"p2'mi", i8 0, i64 8, i1 false)
; CHECK-NEXT:   %malloccall = tail call noalias nonnull dereferenceable(80) dereferenceable_or_null(80) i8* bitcast (i8* (i32)* @malloc to i8* (i64)*)(i64 80)
; CHECK-NEXT:   %r_malloccache = bitcast i8* %malloccall to double*
; CHECK-NEXT:   %malloccall4 = tail call noalias nonnull dereferenceable(80) dereferenceable_or_null(80) i8* bitcast (i8* (i32)* @malloc to i8* (i64)*)(i64 80)
; CHECK-NEXT:   %"a4'ip_phi_malloccache" = bitcast i8* %malloccall4 to double**
; CHECK-NEXT:   %malloccall8 = tail call noalias nonnull dereferenceable(80) dereferenceable_or_null(80) i8* bitcast (i8* (i32)* @malloc to i8* (i64)*)(i64 80)
; CHECK-NEXT:   %subcache_malloccache = bitcast i8* %malloccall8 to double**
; CHECK-NEXT:   br label %loop

; CHECK: loop:                                             ; preds = %loop, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %loop ], [ 0, %entry ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %0 = trunc i64 %iv to i32
; CHECK-NEXT:   %"p3'ipc" = bitcast i8* %"p2'mi" to double**
; CHECK-NEXT:   %p3 = bitcast i8* %p2 to double**
; CHECK-NEXT:   %"a10'ipg" = getelementptr inbounds double, double* %"a0'", i32 %0
; CHECK-NEXT:   %a10 = getelementptr inbounds double, double* %a0, i32 %0
; CHECK-NEXT:   store double* %"a10'ipg", double** %"p3'ipc", align 8
; CHECK-NEXT:   store double* %a10, double** %p3, align 8, !alias.scope !3, !noalias !0
; CHECK-NEXT:   %a4_augmented = call { double*, double*, double* } @augmented_f(double** %p3, double** %"p3'ipc")
; CHECK-NEXT:   %subcache = extractvalue { double*, double*, double* } %a4_augmented, 0
; CHECK-NEXT:   %a4 = extractvalue { double*, double*, double* } %a4_augmented, 1
; CHECK-NEXT:   %"a4'ac" = extractvalue { double*, double*, double* } %a4_augmented, 2
; CHECK-NEXT:   %r = load double, double* %a4
; CHECK-NEXT:   %m2 = fmul double %r, %r
; CHECK-NEXT:   %a13 = getelementptr inbounds double, double* %out, i32 %0
; CHECK-NEXT:   store double %m2, double* %a13, align 8
; CHECK-NEXT:   %1 = getelementptr inbounds double*, double** %subcache_malloccache, i64 %iv
; CHECK-NEXT:   store double* %subcache, double** %1, align 8
; CHECK-NEXT:   %2 = getelementptr inbounds double*, double** %"a4'ip_phi_malloccache", i64 %iv
; CHECK-NEXT:   store double* %"a4'ac", double** %2, align 8
; CHECK-NEXT:   %3 = getelementptr inbounds double, double* %r_malloccache, i64 %iv
; CHECK-NEXT:   store double %r, double* %3, align 8
; CHECK-NEXT:   %a14 = add nuw nsw i32 %0, 1
; CHECK-NEXT:   %a15 = icmp eq i32 %a14, 10
; CHECK-NEXT:   br i1 %a15, label %invertloop, label %loop

; CHECK: invertentry:                                      ; preds = %invertloop
; CHECK-NEXT:   tail call void @free(i8* nonnull %"p2'mi")
; CHECK-NEXT:   tail call void @free(i8* nonnull %p2)
; CHECK-NEXT:   tail call void @free(i8* nonnull %malloccall)
; CHECK-NEXT:   tail call void @free(i8* nonnull %malloccall4)
; CHECK-NEXT:   tail call void @free(i8* nonnull %malloccall8)
; CHECK-NEXT:   ret void

; CHECK: invertloop:                                       ; preds = %loop, %incinvertloop
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %15, %incinvertloop ], [ 9, %loop ]
; CHECK-NEXT:   %_unwrap = trunc i64 %"iv'ac.0" to i32
; CHECK-NEXT:   %"a13'ipg_unwrap" = getelementptr inbounds double, double* %"out'", i32 %_unwrap
; CHECK-NEXT:   %4 = load double, double* %"a13'ipg_unwrap", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"a13'ipg_unwrap", align 8
; CHECK-NEXT:   %5 = getelementptr inbounds double, double* %r_malloccache, i64 %"iv'ac.0"
; CHECK-NEXT:   %6 = load double, double* %5, align 8
; CHECK-NEXT:   %m0differ = fmul fast double %4, %6
; CHECK-NEXT:   %m1differ = fmul fast double %4, %6
; CHECK-NEXT:   %7 = fadd fast double %m0differ, %m1differ
; CHECK-NEXT:   %8 = getelementptr inbounds double*, double** %"a4'ip_phi_malloccache", i64 %"iv'ac.0"
; CHECK-NEXT:   %9 = load double*, double** %8, align 8
; CHECK-NEXT:   %10 = load double, double* %9
; CHECK-NEXT:   %11 = fadd fast double %10, %7
; CHECK-NEXT:   store double %11, double* %9
; CHECK-NEXT:   %p3_unwrap = bitcast i8* %p2 to double**
; CHECK-NEXT:   %"p3'ipc_unwrap" = bitcast i8* %"p2'mi" to double**
; CHECK-NEXT:   %12 = getelementptr inbounds double*, double** %subcache_malloccache, i64 %"iv'ac.0"
; CHECK-NEXT:   %13 = load double*, double** %12, align 8
; CHECK-NEXT:   call void @diffef(double** %p3_unwrap, double** %"p3'ipc_unwrap", double* %13)
; CHECK-NEXT:   %14 = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %14, label %invertentry, label %incinvertloop

; CHECK: incinvertloop:                                    ; preds = %invertloop
; CHECK-NEXT:   %15 = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertloop
; CHECK-NEXT: }
