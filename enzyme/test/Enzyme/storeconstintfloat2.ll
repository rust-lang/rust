; RUN: %opt < %s %loadEnzyme -enzyme -enzyme_preopt=false -mem2reg -sroa -simplifycfg -instcombine -adce -S | FileCheck %s

define double @caller(double %inp) {
entry:
  %call = call double @__enzyme_autodiff(i8* bitcast (double (double)* @_Z6foobard to i8*), double %inp)
  ret double %call
}

declare double @__enzyme_autodiff(i8*, double)

declare noalias i8* @malloc(i64)

; Function Attrs: nounwind uwtable
define dso_local double @_Z6foobard(double %t) {
entry:
  %malloccall = tail call i8* @malloc(i64 8) #4
  %x = bitcast i8* %malloccall to double*
  %0 = bitcast i8* %malloccall to i64*
  ; this is storing double 1.0, but represented as an integer
  ;   such an occurance can happen as a consequence of an optimization
  store i64 4607182418800017408, i64* %0, align 8
  %div = fmul fast double %t, 1.000000e-02
  %x.promoted = load double, double* %x, align 8
  br label %while.body.i.i.i

while.body.i.i.i:                                 ; preds = %while.body.i.i.i, %entry
  %load.i1 = phi double [ %x.promoted, %entry ], [ %add10.i.i.i, %while.body.i.i.i ]
  %step.029.i.i.i = phi i32 [ 0, %entry ], [ %inc.i.i.i, %while.body.i.i.i ]
  %1 = fmul fast double %load.i1, 0xBFF3333333333332
  %reass.mul325.i = fmul fast double %1, %div
  %add10.i.i.i = fadd fast double %reass.mul325.i, %load.i1
  %inc.i.i.i = add nuw nsw i32 %step.029.i.i.i, 1
  %conv8.i.i.i = sitofp i32 %inc.i.i.i to double
  %mul.i.i.i = fmul fast double %div, %conv8.i.i.i
  %add.i.i.i = fadd fast double %mul.i.i.i, %div
  %sub.i.i.i.i = fsub fast double %add.i.i.i, %t
  %cmp2.i.i.i.i = fcmp fast ugt double %sub.i.i.i.i, 0x3CB0000000000000
  br i1 %cmp2.i.i.i.i, label %loopexit, label %while.body.i.i.i

loopexit:                                         ; preds = %while.body.i.i.i
  store double %add10.i.i.i, double* %x, align 8
  ret double %add10.i.i.i
}

; CHECK: define internal { double } @diffe_Z6foobard(double %t, double %differeturn) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"malloccall'mi" = tail call noalias nonnull dereferenceable(8) dereferenceable_or_null(8) i8* @malloc(i64 8)
; CHECK-NEXT:   %0 = bitcast i8* %"malloccall'mi" to i64*
; CHECK-NEXT:   store i64 0, i64* %0, align 1
; CHECK-NEXT:   %[[xipc:.+]] = bitcast i8* %"malloccall'mi" to double*
; CHECK-NEXT:   %[[ipc:.+]] = bitcast i8* %"malloccall'mi" to i64*
; CHECK-NEXT:   %div = fmul fast double %t, 1.000000e-02
; CHECK-NEXT:   br label %while.body.i.i.i

; CHECK: while.body.i.i.i:
; CHECK-NEXT:   %1 = phi i8* [ null, %entry ], [ %[[loadi1_realloccache:.+]], %while.body.i.i.i ]
; CHECK-NEXT:   %iv = phi i64 [ 0, %entry ], [ %iv.next, %while.body.i.i.i ]
; CHECK-NEXT:   %load.i1 = phi double [ 1.000000e+00, %entry ], [ %add10.i.i.i, %while.body.i.i.i ]
; CHECK-NEXT:   %2 = trunc i64 %iv to i32
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %3 = shl nuw nsw i64 %iv.next, 3
; CHECK-NEXT:   %[[loadi1_realloccache]] = call i8* @realloc(i8* %1, i64 %3)
; CHECK-NEXT:   %[[loadi1_realloccast:.+]] = bitcast i8* %[[loadi1_realloccache]] to double*
; CHECK-NEXT:   %4 = fmul fast double %load.i1, 0xBFF3333333333332
; CHECK-NEXT:   %5 = getelementptr inbounds double, double* %[[loadi1_realloccast]], i64 %iv
; CHECK-NEXT:   store double %4, double* %5, align 8, !invariant.group !0
; CHECK-NEXT:   %reass.mul325.i = fmul fast double %4, %div
; CHECK-NEXT:   %add10.i.i.i = fadd fast double %reass.mul325.i, %load.i1
; CHECK-NEXT:   %inc.i.i.i = add nuw nsw i32 %2, 1
; CHECK-NEXT:   %conv8.i.i.i = sitofp i32 %inc.i.i.i to double
; CHECK-NEXT:   %mul.i.i.i = fmul fast double %div, %conv8.i.i.i
; CHECK-NEXT:   %add.i.i.i = fadd fast double %mul.i.i.i, %div
; CHECK-NEXT:   %sub.i.i.i.i = fsub fast double %add.i.i.i, %t
; CHECK-NEXT:   %cmp2.i.i.i.i = fcmp fast ugt double %sub.i.i.i.i, 0x3CB0000000000000
; CHECK-NEXT:   br i1 %cmp2.i.i.i.i, label %loopexit, label %while.body.i.i.i

; CHECK: loopexit:
; CHECK-NEXT:   %6 = load double, double* %[[xipc]], align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %[[xipc]], align 8
; CHECK-NEXT:   %7 = fadd fast double %6, %differeturn
; CHECK-NEXT:   br label %invertwhile.body.i.i.i

; CHECK: invertentry:
; CHECK-NEXT:   %8 = load double, double* %[[xipc]], align 8
; CHECK-NEXT:   %9 = fadd fast double %8, %17
; CHECK-NEXT:   store double %9, double* %[[xipc]], align 8
; CHECK-NEXT:   %m0diffet = fmul fast double %13, 1.000000e-02
; CHECK-NEXT:   store i64 0, i64* %[[ipc]], align 8
; CHECK-NEXT:   tail call void @free(i8* nonnull %"malloccall'mi")
; CHECK-NEXT:   %10 = insertvalue { double } undef, double %m0diffet, 0
; CHECK-NEXT:   tail call void @free(i8* nonnull %_realloccache)
; CHECK-NEXT:   ret { double } %10

; CHECK: invertwhile.body.i.i.i:
; CHECK-NEXT:   %"div'de.0" = phi double [ 0.000000e+00, %loopexit ], [ %13, %incinvertwhile.body.i.i.i ]
; CHECK-NEXT:   %"x.promoted'de.0" = phi double [ 0.000000e+00, %loopexit ], [ %17, %incinvertwhile.body.i.i.i ]
; CHECK-NEXT:   %"add10.i.i.i'de.0" = phi double [ %7, %loopexit ], [ %14, %incinvertwhile.body.i.i.i ]
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %iv, %loopexit ], [ %18, %incinvertwhile.body.i.i.i ]
; CHECK-NEXT:   %m0diffe = fmul fast double %"add10.i.i.i'de.0", %div
; CHECK-NEXT:   %11 = getelementptr inbounds double, double* %[[loadi1_realloccast]], i64 %"iv'ac.0"
; CHECK-NEXT:   %12 = load double, double* %11, align 8, !invariant.group !0
; CHECK-NEXT:   %m1diffediv = fmul fast double %"add10.i.i.i'de.0", %12
; CHECK-NEXT:   %13 = fadd fast double %"div'de.0", %m1diffediv
; CHECK-NEXT:   %m0diffeload.i1 = fmul fast double %m0diffe, 0xBFF3333333333332
; CHECK-NEXT:   %14 = fadd fast double %"add10.i.i.i'de.0", %m0diffeload.i1
; CHECK-NEXT:   %15 = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   %16 = fadd fast double %"x.promoted'de.0", %14
; CHECK-NEXT:   %17 = select{{( fast)?}} i1 %15, double %16, double %"x.promoted'de.0"
; CHECK-NEXT:   br i1 %15, label %invertentry, label %incinvertwhile.body.i.i.i

; CHECK: incinvertwhile.body.i.i.i:
; CHECK-NEXT:   %18 = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertwhile.body.i.i.i
; CHECK-NEXT: }

attributes #4 = { nounwind }
