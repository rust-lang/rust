; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -simplifycfg -instsimplify -adce -S | FileCheck %s

declare double @__enzyme_autodiff(i8*, double)

declare noalias i8* @malloc(i64)








define dso_local double @_Z6foobard(double %t) {
entry:
  %malloccall = tail call i8* @malloc(i64 8)
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

define double @caller(double %inp) {
entry:
  %call = call double @__enzyme_autodiff(i8* bitcast (double (double)* @_Z6foobard to i8*), double %inp)
  ret double %call
}

; CHECK: define internal { double } @diffe_Z6foobard(double %t, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %malloccall = tail call noalias nonnull dereferenceable(8) dereferenceable_or_null(8) i8* @malloc(i64 8)
; CHECK-NEXT:   %"malloccall'mi" = tail call noalias nonnull dereferenceable(8) dereferenceable_or_null(8) i8* @malloc(i64 8)
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull dereferenceable(8) dereferenceable_or_null(8) %"malloccall'mi", i8 0, i64 8, i1 false)
; CHECK-NEXT:   %[[xipc:.+]] = bitcast i8* %"malloccall'mi" to double*
; CHECK-NEXT:   %x = bitcast i8* %malloccall to double*
; CHECK-NEXT:   %[[ipc:.+]] = bitcast i8* %"malloccall'mi" to i64*
; CHECK-NEXT:   %0 = bitcast i8* %malloccall to i64*
; CHECK-NEXT:   store i64 4607182418800017408, i64* %0
; CHECK-NEXT:   %div = fmul fast double %t, 1.000000e-02
; CHECK-NEXT:   %x.promoted = load double, double* %x, align 8
; CHECK-NEXT:   br label %while.body.i.i.i

; CHECK: while.body.i.i.i:
; CHECK-NEXT:   %[[phiload:.+]] = phi double* [ null, %entry ], [ %[[loadi1_realloccast:.+]], %[[mergeblk:.+]] ]
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %[[mergeblk]] ], [ 0, %entry ]
; CHECK-NEXT:   %load.i1 = phi double [ %x.promoted, %entry ], [ %add10.i.i.i, %[[mergeblk]] ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1

; CHECK-NEXT:   %[[phibc:.+]] = bitcast double* %[[phiload]] to i8*
; CHECK-NEXT:   %[[nexttrunc0:.+]] = and i64 %iv.next, 1
; CHECK-NEXT:   %[[nexttrunc:.+]] = icmp ne i64 %[[nexttrunc0]], 0
; CHECK-NEXT:   %[[popcnt:.+]] = call i64 @llvm.ctpop.i64(i64 %iv.next)
; CHECK-NEXT:   %[[le2:.+]] = icmp ult i64 %[[popcnt:.+]], 3
; CHECK-NEXT:   %[[shouldgrow:.+]] = and i1 %[[le2]], %[[nexttrunc]]
; CHECK-NEXT:   br i1 %[[shouldgrow]], label %grow.i, label %[[mergeblk]]

; CHECK: grow.i:                                           ; preds = %while.body.i.i.i
; CHECK-NEXT:   %[[ctlz:.+]] = call i64 @llvm.ctlz.i64(i64 %iv.next, i1 true)
; CHECK-NEXT:   %[[maxbit:.+]] = sub nuw nsw i64 64, %[[ctlz]]
; CHECK-NEXT:   %[[numbytes:.+]] = shl i64 8, %[[maxbit]]
; CHECK-NEXT:   %[[growalloc:.+]] = call i8* @realloc(i8* %[[phibc]], i64 %[[numbytes]])
; CHECK-NEXT:   br label %[[mergeblk]]

; CHECK: [[mergeblk]]:
; CHECK-NEXT:   %[[phiptr:.+]] = phi i8* [ %[[growalloc]], %grow.i ], [ %[[phibc]], %while.body.i.i.i ]
; CHECK-NEXT:   %[[loadi1_realloccast:.+]] = bitcast i8* %[[phiptr]] to double*
; CHECK-NEXT:   %[[tiv:.+]] = trunc i64 %iv to i32
; CHECK-NEXT:   %[[tostore:.+]] = fmul fast double %load.i1, 0xBFF3333333333332
; CHECK-NEXT:   %[[storeplace:.+]] = getelementptr inbounds double, double* %[[loadi1_realloccast]], i64 %iv
; CHECK-NEXT:   store double %[[tostore]], double* %[[storeplace]], align 8, !invariant.group ![[g0:[0-9]+]]
; CHECK-NEXT:   %reass.mul325.i = fmul fast double %[[tostore]], %div
; CHECK-NEXT:   %add10.i.i.i = fadd fast double %reass.mul325.i, %load.i1
; CHECK-NEXT:   %inc.i.i.i = add nuw nsw i32 %[[tiv]], 1
; CHECK-NEXT:   %conv8.i.i.i = sitofp i32 %inc.i.i.i to double
; CHECK-NEXT:   %mul.i.i.i = fmul fast double %div, %conv8.i.i.i
; CHECK-NEXT:   %add.i.i.i = fadd fast double %mul.i.i.i, %div
; CHECK-NEXT:   %sub.i.i.i.i = fsub fast double %add.i.i.i, %t
; CHECK-NEXT:   %cmp2.i.i.i.i = fcmp fast ugt double %sub.i.i.i.i, 0x3CB0000000000000
; CHECK-NEXT:   br i1 %cmp2.i.i.i.i, label %loopexit, label %while.body.i.i.i

; CHECK: loopexit:
; CHECK-NEXT:   store double %add10.i.i.i, double* %x, align 8
; CHECK-NEXT:   %[[a6:.+]] = load double, double* %[[xipc]], align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %[[xipc]], align 8
; CHECK-NEXT:   %[[a7:.+]] = fadd fast double
        ; add of these two (in any order) %[[a6]], %differeturn
        ; TODO figure out how to use filecheck to be order invariant of args
; CHECK-NEXT:   br label %invertwhile.body.i.i.i

; CHECK: invertentry:
; CHECK-NEXT:   %[[a8:.+]] = load double, double* %[[xipc]], align 8
; CHECK-NEXT:   %[[a9:.+]] = fadd fast double %[[a8]], %[[a17:.+]]
; CHECK-NEXT:   store double %[[a9]], double* %[[xipc]], align 8
; CHECK-NEXT:   %m0diffet = fmul fast double %[[a13:.+]], 1.000000e-02
; CHECK-NEXT:   store i64 0, i64* %[[ipc]], align 8
; CHECK-NEXT:   tail call void @free(i8* nonnull %"malloccall'mi")
; CHECK-NEXT:   tail call void @free(i8* nonnull %malloccall)
; CHECK-NEXT:   %[[a10:.+]] = insertvalue { double } undef, double %m0diffet, 0
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[phiptr]])
; CHECK-NEXT:   ret { double } %[[a10]]

; CHECK: invertwhile.body.i.i.i:
; CHECK-NEXT:   %"div'de.0" = phi double [ 0.000000e+00, %loopexit ], [ %[[a13]], %incinvertwhile.body.i.i.i ]
; CHECK-NEXT:   %"x.promoted'de.0" = phi double [ 0.000000e+00, %loopexit ], [ %[[a17]], %incinvertwhile.body.i.i.i ]
; CHECK-NEXT:   %"add10.i.i.i'de.0" = phi double [ %[[a7:.+]], %loopexit ], [ %[[p14:.+]], %incinvertwhile.body.i.i.i ]
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %iv, %loopexit ], [ %[[a18:.+]], %incinvertwhile.body.i.i.i ]
; CHECK-NEXT:   %m0diffe = fmul fast double %"add10.i.i.i'de.0", %div
; CHECK-NEXT:   %[[a11:.+]] = getelementptr inbounds double, double* %[[loadi1_realloccast]], i64 %"iv'ac.0"
; CHECK-NEXT:   %[[a12:.+]] = load double, double* %[[a11]], align 8, !invariant.group ![[g0]]
; CHECK-NEXT:   %m1diffediv = fmul fast double %"add10.i.i.i'de.0", %[[a12]]
; CHECK-NEXT:   %[[a13]] = fadd fast double %"div'de.0", %m1diffediv
; CHECK-NEXT:   %m0diffeload.i1 = fmul fast double %m0diffe, 0xBFF3333333333332
; CHECK-NEXT:   %[[a14:.+]] = fadd fast double %"add10.i.i.i'de.0", %m0diffeload.i1
; CHECK-NEXT:   %[[a15:.+]] = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   %[[p14]] = select{{( fast)?}} i1 %[[a15]], double 0.000000e+00, double %[[a14]]
; CHECK-NEXT:   %[[a16:.+]] = fadd fast double %"x.promoted'de.0", %[[a14]]
; CHECK-NEXT:   %[[a17]] = select{{( fast)?}} i1 %[[a15]], double %[[a16]], double %"x.promoted'de.0"
; CHECK-NEXT:   br i1 %[[a15]], label %invertentry, label %incinvertwhile.body.i.i.i

; CHECK: incinvertwhile.body.i.i.i:
; CHECK-NEXT:   %[[a18]] = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertwhile.body.i.i.i
; CHECK-NEXT: }
