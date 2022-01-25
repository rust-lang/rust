; TODO handle llvm 13
; RUN: if [ %llvmver -lt 13 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -simplifycfg -early-cse -adce -S | FileCheck %s; fi
; ModuleID = 'inp.ll'

declare dso_local void @_Z17__enzyme_autodiffPvPdS0_i(i8*, double*, double*, i64*) local_unnamed_addr #4
define dso_local void @outer(double* %m, double* %m2, i64* %n) local_unnamed_addr #2 {
entry:
  call void @_Z17__enzyme_autodiffPvPdS0_i(i8* bitcast (double (double*, i64*)* @_Z10reduce_maxPdi to i8*), double* nonnull %m, double* nonnull %m2, i64* %n)
  ret void
}
; Function Attrs: nounwind uwtable
define dso_local double @_Z10reduce_maxPdi(double* %vec, i64* %v) #0 {
entry:
  %res = call double @pb(double* %vec, i64* %v)
  store double 0.000000e+00, double* %vec, align 8
  store i64 0, i64* %v, align 8
  %bc = bitcast i64* %v to i8*
  call void @llvm.memset.p0i8.i64(i8* %bc, i8 0, i64 128, i1 false)
  ret double %res
}

declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i1)

define double @pb(double* %x, i64* %sptr) {
entry:
  %step = load i64, i64* %sptr, align 8, !tbaa !6
  br label %for.body

for.body:                                         ; preds = %for.cond.loopexit, %entry
  %i2.0245 = phi i64 [ 0, %entry ], [ %add, %for.cond.loopexit ]
  %add = add nuw nsw i64 %i2.0245, 1
  br label %for.body59

for.body59:                                       ; preds = %for.body59, %for.body
  %k2.0243 = phi i64 [ %add61, %for.body59 ], [ 0, %for.body ]
  %add61 = add nuw nsw i64 %k2.0243, %step
  call void @inner(double* %x)
  %cmp57 = icmp slt i64 %add61, 100
  br i1 %cmp57, label %for.body59, label %for.cond.loopexit

for.cond.loopexit:                                ; preds = %for.body59
  %cmp53 = icmp slt i64 %add, 56
  br i1 %cmp53, label %for.body, label %_ZN5Eigen8internal28aligned_stack_memory_handlerIdED2Ev.exit

_ZN5Eigen8internal28aligned_stack_memory_handlerIdED2Ev.exit: ; preds = %for.cond.loopexit
  ret double 0.000000e+00
}

; Function Attrs: nounwind uwtable
define void @inner(double* %blockA) unnamed_addr #3 align 2 {
entry:
  %ld = load double, double* %blockA, align 8
  %mul = fmul fast double %ld, %ld
  store double %mul, double* %blockA, align 8
  ret void
}

!4 = !{!5, i64 1, !"omnipotent char"}
!5 = !{!"Simple C++ TBAA"}
!6 = !{!7, !7, i64 0, i64 8}
!7 = !{!4, i64 8, !"long"}

attributes #0 = { readnone speculatable }

; CHECK: define internal { double*, i64 } @augmented_pb(double* %x, double* %"x'", i64* %sptr)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %step = load i64, i64* %sptr, align 8, !tbaa !0
; CHECK-NEXT:   %[[_unwrap:.+]] = udiv i64 99, %step
; CHECK-NEXT:   %[[a0:.+]] = add nuw i64 %[[_unwrap]], 1
; CHECK-NEXT:   %[[a1:.+]] = mul nuw nsw i64 %[[a0]], 56
; CHECK-NEXT:   %mallocsize = mul nuw nsw i64 %[[a1]], 8
; CHECK-NEXT:   %malloccall = tail call noalias nonnull i8* @malloc(i64 %mallocsize)
; CHECK-NEXT:   %_augmented_malloccache = bitcast i8* %malloccall to double*
; CHECK-NEXT:   br label %for.body

; CHECK: for.body:                                         ; preds = %for.cond.loopexit, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %for.cond.loopexit ], [ 0, %entry ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   br label %for.body59

; CHECK: for.body59:                                       ; preds = %for.body59, %for.body
; CHECK-NEXT:   %iv1 = phi i64 [ %iv.next2, %for.body59 ], [ 0, %for.body ]
; CHECK-NEXT:   %iv.next2 = add nuw nsw i64 %iv1, 1
; CHECK-NEXT:   %[[a3:.+]] = mul i64 {{(%iv1, %step|%step, %iv1)}}
; CHECK-NEXT:   %add61 = add nuw nsw i64 %[[a3]], %step
; CHECK-NEXT:   %_augmented = call fast double @augmented_inner(double* %x, double* %"x'")
; CHECK-NEXT:   %[[a5:.+]] = mul nuw nsw i64 %iv, %[[a0]]
; CHECK-NEXT:   %[[a6:.+]] = add nuw nsw i64 %iv1, %[[a5]]
; CHECK-NEXT:   %[[a7:.+]] = getelementptr inbounds double, double* %_augmented_malloccache, i64 %[[a6]]
; CHECK-NEXT:   store double %_augmented, double* %[[a7:.+]], align 8, !invariant.group !
; CHECK-NEXT:   %cmp57 = icmp slt i64 %add61, 100
; CHECK-NEXT:   br i1 %cmp57, label %for.body59, label %for.cond.loopexit

; CHECK: for.cond.loopexit:                                ; preds = %for.body59
; CHECK-NEXT:   %cmp53 = icmp ne i64 %iv.next, 56
; CHECK-NEXT:   br i1 %cmp53, label %for.body, label %_ZN5Eigen8internal28aligned_stack_memory_handlerIdED2Ev.exit

; CHECK: _ZN5Eigen8internal28aligned_stack_memory_handlerIdED2Ev.exit: ; preds = %for.cond.loopexit
; CHECK-NEXT:   %.fca.0.insert = insertvalue { double*, i64 } undef, double* %_augmented_malloccache, 0
; CHECK-NEXT:   %.fca.1.insert = insertvalue { double*, i64 } %.fca.0.insert, i64 %step, 1
; CHECK-NEXT:   ret { double*, i64 } %.fca.1.insert
; CHECK-NEXT: }

; CHECK: define internal void @diffepb(double* %x, double* %"x'", i64* %sptr, double %differeturn, { double*, i64 } %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = extractvalue { double*, i64 } %tapeArg, 0
; CHECK-NEXT:   %step = extractvalue { double*, i64 } %tapeArg, 1
; CHECK-NEXT:   %[[_unwrap:.+]] = udiv i64 99, %step
; CHECK-NEXT:   %[[a1:.+]] = add nuw i64 %[[_unwrap]], 1
; CHECK-NEXT:   br label %for.body

; CHECK: for.body:                                         ; preds = %for.cond.loopexit, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %for.cond.loopexit ], [ 0, %entry ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   br label %for.body59

; CHECK: for.body59:                                       ; preds = %for.body59, %for.body
; CHECK-NEXT:   %iv1 = phi i64 [ %iv.next2, %for.body59 ], [ 0, %for.body ]
; CHECK-NEXT:   %iv.next2 = add nuw nsw i64 %iv1, 1
; CHECK-NEXT:   %[[a4:.+]] = mul i64 {{(%iv1, %step|%step, %iv1)}}
; CHECK-NEXT:   %add61 = add nuw nsw i64 %[[a4]], %step
; CHECK-NEXT:   %cmp57 = icmp slt i64 %add61, 100
; CHECK-NEXT:   br i1 %cmp57, label %for.body59, label %for.cond.loopexit

; CHECK: for.cond.loopexit:                                ; preds = %for.body59
; CHECK-NEXT:   %cmp53 = icmp ne i64 %iv.next, 56
; CHECK-NEXT:   br i1 %cmp53, label %for.body, label %invertfor.cond.loopexit

; CHECK: invertentry:                                      ; preds = %invertfor.body
; CHECK-NEXT:   %[[tofree:.+]] = bitcast double* %0 to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[tofree]])
; CHECK-NEXT:   ret void

; CHECK: invertfor.body:                                   ; preds = %invertfor.body59
; CHECK-NEXT:   %[[cmpf:.+]] = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %[[cmpf]], label %invertentry, label %incinvertfor.body

; CHECK: incinvertfor.body:                                ; preds = %invertfor.body
; CHECK-NEXT:   %[[a12:.+]] = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertfor.cond.loopexit

; CHECK: invertfor.body59:                                 ; preds = %invertfor.cond.loopexit, %incinvertfor.body59
; CHECK-NEXT:   %"iv1'ac.0" = phi i64 [ %[[_unwrap:.+]], %invertfor.cond.loopexit ], [ %[[a15:.+]], %incinvertfor.body59 ]
; CHECK-NEXT:   %[[_unwrap5:.+]] = mul nuw nsw i64 %"iv'ac.0", %[[a1]]
; CHECK-NEXT:   %[[_unwrap6:.+]] = add nuw nsw i64 %"iv1'ac.0", %[[_unwrap5]]
; CHECK-NEXT:   %[[_unwrap7:.+]] = getelementptr inbounds double, double* %0, i64 %[[_unwrap6]]
; TODO make the invariant group here the same in the augmented forward
; CHECK-NEXT:   %[[tapeArg3_unwrap:.+]] = load double, double* %[[_unwrap7:.+]], align 8, !invariant.group !
; CHECK-NEXT:   call void @diffeinner(double* %x, double* %"x'", double %[[tapeArg3_unwrap]])
; CHECK-NEXT:   %[[a13:.+]] = icmp eq i64 %"iv1'ac.0", 0
; CHECK-NEXT:   br i1 %[[a13]], label %invertfor.body, label %incinvertfor.body59

; CHECK: incinvertfor.body59:                              ; preds = %invertfor.body59
; CHECK-NEXT:   %[[a15]] = add nsw i64 %"iv1'ac.0", -1
; CHECK-NEXT:   br label %invertfor.body59

; CHECK: invertfor.cond.loopexit:                          ; preds = %for.cond.loopexit, %incinvertfor.body
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %[[a12]], %incinvertfor.body ], [ 55, %for.cond.loopexit ]
; CHECK-NEXT:   br label %invertfor.body59
; CHECK-NEXT: }

; CHECK: define internal void @diffeinner(double* %blockA, double* %"blockA'", double %ld) unnamed_addr align 2 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = load double, double* %"blockA'", align 8
; CHECK-NEXT:   %m0diffeld = fmul fast double %0, %ld
; CHECK-NEXT:   %1 = fadd fast double %m0diffeld, %m0diffeld
; CHECK-NEXT:   store double %1, double* %"blockA'", align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
