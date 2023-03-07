; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -simplifycfg -S | FileCheck %s
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
  store i64 0, i64* %v, align 8
  ret double %res
}

define double @pb(double* %__x, i64* %v) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %tiv = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %ig = getelementptr inbounds i64, i64* %v, i64 %tiv
  %iload = load i64, i64* %ig
  %icall = call double @tpfop(i64 %iload)

  %dg = getelementptr inbounds double, double* %__x, i64 %tiv
  %dload = load double, double* %dg
  %mul = fmul double %dload, %icall
  store double %mul, double* %dg

  %inc = add nsw i64 %tiv, 1
  %cmp = icmp slt i64 %inc, 4
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.cond
  ret double 0.000000e+00
}

define double @usesize(double* %ptr, i64 %off) {
entry:
  %p2 = getelementptr inbounds double, double* %ptr, i64 %off
  %ld = load double, double* %p2, align 8
  ret double %ld
}

define double @tpfop(i64 %ptr) #0 {
entry:
  %d = bitcast i64 %ptr to double
  ret double %d
}

!4 = !{!5, i64 1, !"omnipotent char"}
!5 = !{!"Simple C++ TBAA"}
!6 = !{!7, !7, i64 0, i64 8}
!7 = !{!4, i64 8, !"long"}

attributes #0 = { readnone speculatable }


; CHECK: define internal double* @augmented_pb(double* %__x, double* %"__x'", i64* %v)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %malloccall = tail call noalias nonnull dereferenceable(32) dereferenceable_or_null(32) i8* @malloc(i64 32)
; CHECK-NEXT:   %icall_malloccache = bitcast i8* %malloccall to double*
; CHECK-NEXT:   br label %for.body

; CHECK: for.body:                                         ; preds = %for.body, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %ig = getelementptr inbounds i64, i64* %v, i64 %iv
; CHECK-NEXT:   %iload = load i64, i64* %ig
; CHECK-NEXT:   %icall = call double @tpfop(i64 %iload)
; CHECK-NEXT:   %dg = getelementptr inbounds double, double* %__x, i64 %iv
; CHECK-NEXT:   %dload = load double, double* %dg
; CHECK-NEXT:   %mul = fmul double %dload, %icall
; CHECK-NEXT:   store double %mul, double* %dg
; CHECK-NEXT:   %0 = getelementptr inbounds double, double* %icall_malloccache, i64 %iv
; CHECK-NEXT:   store double %icall, double* %0, align 8, !invariant.group !
; CHECK-NEXT:   %cmp = icmp ne i64 %iv.next, 4
; CHECK-NEXT:   br i1 %cmp, label %for.body, label %for.end

; CHECK: for.end:                                          ; preds = %for.body
; CHECK-NEXT:   ret double* %icall_malloccache
; CHECK-NEXT: }

; CHECK: define internal void @diffepb(double* %__x, double* %"__x'", i64* %v, double %differeturn, double* %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   br label %for.body

; CHECK: for.body:                                         ; preds = %for.body, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %0 = getelementptr inbounds double, double* %tapeArg, i64 %iv
; CHECK-NEXT:   %icall = load double, double* %0, align 8, !invariant.group !
; CHECK-NEXT:   %"dg'ipg" = getelementptr inbounds double, double* %"__x'", i64 %iv
; CHECK-NEXT:   %cmp = icmp ne i64 %iv.next, 4
; CHECK-NEXT:   br i1 %cmp, label %for.body, label %invertfor.body

; CHECK: invertentry:                                      ; preds = %invertfor.body
; CHECK-NEXT:   %1 = bitcast double* %tapeArg to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %1)
; CHECK-NEXT:   ret void

; CHECK: invertfor.body:                                   ; preds = %for.body, %incinvertfor.body
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %11, %incinvertfor.body ], [ 3, %for.body ]
; CHECK-NEXT:   %"dg'ipg_unwrap" = getelementptr inbounds double, double* %"__x'", i64 %"iv'ac.0"
; CHECK-NEXT:   %2 = load double, double* %"dg'ipg_unwrap"
; CHECK-NEXT:   store double 0.000000e+00, double* %"dg'ipg_unwrap"
; CHECK-NEXT:   %3 = fadd fast double 0.000000e+00, %2
; CHECK-NEXT:   %4 = getelementptr inbounds double, double* %tapeArg, i64 %"iv'ac.0"
; CHECK-NEXT:   %5 = load double, double* %4, align 8, !invariant.group !
; CHECK-NEXT:   %m0diffedload = fmul fast double %3, %5
; CHECK-NEXT:   %6 = fadd fast double 0.000000e+00, %m0diffedload
; CHECK-NEXT:   %7 = load double, double* %"dg'ipg_unwrap"
; CHECK-NEXT:   %8 = fadd fast double %7, %6
; CHECK-NEXT:   store double %8, double* %"dg'ipg_unwrap"
; CHECK-NEXT:   %9 = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   %10 = xor i1 %9, true
; CHECK-NEXT:   br i1 %9, label %invertentry, label %incinvertfor.body

; CHECK: incinvertfor.body:                                ; preds = %invertfor.body
; CHECK-NEXT:   %11 = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertfor.body
; CHECK-NEXT: }
