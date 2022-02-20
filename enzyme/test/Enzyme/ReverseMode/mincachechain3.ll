; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -simplifycfg -adce -S | FileCheck %s
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
  %size.0 = phi i64 [ 1, %entry ], [ %a2, %for.body ]
  %call3 = call i64* @_ZNKSt5arrayIlLm4EEixEm(i64* %v)
  %a2 = load i64, i64* %call3, align 8, !tbaa !6
  %inc = add nsw i64 %tiv, 1
  %cmp = icmp slt i64 %inc, 4
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.cond
  %ld = call double @usesize(double* %__x, i64 %size.0)
  ret double %ld
}

define double @usesize(double* %ptr, i64 %off) {
entry:
  %p2 = getelementptr inbounds double, double* %ptr, i64 %off
  %ld = load double, double* %p2, align 8
  ret double %ld
}

define i64* @_ZNKSt5arrayIlLm4EEixEm(i64* %ptr) {
entry:
  ret i64* %ptr
}

!4 = !{!5, i64 1, !"omnipotent char"}
!5 = !{!"Simple C++ TBAA"}
!6 = !{!7, !7, i64 0, i64 8}
!7 = !{!4, i64 8, !"long"}

; CHECK: define internal void @diffe_Z10reduce_maxPdi(double* %vec, double* %"vec'", i64* %v, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %res_augmented = call i64* @augmented_pb(double* %vec, double* %"vec'", i64* %v)
; CHECK-NEXT:   store i64 0, i64* %v, align 8
; CHECK-NEXT:   call void @diffepb(double* %vec, double* %"vec'", i64* %v, double %differeturn, i64* %res_augmented)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal i64* @augmented__ZNKSt5arrayIlLm4EEixEm(i64* %ptr)
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret i64* %ptr
; CHECK-NEXT: }

; CHECK: define internal void @augmented_usesize(double* %ptr, double* %"ptr'", i64 %off)
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal i64* @augmented_pb(double* %__x, double* %"__x'", i64* %v)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %malloccall = tail call noalias nonnull dereferenceable(32) dereferenceable_or_null(32) i8* @malloc(i64 32)
; CHECK-NEXT:   %a2_malloccache = bitcast i8* %malloccall to i64*
; CHECK-NEXT:   br label %for.body

; CHECK: for.body:                                         ; preds = %for.body, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
; CHECK-NEXT:   %size.0 = phi i64 [ 1, %entry ], [ %a2, %for.body ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %call3 = call i64* @augmented__ZNKSt5arrayIlLm4EEixEm(i64* %v)
; CHECK-NEXT:   %a2 = load i64, i64* %call3, align 8, !tbaa !0
; CHECK-NEXT:   %0 = getelementptr inbounds i64, i64* %a2_malloccache, i64 %iv
; CHECK-NEXT:   store i64 %a2, i64* %0, align 8, !tbaa !0, !invariant.group 
; CHECK-NEXT:   %cmp = icmp ne i64 %iv.next, 4
; CHECK-NEXT:   br i1 %cmp, label %for.body, label %for.end

; CHECK: for.end:                                          ; preds = %for.body
; CHECK-NEXT:   call void @augmented_usesize(double* %__x, double* %"__x'", i64 %size.0)
; CHECK-NEXT:   ret i64* %a2_malloccache
; CHECK-NEXT: }

; CHECK: define internal void @diffepb(double* %__x, double* %"__x'", i64* %v, double %differeturn, i64* %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   br label %for.body

; CHECK: for.body:                                         ; preds = %for.body, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
; CHECK-NEXT:   %size.0 = phi i64 [ 1, %entry ], [ %a2, %for.body ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %0 = getelementptr inbounds i64, i64* %tapeArg, i64 %iv
; CHECK-NEXT:   %a2 = load i64, i64* %0, align 8, !invariant.group !
; CHECK-NEXT:   %cmp = icmp ne i64 %iv.next, 4
; CHECK-NEXT:   br i1 %cmp, label %for.body, label %invertfor.end

; CHECK: invertentry:                                      ; preds = %invertfor.body
; CHECK-NEXT:   %[[tofree:.+]] = bitcast i64* %tapeArg to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[tofree]])
; CHECK-NEXT:   ret void

; CHECK: invertfor.body:                                   ; preds = %invertfor.end, %incinvertfor.body
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ 3, %invertfor.end ], [ %[[dec:.+]], %incinvertfor.body ]
; CHECK-NEXT:   call void @diffe_ZNKSt5arrayIlLm4EEixEm(i64* %v)
; CHECK-NEXT:   %[[rcmp:.+]] = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %[[rcmp]], label %invertentry, label %incinvertfor.body

; CHECK: incinvertfor.body:                                ; preds = %invertfor.body
; CHECK-NEXT:   %[[dec]] = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertfor.body

; CHECK: invertfor.end:                                    ; preds = %for.body
; CHECK-NEXT:   call void @diffeusesize(double* %__x, double* %"__x'", i64 %size.0, double %differeturn)
; CHECK-NEXT:   br label %invertfor.body
; CHECK-NEXT: }

; CHECK: define internal void @diffe_ZNKSt5arrayIlLm4EEixEm(i64* %ptr)
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @diffeusesize(double* %ptr, double* %"ptr'", i64 %off, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"p2'ipg" = getelementptr inbounds double, double* %"ptr'", i64 %off
; CHECK-NEXT:   %0 = load double, double* %"p2'ipg", align 8
; CHECK-NEXT:   %1 = fadd fast double %0, %differeturn
; CHECK-NEXT:   store double %1, double* %"p2'ipg", align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
