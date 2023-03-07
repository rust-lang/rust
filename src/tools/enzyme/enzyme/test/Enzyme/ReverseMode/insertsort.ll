; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -inline -mem2reg -adce -instsimplify -simplifycfg -S | FileCheck %s

; Function Attrs: noinline norecurse nounwind uwtable
define dso_local void @insertion_sort_inner(float* nocapture %array, i32 %i) local_unnamed_addr #0 {
entry:
  %cmp29 = icmp sgt i32 %i, 0
  br i1 %cmp29, label %land.rhs.preheader, label %while.end

land.rhs.preheader:                               ; preds = %entry
  %0 = sext i32 %i to i64
  br label %land.rhs

land.rhs:                                         ; preds = %land.rhs.preheader, %while.body
  %indvars.iv = phi i64 [ %0, %land.rhs.preheader ], [ %indvars.iv.next, %while.body ]
  %indvars.iv.next = add nsw i64 %indvars.iv, -1
  %arrayidx = getelementptr inbounds float, float* %array, i64 %indvars.iv.next
  %1 = load float, float* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds float, float* %array, i64 %indvars.iv
  %2 = load float, float* %arrayidx2, align 4
  %cmp3 = fcmp ogt float %1, %2
  br i1 %cmp3, label %while.body, label %while.end

while.body:                                       ; preds = %land.rhs
  store float %1, float* %arrayidx2, align 4
  store float %2, float* %arrayidx, align 4
  %cmp = icmp sgt i64 %indvars.iv, 1
  br i1 %cmp, label %land.rhs, label %while.end

while.end:                                        ; preds = %land.rhs, %while.body, %entry
  ret void
}


define dso_local void @dsum(float* %x, float* %xp, i32 %n) {
entry:
  %0 = tail call double (void (float*, i32)*, ...) @__enzyme_autodiff(void (float*, i32)* nonnull @insertion_sort_inner, float* %x, float* %xp, i32 %n)
  ret void
}

declare double @__enzyme_autodiff(void (float*, i32)*, ...)

attributes #0 = { noinline norecurse nounwind uwtable }

; CHECK: define internal void @diffeinsertion_sort_inner(float* nocapture %array, float* nocapture %"array'", i32 %i)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %cmp29 = icmp sgt i32 %i, 0
; CHECK-NEXT:   br i1 %cmp29, label %land.rhs.preheader, label %invertwhile.end

; CHECK: land.rhs.preheader:                               ; preds = %entry
; CHECK-NEXT:   %0 = sext i32 %i to i64
; CHECK-NEXT:   br label %land.rhs

; CHECK: land.rhs:                                         ; preds = %while.body, %land.rhs.preheader
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %while.body ], [ 0, %land.rhs.preheader ]
; CHECK-DAG:    %iv.next = add nuw nsw i64 %iv, 1
; CHECK-DAG:    %1 = mul {{(nuw )?}}{{(nsw )?}}i64 %iv, -1
; CHECK-DAG:   %[[a1:.+]] = add i64 %0, %1
; CHECK-DAG:   %indvars.iv.next = add nsw i64 %[[a1]], -1
; CHECK-NEXT:   %arrayidx = getelementptr inbounds float, float* %array, i64 %indvars.iv.next
; CHECK-NEXT:   %[[a2:.+]] = load float, float* %arrayidx, align 4
; CHECK-NEXT:   %arrayidx2 = getelementptr inbounds float, float* %array, i64 %[[a1]]
; CHECK-NEXT:   %[[a3:.+]] = load float, float* %arrayidx2, align 4
; CHECK-NEXT:   %cmp3 = fcmp ogt float %[[a2]], %[[a3]]
; CHECK-NEXT:   br i1 %cmp3, label %while.body, label %invertwhile.end

; CHECK: while.body:                                       ; preds = %land.rhs
; CHECK-NEXT:   store float %[[a2]], float* %arrayidx2, align 4
; CHECK-NEXT:   store float %[[a3]], float* %arrayidx, align 4
; CHECK-NEXT:   %cmp = icmp sgt i64 %[[a1]], 1
; CHECK-NEXT:   br i1 %cmp, label %land.rhs, label %invertwhile.end

; CHECK: invertentry:                                      ; preds = %invertland.rhs, %invertwhile.end
; CHECK-NEXT:   ret void

; CHECK: invertland.rhs:                                   ; preds = %invertwhile.end.loopexit, %invertwhile.body
; CHECK-NEXT:   %[[de5:.+]] = phi float [ %[[dein:.+]], %invertwhile.body ], [ 0.000000e+00, %invertwhile.end.loopexit ]
; CHECK-NEXT:   %"'de.0" = phi float [ %[[pdein:.+]], %invertwhile.body ], [ 0.000000e+00, %invertwhile.end.loopexit ]
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %"iv'ac.1", %invertwhile.body ], [ %loopLimit_cache.0, %invertwhile.end.loopexit ]
; CHECK-NEXT:   %_unwrap = sext i32 %i to i64
; CHECK-NEXT:   %_unwrap1 = mul {{(nuw )?}}{{(nsw )?}}i64 %"iv'ac.0", -1
; CHECK-NEXT:   %[[_unwrap2:.+]] = add i64 %_unwrap, %_unwrap1
; CHECK-NEXT:   %[[arrayidx2ipg:.+]] = getelementptr inbounds float, float* %"array'", i64 %[[_unwrap2]]
; CHECK-NEXT:   %[[a4:.+]] = load float, float* %[[arrayidx2ipg]], align 4
; CHECK-NEXT:   %[[a5:.+]] = fadd fast float %[[a4]], %"'de.0"
; CHECK-NEXT:   store float %[[a5]], float* %[[arrayidx2ipg]], align 4
; CHECK-NEXT:   %indvars.iv.next_unwrap = add nsw i64 %[[_unwrap2]], -1
; CHECK-NEXT:   %[[arrayidxipg:.+]] = getelementptr inbounds float, float* %"array'", i64 %indvars.iv.next_unwrap
; CHECK-NEXT:   %[[loade5:.+]] = load float, float* %[[arrayidxipg]], align 4
; CHECK-NEXT:   %[[adde5:.+]] = fadd fast float %[[loade5]], %[[de5]]
; CHECK-NEXT:   store float %[[adde5]], float* %[[arrayidxipg]], align 4
; CHECK-NEXT:   %[[cmpeq:.+]] = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %[[cmpeq]], label %invertentry, label %incinvertland.rhs

; CHECK: incinvertland.rhs:                                ; preds = %invertland.rhs
; CHECK-NEXT:   %[[sub1h:.+]] = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertwhile.body

; CHECK: invertwhile.body:                                 ; preds = %invertwhile.end.loopexit, %incinvertland.rhs
; CHECK-NEXT:   %"iv'ac.1" = phi i64 [ %[[sub1h]], %incinvertland.rhs ], [ %loopLimit_cache.0, %invertwhile.end.loopexit ]
; CHECK-NEXT:   %_unwrap4 = sext i32 %i to i64
; CHECK-NEXT:   %_unwrap5 = mul {{(nuw )?}}{{(nsw )?}}i64 %"iv'ac.1", -1
; CHECK-NEXT:   %_unwrap6 = add {{(nuw )?}}{{(nsw )?}}i64 %_unwrap4, %_unwrap5
; CHECK-NEXT:   %[[indvarsivnext_unwrap7:.+]] = add nsw i64 %_unwrap6, -1
; CHECK-NEXT:   %[[arrayidxipg10:.+]] = getelementptr inbounds float, float* %"array'", i64 %[[indvarsivnext_unwrap7]]
; CHECK-NEXT:   %[[pdein]] = load float, float* %[[arrayidxipg10]], align 4
; CHECK-NEXT:   store float 0.000000e+00, float* %[[arrayidxipg10]], align 4
; CHECK-NEXT:   %[[arrayidx2ipg17:.+]] = getelementptr inbounds float, float* %"array'", i64 %_unwrap6
; CHECK-NEXT:   %[[dein]] = load float, float* %[[arrayidx2ipg17]], align 4
; CHECK-NEXT:   store float 0.000000e+00, float* %[[arrayidx2ipg17]], align 4
; CHECK-NEXT:   br label %invertland.rhs

; CHECK: invertwhile.end.loopexit:                         ; preds = %invertwhile.end
; CHECK-NEXT:   br i1 %[[cmp3_rev:.+]], label %invertwhile.body, label %invertland.rhs

; CHECK: invertwhile.end:                                  ; preds = %entry, %while.body, %land.rhs
; CHECK-NEXT:   %[[cmp3_rev]] = phi i1 [ undef, %entry ], [ %cmp3, %while.body ], [ %cmp3, %land.rhs ]
; CHECK-NEXT:   %loopLimit_cache.0 = phi i64 [ undef, %entry ], [ %iv, %while.body ], [ %iv, %land.rhs ]
; CHECK-NEXT:   br i1 %cmp29, label %invertwhile.end.loopexit, label %invertentry
; CHECK-NEXT: }
