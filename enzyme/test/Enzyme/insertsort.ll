; RUN: opt < %s %loadEnzyme -enzyme -enzyme_preopt=false -inline -mem2reg -correlated-propagation -instsimplify -adce -loop-deletion -simplifycfg -S | FileCheck %s

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

; CHECK: define internal {} @diffeinsertion_sort_inner(float* nocapture %array, float* %"array'", i32 %i) local_unnamed_addr #0 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %cmp29 = icmp sgt i32 %i, 0
; CHECK-NEXT:   br i1 %cmp29, label %land.rhs.preheader, label %while.end

; CHECK: land.rhs.preheader:                               ; preds = %entry
; CHECK-NEXT:   %0 = sext i32 %i to i64
; CHECK-NEXT:   br label %land.rhs

; CHECK-NEXT: land.rhs:                                         ; preds = %while.body, %land.rhs.preheader
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %while.body ], [ 0, %land.rhs.preheader ]
; CHECK-NEXT:   %1 = mul i64 %iv, -1
; CHECK-NEXT:   %2 = add i64 %0, %1
; CHECK-NEXT:   %iv.next = add nuw i64 %iv, 1
; CHECK-NEXT:   %indvars.iv.next = add nsw i64 %2, -1
; CHECK-NEXT:   %arrayidx = getelementptr inbounds float, float* %array, i64 %indvars.iv.next
; CHECK-NEXT:   %3 = load float, float* %arrayidx, align 4
; CHECK-NEXT:   %arrayidx2 = getelementptr inbounds float, float* %array, i64 %2
; CHECK-NEXT:   %4 = load float, float* %arrayidx2, align 4
; CHECK-NEXT:   %cmp3 = fcmp ogt float %3, %4
; CHECK-NEXT:   br i1 %cmp3, label %while.body, label %while.end.loopexit

; CHECK: while.body:                                       ; preds = %land.rhs
; CHECK-NEXT:   store float %3, float* %arrayidx2, align 4
; CHECK-NEXT:   store float %4, float* %arrayidx, align 4
; CHECK-NEXT:   %cmp = icmp sgt i64 %2, 1
; CHECK-NEXT:   br i1 %cmp, label %land.rhs, label %while.end.loopexit

; CHECK: while.end.loopexit:                               ; preds = %while.body, %land.rhs
; CHECK-NEXT:   %"cmp3!manual_lcssa" = phi i1 [ %cmp3, %while.body ], [ %cmp3, %land.rhs ]
; CHECK-NEXT:   %5 = phi i8 [ 0, %while.body ], [ 1, %land.rhs ]
; CHECK-NEXT:   %6 = phi i64 [ %iv, %while.body ], [ %iv, %land.rhs ]
; CHECK-NEXT:   br label %while.end

; CHECK: while.end:                                        ; preds = %while.end.loopexit, %entry
; CHECK-NEXT:   %"cmp3!manual_lcssa_cache.0" = phi i1 [ %"cmp3!manual_lcssa", %while.end.loopexit ], [ undef, %entry ]
; CHECK-NEXT:   %_cache1.0 = phi i8 [ %5, %while.end.loopexit ], [ undef, %entry ]
; CHECK-NEXT:   %_cache.0 = phi i64 [ %6, %while.end.loopexit ], [ undef, %entry ]
; CHECK-NEXT:   br label %invertwhile.end

; CHECK: invertentry:                                      ; preds = %invertwhile.end, %invertland.rhs.preheader
; CHECK-NEXT:   ret {} undef

; CHECK: invertland.rhs.preheader:                         ; preds = %invertland.rhs
; CHECK-NEXT:   br label %invertentry

; CHECK: invertland.rhs:                                   ; preds = %invertwhile.body, %loopMerge
; CHECK-NEXT:   %"'de2.0" = phi float [ 0.000000e+00, %loopMerge ], [ %30, %invertwhile.body ]
; CHECK-NEXT:   %"'de.0" = phi float [ 0.000000e+00, %loopMerge ], [ %24, %invertwhile.body ]
; CHECK-NEXT:   %_unwrap = sext i32 %i to i64
; CHECK-NEXT:   %7 = mul i64 %"iv'phi", -1
; CHECK-NEXT:   %8 = add i64 %_unwrap, %7
; CHECK-NEXT:   %"arrayidx2'ipg" = getelementptr float, float* %"array'", i64 %8
; CHECK-NEXT:   %9 = load float, float* %"arrayidx2'ipg"
; CHECK-NEXT:   %10 = fadd fast float %9, %"'de.0"
; CHECK-NEXT:   store float %10, float* %"arrayidx2'ipg"
; CHECK-NEXT:   %_unwrap3 = sext i32 %i to i64
; CHECK-NEXT:   %11 = mul i64 %"iv'phi", -1
; CHECK-NEXT:   %12 = add i64 %_unwrap3, %11
; CHECK-NEXT:   %13 = add i64 %12, -1
; CHECK-NEXT:   %"arrayidx'ipg" = getelementptr float, float* %"array'", i64 %13
; CHECK-NEXT:   %14 = load float, float* %"arrayidx'ipg"
; CHECK-NEXT:   %15 = fadd fast float %14, %"'de2.0"
; CHECK-NEXT:   store float %15, float* %"arrayidx'ipg"
; CHECK-NEXT:   %16 = icmp eq i64 %"iv'phi", 0
; CHECK-NEXT:   br i1 %16, label %invertland.rhs.preheader, label %loopMerge

; CHECK: invertwhile.body:                                 ; preds = %loopMerge
; CHECK-NEXT:   %_unwrap4 = sext i32 %i to i64
; CHECK-NEXT:   %17 = mul i64 %"iv'phi", -1
; CHECK-NEXT:   %18 = add i64 %_unwrap4, %17
; CHECK-NEXT:   %19 = add i64 %18, -1
; CHECK-NEXT:   %"arrayidx'ipg5" = getelementptr float, float* %"array'", i64 %19
; CHECK-NEXT:   %20 = load float, float* %"arrayidx'ipg5"
; CHECK-NEXT:   %_unwrap6 = sext i32 %i to i64
; CHECK-NEXT:   %21 = mul i64 %"iv'phi", -1
; CHECK-NEXT:   %22 = add i64 %_unwrap6, %21
; CHECK-NEXT:   %23 = add i64 %22, -1
; CHECK-NEXT:   %"arrayidx'ipg7" = getelementptr float, float* %"array'", i64 %23
; CHECK-NEXT:   store float 0.000000e+00, float* %"arrayidx'ipg7"
; CHECK-NEXT:   %24 = fadd fast float 0.000000e+00, %20
; CHECK-NEXT:   %_unwrap8 = sext i32 %i to i64
; CHECK-NEXT:   %25 = mul i64 %"iv'phi", -1
; CHECK-NEXT:   %26 = add i64 %_unwrap8, %25
; CHECK-NEXT:   %"arrayidx2'ipg9" = getelementptr float, float* %"array'", i64 %26
; CHECK-NEXT:   %27 = load float, float* %"arrayidx2'ipg9"
; CHECK-NEXT:   %_unwrap10 = sext i32 %i to i64
; CHECK-NEXT:   %28 = mul i64 %"iv'phi", -1
; CHECK-NEXT:   %29 = add i64 %_unwrap10, %28
; CHECK-NEXT:   %"arrayidx2'ipg11" = getelementptr float, float* %"array'", i64 %29
; CHECK-NEXT:   store float 0.000000e+00, float* %"arrayidx2'ipg11"
; CHECK-NEXT:   %30 = fadd fast float 0.000000e+00, %27
; CHECK-NEXT:   br label %invertland.rhs

; CHECK: invertwhile.end.loopexit:                         ; preds = %invertwhile.end
; CHECK-NEXT:   br label %loopMerge

; CHECK: invertwhile.end:                                  ; preds = %while.end
; CHECK-NEXT:   %31 = icmp sgt i32 %i, 0
; CHECK-NEXT:   br i1 %31, label %invertwhile.end.loopexit, label %invertentry

; CHECK: loopMerge:                                        ; preds = %invertwhile.end.loopexit, %invertland.rhs
; CHECK-NEXT:   %"iv'phi" = phi i64 [ %_cache.0, %invertwhile.end.loopexit ], [ %32, %invertland.rhs ]
; CHECK-NEXT:   %32 = sub i64 %"iv'phi", 1
; CHECK-NEXT:   switch i8 %_cache1.0, label %invertland.rhs [
; CHECK-NEXT:     i8 0, label %invertwhile.body
; CHECK-NEXT:   ]
; CHECK-NEXT: }
