; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -adce -S | FileCheck %s
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(adce)" -enzyme-preopt=false -S | FileCheck %s

define dso_local void @sum(float* %array, float* %ret) #4 {
entry:
  br label %do.body

do.body:                                          ; preds = %do.body, %entry
  %i = phi i64 [ %inc, %do.body ], [ 0, %entry ]
  %intsum = phi i32 [ 0, %entry ], [ %intadd, %do.body ]
  %arrayidx = getelementptr inbounds float, float* %array, i64 %i
  %loaded = load float, float* %arrayidx
  %fltload = bitcast i32 %intsum to float
  %add = fadd float %fltload, %loaded
  %intadd = bitcast float %add to i32
  %inc = add nuw nsw i64 %i, 1
  %cmp = icmp eq i64 %inc, 5
  br i1 %cmp, label %do.end, label %do.body

do.end:                                           ; preds = %do.body
  %lcssa = phi float [ %add, %do.body ]
  store float %lcssa, float* %ret, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define dso_local void @dsum(float* %x, float* %xp, float* %n, float* %np) local_unnamed_addr #1 {
entry:
  %0 = tail call double (void (float*, float*)*, ...) @__enzyme_fwddiff(void (float*, float*)* nonnull @sum, float* %x, float* %xp, float* %n, float* %np)
  ret void
}

declare double @__enzyme_fwddiff(void (float*, float*)*, ...) #2


; CHECK: define internal void @fwddiffesum(float* %array, float* %"array'", float* %ret, float* %"ret'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   br label %do.body

; CHECK: do.body:                                          ; preds = %do.body, %entry
; CHECK-DAG:   %iv = phi i64 [ %iv.next, %do.body ], [ 0, %entry ]
; CHECK-DAG:   %intsum = phi i32 [ 0, %entry ], [ %intadd, %do.body ]
; CHECK-DAG:   %[[dintsum:.+]] = phi i32 [ 0, %entry ], [ %[[i3:.+]], %do.body ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %"arrayidx'ipg" = getelementptr inbounds float, float* %"array'", i64 %iv
; CHECK-NEXT:   %arrayidx = getelementptr inbounds float, float* %array, i64 %iv
; CHECK-NEXT:   %[[i0:.+]] = load float, float* %"arrayidx'ipg"
; CHECK-NEXT:   %loaded = load float, float* %arrayidx
; CHECK-NEXT:   %[[i1:.+]] = bitcast i32 %[[dintsum]] to float
; CHECK-NEXT:   %fltload = bitcast i32 %intsum to float
; CHECK-NEXT:   %add = fadd float %fltload, %loaded
; CHECK-NEXT:   %[[i2:.+]] = fadd fast float %[[i1]], %[[i0]]
; CHECK-NEXT:   %[[i3]] = bitcast float %[[i2]] to i32
; CHECK-NEXT:   %intadd = bitcast float %add to i32
; CHECK-NEXT:   %cmp = icmp eq i64 %iv.next, 5
; CHECK-NEXT:   br i1 %cmp, label %do.end, label %do.body

; CHECK: do.end:                                           ; preds = %do.body
; CHECK-NEXT:   store float %add, float* %ret, align 4
; CHECK-NEXT:   store float %[[i2]], float* %"ret'", align 4
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
