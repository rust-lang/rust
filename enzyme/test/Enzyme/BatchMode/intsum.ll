; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -S | FileCheck %s

define float @sum(float* %array) {
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
  ret float %lcssa
}

; Function Attrs: nounwind uwtable
define [4 x float] @dsum(float* %x1, float* %x2, float* %x3, float* %x4) {
entry:
  %0 = call [4 x float] (...) @__enzyme_batch(float (float*)* @sum, metadata !"enzyme_width", i64 4, metadata !"enzyme_vector", float* %x1, float* %x2, float* %x3, float* %x4)
  ret [4 x float] %0
}

declare [4 x float] @__enzyme_batch(...)


; CHECK: define internal [4 x float] @batch_sum([4 x float*] %array)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %unwrap.array0 = extractvalue [4 x float*] %array, 0
; CHECK-NEXT:   %unwrap.array1 = extractvalue [4 x float*] %array, 1
; CHECK-NEXT:   %unwrap.array2 = extractvalue [4 x float*] %array, 2
; CHECK-NEXT:   %unwrap.array3 = extractvalue [4 x float*] %array, 3
; CHECK-NEXT:   br label %do.body

; CHECK: do.body:                                          ; preds = %do.body, %entry
; CHECK-NEXT:   %i = phi i64 [ %inc, %do.body ], [ 0, %entry ]
; CHECK-NEXT:   %intsum = phi i32 [ 0, %entry ], [ %intadd, %do.body ]
; CHECK-NEXT:   %arrayidx0 = getelementptr inbounds float, float* %unwrap.array0, i64 %i
; CHECK-NEXT:   %arrayidx1 = getelementptr inbounds float, float* %unwrap.array1, i64 %i
; CHECK-NEXT:   %arrayidx2 = getelementptr inbounds float, float* %unwrap.array2, i64 %i
; CHECK-NEXT:   %arrayidx3 = getelementptr inbounds float, float* %unwrap.array3, i64 %i
; CHECK-NEXT:   %loaded0 = load float, float* %arrayidx0
; CHECK-NEXT:   %loaded1 = load float, float* %arrayidx1
; CHECK-NEXT:   %loaded2 = load float, float* %arrayidx2
; CHECK-NEXT:   %loaded3 = load float, float* %arrayidx3
; CHECK-NEXT:   %fltload = bitcast i32 %intsum to float
; CHECK-NEXT:   %add0 = fadd float %fltload, %loaded0
; CHECK-NEXT:   %add1 = fadd float %fltload, %loaded1
; CHECK-NEXT:   %add2 = fadd float %fltload, %loaded2
; CHECK-NEXT:   %add3 = fadd float %fltload, %loaded3
; CHECK-NEXT:   %intadd = bitcast float %add0 to i32
; CHECK-NEXT:   %inc = add nuw nsw i64 %i, 1
; CHECK-NEXT:   %cmp = icmp eq i64 %inc, 5
; CHECK-NEXT:   br i1 %cmp, label %do.end, label %do.body

; CHECK: do.end:                                           ; preds = %do.body
; CHECK-NEXT:   %lcssa = phi float [ %add0, %do.body ]
; CHECK-NEXT:   %mrv = insertvalue [4 x float] undef, float %lcssa, 0
; CHECK-NEXT:   %mrv1 = insertvalue [4 x float] %mrv, float %lcssa, 1
; CHECK-NEXT:   %mrv2 = insertvalue [4 x float] %mrv1, float %lcssa, 2
; CHECK-NEXT:   %mrv3 = insertvalue [4 x float] %mrv2, float %lcssa, 3
; CHECK-NEXT:   ret [4 x float] %mrv3
; CHECK-NEXT: }