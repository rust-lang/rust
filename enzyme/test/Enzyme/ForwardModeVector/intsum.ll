; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -adce -correlated-propagation -simplifycfg -adce -S | FileCheck %s

%struct.Gradients = type { float*, float*, float* }

; Function Attrs: nounwind
declare void @__enzyme_fwddiff(void (float*, float*)*, ...)

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
define dso_local void @dsum(float* %x, float* %xp1, float* %xp2, float* %xp3, float* %n, float* %np1, float* %np2, float* %np3) local_unnamed_addr #1 {
entry:
  tail call void (void (float*, float*)*, ...) @__enzyme_fwddiff(void (float*, float*)* nonnull @sum, metadata !"enzyme_width", i64 3, float* %x, float* %xp1, float* %xp2, float* %xp3, float* %n, float* %np1, float* %np2, float* %np3)
  ret void
}


; CHECK: define internal void @fwddiffe3sum(float* %array, [3 x float*] %"array'", float* %ret, [3 x float*] %"ret'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   br label %do.body

; CHECK: do.body:                                          ; preds = %do.body, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %do.body ], [ 0, %entry ]
; CHECK-NEXT:   %intsum = phi i32 [ 0, %entry ], [ %intadd, %do.body ]
; CHECK-NEXT:   %"intsum'" = phi [3 x i32] [ zeroinitializer, %entry ], [ %20, %do.body ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %0 = extractvalue [3 x float*] %"array'", 0
; CHECK-NEXT:   %"arrayidx'ipg" = getelementptr inbounds float, float* %0, i64 %iv
; CHECK-NEXT:   %1 = extractvalue [3 x float*] %"array'", 1
; CHECK-NEXT:   %"arrayidx'ipg1" = getelementptr inbounds float, float* %1, i64 %iv
; CHECK-NEXT:   %2 = extractvalue [3 x float*] %"array'", 2
; CHECK-NEXT:   %"arrayidx'ipg2" = getelementptr inbounds float, float* %2, i64 %iv
; CHECK-NEXT:   %arrayidx = getelementptr inbounds float, float* %array, i64 %iv
; CHECK-NEXT:   %loaded = load float, float* %arrayidx
; CHECK-NEXT:   %3 = load float, float* %"arrayidx'ipg"
; CHECK-NEXT:   %4 = load float, float* %"arrayidx'ipg1"
; CHECK-NEXT:   %5 = load float, float* %"arrayidx'ipg2"
; CHECK-NEXT:   %fltload = bitcast i32 %intsum to float
; CHECK-NEXT:   %6 = extractvalue [3 x i32] %"intsum'", 0
; CHECK-NEXT:   %7 = bitcast i32 %6 to float
; CHECK-NEXT:   %8 = extractvalue [3 x i32] %"intsum'", 1
; CHECK-NEXT:   %9 = bitcast i32 %8 to float
; CHECK-NEXT:   %10 = extractvalue [3 x i32] %"intsum'", 2
; CHECK-NEXT:   %11 = bitcast i32 %10 to float
; CHECK-NEXT:   %add = fadd float %fltload, %loaded
; CHECK-NEXT:   %12 = fadd fast float %7, %3
; CHECK-NEXT:   %13 = fadd fast float %9, %4
; CHECK-NEXT:   %14 = fadd fast float %11, %5
; CHECK-NEXT:   %intadd = bitcast float %add to i32
; CHECK-NEXT:   %15 = bitcast float %12 to i32
; CHECK-NEXT:   %16 = insertvalue [3 x i32] undef, i32 %15, 0
; CHECK-NEXT:   %17 = bitcast float %13 to i32
; CHECK-NEXT:   %18 = insertvalue [3 x i32] %16, i32 %17, 1
; CHECK-NEXT:   %19 = bitcast float %14 to i32
; CHECK-NEXT:   %20 = insertvalue [3 x i32] %18, i32 %19, 2
; CHECK-NEXT:   %cmp = icmp eq i64 %iv.next, 5
; CHECK-NEXT:   br i1 %cmp, label %do.end, label %do.body

; CHECK: do.end:                                           ; preds = %do.body
; CHECK-NEXT:   store float %add, float* %ret, align 4
; CHECK-NEXT:   %21 = extractvalue [3 x float*] %"ret'", 0
; CHECK-NEXT:   store float %12, float* %21, align 4
; CHECK-NEXT:   %22 = extractvalue [3 x float*] %"ret'", 1
; CHECK-NEXT:   store float %13, float* %22, align 4
; CHECK-NEXT:   %23 = extractvalue [3 x float*] %"ret'", 2
; CHECK-NEXT:   store float %14, float* %23, align 4
; CHECK-NEXT:   ret void
; CHECK-NEXT: }