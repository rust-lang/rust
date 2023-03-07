; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

; Function Attrs: noinline nounwind readnone uwtable
define void @tester(double* noalias %in, double* noalias %out) {
entry:
  br label %loop

loop:
  %idx = phi i64 [ 0, %entry ], [ %inc, %loop ]
  %inc = add nuw nsw i64 %idx, 1
  %igep = getelementptr inbounds double, double* %in, i64 %idx
  %ogep = getelementptr inbounds double, double* %out, i64 %idx
  
  %ival = load double, double* %igep
  %sq = fmul fast double %ival, %ival
  store double %sq, double* %ogep
  %cmp = icmp eq i64 %inc, 6
  br i1 %cmp, label %end, label %loop

end:
  ret void
}

define void @test_derivative(double* %x, double* %dx, double* %y, double* %dy) {
entry:
  %size = call i64 (void (double*, double*)*, ...) @__enzyme_augmentsize(void (double*, double*)* nonnull @tester, metadata !"enzyme_dup", metadata !"enzyme_dup")
  %cache = alloca i8, i64 %size, align 1
  call void (void (double*, double*)*, ...) @__enzyme_augmentfwd(void (double*, double*)* nonnull @tester, metadata !"enzyme_allocated", i64 %size, metadata !"enzyme_tape", i8* %cache, double* %x, double* %dx, double* %y, double* %dy)
  tail call void (void (double*, double*)*, ...) @__enzyme_reverse(void (double*, double*)* nonnull @tester, metadata !"enzyme_allocated", i64 %size, metadata !"enzyme_nofree", metadata !"enzyme_tape", i8* %cache, double* %x, double* %dx, double* %y, double* %dy)
  ret void
}

; Function Attrs: nounwind
declare void @__enzyme_augmentfwd(void (double*, double*)*, ...)
declare i64 @__enzyme_augmentsize(void (double*, double*)*, ...)
declare void @__enzyme_reverse(void (double*, double*)*, ...)

; CHECK: define void @test_derivative(double* %x, double* %dx, double* %y, double* %dy)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %cache = alloca i8, i64 8, align 1
; CHECK-NEXT:   %0 = call double* @augmented_tester(double* %x, double* %dx, double* %y, double* %dy)
; CHECK-NEXT:   %1 = bitcast i8* %cache to double**
; CHECK-NEXT:   store double* %0, double** %1
; CHECK-NEXT:   %2 = bitcast i8* %cache to double**
; CHECK-NEXT:   %3 = load double*, double** %2
; CHECK-NEXT:   call void @diffetester(double* %x, double* %dx, double* %y, double* %dy, double* %3)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @diffetester(double* noalias %in, double* %"in'", double* noalias %out, double* %"out'", double* %tapeArg)
; CHECK-NOT: free
; CHECK: }
