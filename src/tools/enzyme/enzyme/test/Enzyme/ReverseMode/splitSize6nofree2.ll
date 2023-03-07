; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1)

; Function Attrs: noinline nounwind readnone uwtable
define void @tester(double* noalias %in, double* noalias %out) {
entry:
  %tmp = alloca double, i64 6
  br label %loop

loop:
  %idx = phi i64 [ 0, %entry ], [ %inc, %loop ]
  %inc = add nuw nsw i64 %idx, 1
  %igep = getelementptr inbounds double, double* %in, i64 %idx
  %tgep = getelementptr inbounds double, double* %tmp, i64 %idx
  
  %ival = load double, double* %igep
  store double %ival, double* %tgep
  %cmp = icmp eq i64 %inc, 6
  br i1 %cmp, label %end, label %loop

end:
  %dst = bitcast double* %out to i8*
  %src = bitcast double* %tmp to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst, i8* %src, i64 48, i1 false)
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
; CHECK-NEXT:   %0 = call i8* @augmented_tester(double* %x, double* %dx, double* %y, double* %dy)
; CHECK-NEXT:   %1 = bitcast i8* %cache to i8**
; CHECK-NEXT:   store i8* %0, i8** %1
; CHECK-NEXT:   %2 = bitcast i8* %cache to i8**
; CHECK-NEXT:   %3 = load i8*, i8** %2
; CHECK-NEXT:   call void @diffetester(double* %x, double* %dx, double* %y, double* %dy, i8* %3)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @diffetester(double* noalias %in, double* %"in'", double* noalias %out, double* %"out'", i8*
; CHECK-NOT: @free
