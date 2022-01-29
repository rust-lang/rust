; RUN: if [ %llvmver -ge 9 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -gvn -adce -instsimplify -early-cse-memssa -simplifycfg -S | FileCheck %s; fi

; ModuleID = 'cuda.cu'
source_filename = "cuda.cu"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

%struct.cudaFuncAttributes = type { i64, i64, i64, i32, i32, i32, i32, i32, i32, i32 }

; Function Attrs: nofree nounwind
define dso_local void @_Z4axpyfPfS_(double %x, double* nocapture readonly %y, double* nocapture %z) {
  %tix = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3, !range !10
  %zext = zext i32 %tix to i64
  %g0 = getelementptr inbounds double, double* %y, i64 %zext
  %ld = load double, double* %g0, align 4, !tbaa !11
  %res = fmul contract double %ld, %x
  %gep = getelementptr inbounds double, double* %z, i64 %zext
  store double %res, double* %gep, align 4, !tbaa !11
  ret void
}

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() #2

; Function Attrs: nounwind
declare void @__enzyme_autodiff(i8*, ...)

define void @test_derivative(double %x, double* %y, double* %yp, double* %z, double* %zp) {
entry:
  call void (i8*, ...) @__enzyme_autodiff(i8* bitcast (void (double, double*, double*)* @_Z4axpyfPfS_ to i8*), metadata !"enzyme_const", double %x, double* %y, double* %yp, double* %z, double* %zp)
  ret void
}

; Function Attrs: nounwind readnone speculatable
declare double @llvm.cos.f64(double)

; Function Attrs: nounwind readnone speculatable
declare double @llvm.sin.f64(double)


;   /home/wmoses/git/Enzyme/build/./bin/opt < /mnt/Data/git/Enzyme/enzyme/test/Enzyme/cuda.ll  -load=/home/wmoses/git/Enzyme/enzyme/build7/Enzyme/LLVMEnzyme-7.so -enzyme -enzyme-preopt=false -inline -mem2reg -gvn -adce -instcombine -instsimplify -early-cse-memssa -simplifycfg -correlated-propagation -adce -loop-simplify -jump-threading -instsimplify -early-cse -simplifycfg -S 

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="sm_50" "target-features"="+ptx64,+sm_50" "unsafe-fp-math"="false" "use-soft-double"="false" }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind }

!llvm.module.flags = !{!0, !1, !2}
!nvvm.annotations = !{!3, !4, !5, !4, !6, !6, !6, !6, !7, !7, !6}
!llvm.ident = !{!8}
!nvvmir.version = !{!9}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 10, i32 1]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!3 = !{void (double, double*, double*)* @_Z4axpyfPfS_, !"kernel", i32 1}
!4 = !{null, !"align", i32 8}
!5 = !{null, !"align", i32 8, !"align", i32 65544, !"align", i32 131080}
!6 = !{null, !"align", i32 16}
!7 = !{null, !"align", i32 16, !"align", i32 65552, !"align", i32 131088}
!8 = !{!"Ubuntu clang version 10.0.1-++20200809072545+ef32c611aa2-1~exp1~20200809173142.193"}
!9 = !{i32 1, i32 4}
!10 = !{i32 0, i32 1024}
!11 = !{!12, !12, i64 0}
!12 = !{!"double", !13, i64 0}
!13 = !{!"omnipotent char", !14, i64 0}
!14 = !{!"Simple C++ TBAA"}

; CHECK: define internal void @diffe_Z4axpyfPfS_(double %x, double* nocapture readonly %y, double* nocapture %"y'", double* nocapture %z, double* nocapture %"z'")
; CHECK-NEXT: invert:
; CHECK-NEXT:   %tix = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
; CHECK-NEXT:   %zext = zext i32 %tix to i64
; CHECK-NEXT:   %"g0'ipg" = getelementptr inbounds double, double* %"y'", i64 %zext
; CHECK-NEXT:   %g0 = getelementptr inbounds double, double* %y, i64 %zext
; CHECK-NEXT:   %ld = load double, double* %g0, align 4, !tbaa !11
; CHECK-NEXT:   %res = fmul contract double %ld, %x
; CHECK-NEXT:   %"gep'ipg" = getelementptr inbounds double, double* %"z'", i64 %zext
; CHECK-NEXT:   %gep = getelementptr inbounds double, double* %z, i64 %zext
; CHECK-NEXT:   store double %res, double* %gep, align 4, !tbaa !11
; CHECK-NEXT:   %0 = load double, double* %"gep'ipg", align 4
; CHECK-NEXT:   store double 0.000000e+00, double* %"gep'ipg", align 4
; CHECK-NEXT:   %m0diffeld = fmul fast double %0, %x
; CHECK-NEXT:   %1 = atomicrmw fadd double* %"g0'ipg", double %m0diffeld monotonic
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
