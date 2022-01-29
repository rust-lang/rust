; RUN: %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=_Z4axpyfPfS_ -o /dev/null | FileCheck %s
; ModuleID = 'cuda.cu'
source_filename = "cuda.cu"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

%struct.cudaFuncAttributes = type { i64, i64, i64, i32, i32, i32, i32, i32, i32, i32 }

; Function Attrs: nofree nounwind
define dso_local void @_Z4axpyfPfS_() {
entry:
  %tix = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  ret void
}

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() #2

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
!3 = !{void ()* @_Z4axpyfPfS_, !"kernel", i32 1}
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

; CHECK: _Z4axpyfPfS_ - {} |
; CHECK-NEXT: entry
; CHECK-NEXT:   %tix = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(): {[-1]:Integer}
; CHECK-NEXT:   ret void: {}
