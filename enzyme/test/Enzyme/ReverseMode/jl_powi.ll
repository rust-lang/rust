; RUN: if [ %llvmver -ge 13 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
source_filename = "text"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

define private fastcc double @julia___2797(double %0, i64 signext %1) unnamed_addr #0 !dbg !7 {
top:
  switch i64 %1, label %L20 [
    i64 -1, label %L3
    i64 0, label %L7
    i64 1, label %L7.fold.split
    i64 2, label %L13
    i64 3, label %L17
  ], !dbg !9

L3:                                               ; preds = %top
  %2 = fdiv double 1.000000e+00, %0, !dbg !10
  ret double %2, !dbg !9

L7.fold.split:                                    ; preds = %top
  br label %L7, !dbg !16

L7:                                               ; preds = %top, %L7.fold.split
  %merge = phi double [ 1.000000e+00, %top ], [ %0, %L7.fold.split ]
  ret double %merge, !dbg !16

L13:                                              ; preds = %top
  %3 = fmul double %0, %0, !dbg !17
  ret double %3, !dbg !19

L17:                                              ; preds = %top
  %4 = fmul double %0, %0, !dbg !20
  %5 = fmul double %4, %0, !dbg !20
  ret double %5, !dbg !24

L20:                                              ; preds = %top
  %6 = sitofp i64 %1 to double, !dbg !25
  %7 = call double @llvm.pow.f64(double %0, double %6), !dbg !27
  ret double %7, !dbg !27
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare double @llvm.pow.f64(double, double) #1

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double, i64)*, ...)

; Function Attrs: alwaysinline nosync readnone
define double @julia_f_2794(double %0, i64 signext %1) local_unnamed_addr #2 !dbg !28 {
entry:
  %2 = call fastcc double @julia___2797(double %0, i64 signext %1) #5, !dbg !29
  ret double %2
}

define double @test_derivative(double %x, i64 %y) {
entry:
  %0 = tail call double (double (double, i64)*, ...) @__enzyme_autodiff(double (double, i64)* nonnull @julia_f_2794, double %x, i64 %y)
  ret double %0
}

; CHECK: define internal { double } @diffejulia_f_2794(double %0, i64 signext %1, double %differeturn) local_unnamed_addr #5 !dbg !35 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %2 = sub i64 %1, 1
; CHECK-NEXT:   %3 = call fast fastcc double @julia___2797(double %0, i64 %2), !dbg !36
; CHECK-NEXT:   %4 = sitofp i64 %1 to double
; CHECK-NEXT:   %5 = fmul fast double %differeturn, %3
; CHECK-NEXT:   %6 = fmul fast double %5, %4
; CHECK-NEXT:   %7 = insertvalue { double } undef, double %6, 0
; CHECK-NEXT:   ret { double } %7
; CHECK-NEXT: }

; Function Attrs: inaccessiblemem_or_argmemonly
declare void @jl_gc_queue_root({} addrspace(10)*) #3

; Function Attrs: allocsize(1)
declare noalias nonnull {} addrspace(10)* @jl_gc_pool_alloc(i8*, i32, i32) #4

; Function Attrs: allocsize(1)
declare noalias nonnull {} addrspace(10)* @jl_gc_big_alloc(i8*, i64) #4

attributes #0 = { noinline nosync readnone "enzyme_math"="powi" "enzyme_shouldrecompute"="powi" "probe-stack"="inline-asm" }
attributes #1 = { nofree nosync nounwind readnone speculatable willreturn }
attributes #2 = { alwaysinline nosync readnone "probe-stack"="inline-asm" }
attributes #3 = { inaccessiblemem_or_argmemonly }
attributes #4 = { allocsize(1) }
attributes #5 = { "probe-stack"="inline-asm" }

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2, !5}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !3, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !4, nameTableKind: None)
!3 = !DIFile(filename: "math.jl", directory: ".")
!4 = !{}
!5 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !6, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !4, nameTableKind: None)
!6 = !DIFile(filename: "REPL[3]", directory: ".")
!7 = distinct !DISubprogram(name: "^", linkageName: "julia_^_2797", scope: null, file: !3, line: 922, type: !8, scopeLine: 922, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !4)
!8 = !DISubroutineType(types: !4)
!9 = !DILocation(line: 923, scope: !7)
!10 = !DILocation(line: 408, scope: !11, inlinedAt: !13)
!11 = distinct !DISubprogram(name: "/;", linkageName: "/", scope: !12, file: !12, type: !8, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !4)
!12 = !DIFile(filename: "float.jl", directory: ".")
!13 = !DILocation(line: 243, scope: !14, inlinedAt: !9)
!14 = distinct !DISubprogram(name: "inv;", linkageName: "inv", scope: !15, file: !15, type: !8, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !4)
!15 = !DIFile(filename: "number.jl", directory: ".")
!16 = !DILocation(line: 924, scope: !7)
!17 = !DILocation(line: 405, scope: !18, inlinedAt: !19)
!18 = distinct !DISubprogram(name: "*;", linkageName: "*", scope: !12, file: !12, type: !8, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !4)
!19 = !DILocation(line: 926, scope: !7)
!20 = !DILocation(line: 405, scope: !18, inlinedAt: !21)
!21 = !DILocation(line: 655, scope: !22, inlinedAt: !24)
!22 = distinct !DISubprogram(name: "*;", linkageName: "*", scope: !23, file: !23, type: !8, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !4)
!23 = !DIFile(filename: "operators.jl", directory: ".")
!24 = !DILocation(line: 927, scope: !7)
!25 = !DILocation(line: 146, scope: !26, inlinedAt: !27)
!26 = distinct !DISubprogram(name: "Float64;", linkageName: "Float64", scope: !12, file: !12, type: !8, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !4)
!27 = !DILocation(line: 928, scope: !7)
!28 = distinct !DISubprogram(name: "f", linkageName: "julia_f_2794", scope: null, file: !6, line: 1, type: !8, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !5, retainedNodes: !4)
!29 = !DILocation(line: 1, scope: !28, inlinedAt: !30)
!30 = distinct !DILocation(line: 0, scope: !28)
