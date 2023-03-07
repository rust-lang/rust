; RUN: if [ %llvmver -ge 9 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -simplifycfg -adce -instsimplify -S | FileCheck %s; fi

source_filename = "text"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:10:11:12:13"
target triple = "x86_64-pc-linux-gnu"

define void @tester(i64 addrspace(12)* %i2, i64 addrspace(13)* %i7) {
entry:
  %i3 = load i64, i64 addrspace(12)* %i2, align 8, !dbg !5, !tbaa !15
  store i64 %i3, i64 addrspace(13)* %i7, align 8, !dbg !35, !tbaa !40
  ret void
}

declare void @__enzyme_reverse(...)

define void @test_derivative(i64 addrspace(12)* %x, i64 addrspace(12)* %dx1, i64 addrspace(12)* %dx2, {} addrspace(13)* %y, {} addrspace(13)* %dy1, {} addrspace(13)* %dy2,  i8* %tape) {
entry:
  call void (...) @__enzyme_reverse(void (i64 addrspace(12)*, i64 addrspace(13)*)* nonnull @tester, metadata !"enzyme_width", i64 2, metadata !"enzyme_dup", i64 addrspace(12)* %x, i64 addrspace(12)* %dx1, i64 addrspace(12)* %dx2, metadata !"enzyme_dup", {} addrspace(13)* %y, {} addrspace(13)* %dy1, {} addrspace(13)* %dy2, i8* %tape)
  ret void
}

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !3, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !4, nameTableKind: None)
!3 = !DIFile(filename: "/mnt/Data/git/Enzyme.jl/revjac.jl", directory: ".")
!4 = !{}
!5 = !DILocation(line: 33, scope: !6, inlinedAt: !9)
!6 = distinct !DISubprogram(name: "getproperty;", linkageName: "getproperty", scope: !7, file: !7, type: !8, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !4)
!7 = !DIFile(filename: "Base.jl", directory: ".")
!8 = !DISubroutineType(types: !4)
!9 = distinct !DILocation(line: 56, scope: !10, inlinedAt: !12)
!10 = distinct !DISubprogram(name: "getindex;", linkageName: "getindex", scope: !11, file: !11, type: !8, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !4)
!11 = !DIFile(filename: "refvalue.jl", directory: ".")
!12 = distinct !DILocation(line: 6, scope: !13, inlinedAt: !14)
!13 = distinct !DISubprogram(name: "batchbwd", linkageName: "julia_batchbwd_1599", scope: null, file: !3, line: 5, type: !8, scopeLine: 5, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !4)
!14 = distinct !DILocation(line: 0, scope: !13)
!15 = !{!16, !16, i64 0}
!16 = !{!"double", !17, i64 0}
!17 = !{!"jtbaa_value", !18, i64 0}
!18 = !{!"jtbaa_data", !19, i64 0}
!19 = !{!"jtbaa", !20, i64 0}
!20 = !{!"jtbaa"}
!21 = !DILocation(line: 448, scope: !22, inlinedAt: !24)
!22 = distinct !DISubprogram(name: "Array;", linkageName: "Array", scope: !23, file: !23, type: !8, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !4)
!23 = !DIFile(filename: "boot.jl", directory: ".")
!24 = distinct !DILocation(line: 457, scope: !22, inlinedAt: !25)
!25 = distinct !DILocation(line: 785, scope: !26, inlinedAt: !28)
!26 = distinct !DISubprogram(name: "similar;", linkageName: "similar", scope: !27, file: !27, type: !8, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !4)
!27 = !DIFile(filename: "abstractarray.jl", directory: ".")
!28 = distinct !DILocation(line: 784, scope: !26, inlinedAt: !29)
!29 = distinct !DILocation(line: 672, scope: !30, inlinedAt: !32)
!30 = distinct !DISubprogram(name: "_array_for;", linkageName: "_array_for", scope: !31, file: !31, type: !8, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !4)
!31 = !DIFile(filename: "array.jl", directory: ".")
!32 = distinct !DILocation(line: 670, scope: !30, inlinedAt: !33)
!33 = distinct !DILocation(line: 108, scope: !34, inlinedAt: !12)
!34 = distinct !DISubprogram(name: "vect;", linkageName: "vect", scope: !31, file: !31, type: !8, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !4)
!35 = !DILocation(line: 843, scope: !36, inlinedAt: !33)
!36 = distinct !DISubprogram(name: "setindex!;", linkageName: "setindex!", scope: !31, file: !31, type: !8, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !4)
!37 = !{!38, !38, i64 0}
!38 = !{!"jtbaa_arrayptr", !39, i64 0}
!39 = !{!"jtbaa_array", !19, i64 0}
!40 = !{!41, !41, i64 0}
!41 = !{!"jtbaa_arraybuf", !18, i64 0}

; CHECK: define internal void @diffe2tester(i64 addrspace(12)* %i2, [2 x i64 addrspace(12)*] %"i2'", i64 addrspace(13)* %i7, [2 x i64 addrspace(13)*] %"i7'", i8* %tapeArg) 
; CHECK-NEXT: entry:
; CHECK-NEXT:   tail call void @free(i8* nonnull %tapeArg)
; CHECK-NEXT:   %"i3'de" = alloca [2 x i64]
; CHECK-NEXT:   store [2 x i64] zeroinitializer, [2 x i64]* %"i3'de"
; CHECK-NEXT:   %0 = extractvalue [2 x i64 addrspace(13)*] %"i7'", 0
; CHECK-NEXT:   %1 = load i64, i64 addrspace(13)* %0
; CHECK-NEXT:   %2 = extractvalue [2 x i64 addrspace(13)*] %"i7'", 1
; CHECK-NEXT:   %3 = load i64, i64 addrspace(13)* %2
; CHECK-NEXT:   %4 = extractvalue [2 x i64 addrspace(13)*] %"i7'", 0
; CHECK-NEXT:   store i64 0, i64 addrspace(13)* %4
; CHECK-NEXT:   %5 = extractvalue [2 x i64 addrspace(13)*] %"i7'", 1
; CHECK-NEXT:   store i64 0, i64 addrspace(13)* %5
; CHECK-NEXT:   %6 = getelementptr inbounds [2 x i64], [2 x i64]* %"i3'de", i32 0, i32 0
; CHECK-NEXT:   %7 = load i64, i64* %6
; CHECK-NEXT:   %8 = bitcast i64 %7 to double
; CHECK-NEXT:   %9 = bitcast i64 %1 to double
; CHECK-NEXT:   %10 = fadd fast double %8, %9
; CHECK-NEXT:   %11 = bitcast double %10 to i64
; CHECK-NEXT:   store i64 %11, i64* %6
; CHECK-NEXT:   %12 = getelementptr inbounds [2 x i64], [2 x i64]* %"i3'de", i32 0, i32 1
; CHECK-NEXT:   %13 = load i64, i64* %12
; CHECK-NEXT:   %14 = bitcast i64 %13 to double
; CHECK-NEXT:   %15 = bitcast i64 %3 to double
; CHECK-NEXT:   %16 = fadd fast double %14, %15
; CHECK-NEXT:   %17 = bitcast double %16 to i64
; CHECK-NEXT:   store i64 %17, i64* %12
; CHECK-NEXT:   %18 = load [2 x i64], [2 x i64]* %"i3'de"
; CHECK-NEXT:   store [2 x i64] zeroinitializer, [2 x i64]* %"i3'de"
; CHECK-NEXT:   %19 = extractvalue [2 x i64 addrspace(12)*] %"i2'", 0
; CHECK-NEXT:   %20 = bitcast i64 addrspace(12)* %19 to double addrspace(12)*
; CHECK-NEXT:   %21 = extractvalue [2 x i64 addrspace(12)*] %"i2'", 1
; CHECK-NEXT:   %22 = bitcast i64 addrspace(12)* %21 to double addrspace(12)*
; CHECK-NEXT:   %23 = extractvalue [2 x i64] %18, 0
; CHECK-DAG:   %[[i24:.+]] = bitcast i64 %23 to double
; CHECK-NEXT:   %[[i28:.+]] = extractvalue [2 x i64] %18, 1
; CHECK-DAG:   %[[i29:.+]] = bitcast i64 %[[i28]] to double
; CHECK-DAG:   %[[i25:.+]] = load double, double addrspace(12)* %20
; CHECK-NEXT:   %[[i26:.+]] = fadd fast double %[[i25]], %[[i24]]
; CHECK-NEXT:   store double %[[i26]], double addrspace(12)* %20
; CHECK-DAG:   %[[i30:.+]] = load double, double addrspace(12)* %22
; CHECK-NEXT:   %[[i31:.+]] = fadd fast double %[[i30]], %[[i29]]
; CHECK-NEXT:   store double %[[i31]], double addrspace(12)* %22
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

