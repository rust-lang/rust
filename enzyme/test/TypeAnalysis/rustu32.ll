; RUN: %opt < %s %loadEnzyme -enzyme-rust-type -print-type-analysis -type-analysis-func=callee -o /dev/null | FileCheck %s


declare void @llvm.dbg.declare(metadata, metadata, metadata)

define internal void @callee(i8* %x) {
start:
  %x.dbg.spill = bitcast i8* %x to i32*
  call void @llvm.dbg.declare(metadata i32* %x.dbg.spill, metadata !171, metadata !DIExpression()), !dbg !172
  ret void
}

!llvm.module.flags = !{!14, !15, !16, !17}
!llvm.dbg.cu = !{!18}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "vtable", scope: null, file: !2, type: !3, isLocal: true, isDefinition: true)
!2 = !DIFile(filename: "<unknown>", directory: "")
!3 = !DICompositeType(tag: DW_TAG_structure_type, name: "vtable", file: !2, align: 64, flags: DIFlagArtificial, elements: !4, vtableHolder: !5, identifier: "vtable")
!4 = !{}
!5 = !DICompositeType(tag: DW_TAG_structure_type, name: "closure-0", scope: !6, file: !2, size: 64, align: 64, elements: !9, templateParams: !4, identifier: "e5f9fb156939f3ec2778ebca8e63f246")
!6 = !DINamespace(name: "lang_start", scope: !7)
!7 = !DINamespace(name: "rt", scope: !8)
!8 = !DINamespace(name: "std", scope: null)
!9 = !{!10}
!10 = !DIDerivedType(tag: DW_TAG_member, name: "__0", scope: !5, file: !2, baseType: !11, size: 64, align: 64)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "fn()", baseType: !12, size: 64, align: 64, dwarfAddressSpace: 0)
!12 = !DISubroutineType(types: !13)
!13 = !{null}
!14 = !{i32 7, !"PIC Level", i32 2}
!15 = !{i32 7, !"PIE Level", i32 2}
!16 = !{i32 2, !"RtLibUseGOT", i32 1}
!17 = !{i32 2, !"Debug Info Version", i32 3}
!18 = distinct !DICompileUnit(language: DW_LANG_Rust, file: !19, producer: "clang LLVM (rustc version 1.54.0 (a178d0322 2021-07-26))", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !20)
!19 = !DIFile(filename: "rustu16.rs", directory: "/home/nomanous/Space/Tmp/EnzymeTest")
!20 = !{!0}
!163 = distinct !DISubprogram(name: "callee", linkageName: "callee", scope: !165, file: !164, line: 1, type: !166, scopeLine: 1, flags: DIFlagPrototyped, unit: !18, templateParams: !4, retainedNodes: !170)
!164 = !DIFile(filename: "rustu16.rs", directory: "/home/nomanous/Space/Tmp/EnzymeTest", checksumkind: CSK_MD5, checksum: "51b14a20936bd08eca2ebbc55c3aeb26")
!165 = !DINamespace(name: "rustu16", scope: null)
!166 = !DISubroutineType(types: !167)
!167 = !{!168, !169}
!168 = !DIBasicType(name: "f32", size: 32, encoding: DW_ATE_float)
!169 = !DIBasicType(name: "u32", size: 16, encoding: DW_ATE_unsigned)
!170 = !{!171}
!171 = !DILocalVariable(name: "x", arg: 1, scope: !163, file: !164, line: 1, type: !169)
!172 = !DILocation(line: 1, column: 11, scope: !163)

; CHECK: callee - {} |{[-1]:Pointer}:{}
; CHECK-NEXT: i8* %x: {[-1]:Pointer, [-1,0]:Integer}
; CHECK-NEXT: start
; CHECK-NEXT:   %x.dbg.spill = bitcast i8* %x to i32*: {[-1]:Pointer, [-1,0]:Integer}
; CHECK-NEXT:   call void @llvm.dbg.declare(metadata i32* %x.dbg.spill, metadata !21, metadata !DIExpression()), !dbg !30: {}
; CHECK-NEXT:   ret void: {}
