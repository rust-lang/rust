; RUN: %opt < %s %loadEnzyme -enzyme-rust-type -print-type-analysis -type-analysis-func=callee -o /dev/null | FileCheck %s

declare void @llvm.dbg.declare(metadata, metadata, metadata)

define internal void @callee(i8* %t) {
start:
  %t.dbg.spill = bitcast i8* %t to float**
  call void @llvm.dbg.declare(metadata float** %t.dbg.spill, metadata !380, metadata !DIExpression()), !dbg !381
  ret void
}

!llvm.module.flags = !{!14, !15, !16, !17}
!llvm.dbg.cu = !{!18}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "vtable", scope: null, file: !2, type: !3, isLocal: true, isDefinition: true)
!2 = !DIFile(filename: "<unknown>", directory: "")
!3 = !DICompositeType(tag: DW_TAG_structure_type, name: "vtable", file: !2, align: 64, flags: DIFlagArtificial, elements: !4, vtableHolder: !5, identifier: "vtable")
!4 = !{}
!5 = !DICompositeType(tag: DW_TAG_structure_type, name: "{closure#0}", scope: !6, file: !2, size: 64, align: 64, elements: !9, templateParams: !4, identifier: "c211ca2a5a4c8dd717d1e5fba4a6ae0")
!6 = !DINamespace(name: "lang_start", scope: !7)
!7 = !DINamespace(name: "rt", scope: !8)
!8 = !DINamespace(name: "std", scope: null)
!9 = !{!10}
!10 = !DIDerivedType(tag: DW_TAG_member, name: "main", scope: !5, file: !2, baseType: !11, size: 64, align: 64)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "fn()", baseType: !12, size: 64, align: 64, dwarfAddressSpace: 0)
!12 = !DISubroutineType(types: !13)
!13 = !{null}
!14 = !{i32 7, !"PIC Level", i32 2}
!15 = !{i32 7, !"PIE Level", i32 2}
!16 = !{i32 2, !"RtLibUseGOT", i32 1}
!17 = !{i32 2, !"Debug Info Version", i32 3}
!18 = distinct !DICompileUnit(language: DW_LANG_Rust, file: !19, producer: "clang LLVM (rustc version 1.56.0 (09c42c458 2021-10-18))", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !20, globals: !37)
!19 = !DIFile(filename: "rustconstpointer.rs", directory: "/home/nomanous/Space/Tmp/Enzyme")
!20 = !{!21, !28}
!21 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "Result", scope: !22, file: !2, baseType: !24, size: 8, align: 8, elements: !25)
!22 = !DINamespace(name: "result", scope: !23)
!23 = !DINamespace(name: "core", scope: null)
!24 = !DIBasicType(name: "u8", size: 8, encoding: DW_ATE_unsigned)
!25 = !{!26, !27}
!26 = !DIEnumerator(name: "Ok", value: 0)
!27 = !DIEnumerator(name: "Err", value: 1)
!28 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "Alignment", scope: !29, file: !2, baseType: !24, size: 8, align: 8, elements: !32)
!29 = !DINamespace(name: "v1", scope: !30)
!30 = !DINamespace(name: "rt", scope: !31)
!31 = !DINamespace(name: "fmt", scope: !23)
!32 = !{!33, !34, !35, !36}
!33 = !DIEnumerator(name: "Left", value: 0)
!34 = !DIEnumerator(name: "Right", value: 1)
!35 = !DIEnumerator(name: "Center", value: 2)
!36 = !DIEnumerator(name: "Unknown", value: 3)
!37 = !{!0}
!156 = !DIBasicType(name: "f32", size: 32, encoding: DW_ATE_float)
!373 = distinct !DISubprogram(name: "callee", linkageName: "callee", scope: !375, file: !374, line: 1, type: !376, scopeLine: 1, flags: DIFlagPrototyped, unit: !18, templateParams: !4, retainedNodes: !379)
!374 = !DIFile(filename: "rustconstpointer.rs", directory: "/home/nomanous/Space/Tmp/Enzyme", checksumkind: CSK_MD5, checksum: "79fab3728ac9ae6db905f8a4e0a87d75")
!375 = !DINamespace(name: "rustconstpointer", scope: null)
!376 = !DISubroutineType(types: !377)
!377 = !{!156, !378}
!378 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "*mut f32", baseType: !156, size: 64, align: 64, dwarfAddressSpace: 0)
!379 = !{!380}
!380 = !DILocalVariable(name: "t", arg: 1, scope: !373, file: !374, line: 1, type: !378)
!381 = !DILocation(line: 1, column: 18, scope: !373)

; CHECK: callee - {} |{[-1]:Pointer}:{} 
; CHECK-NEXT: i8* %t: {[-1]:Pointer, [-1,0]:Pointer, [-1,0,-1]:Float@float}
; CHECK-NEXT: start
; CHECK-NEXT:   %t.dbg.spill = bitcast i8* %t to float**: {[-1]:Pointer, [-1,0]:Pointer, [-1,0,-1]:Float@float}
; CHECK-NEXT:   call void @llvm.dbg.declare(metadata float** %t.dbg.spill, metadata !38, metadata !DIExpression()), !dbg !47: {}
; CHECK-NEXT:   ret void: {}
