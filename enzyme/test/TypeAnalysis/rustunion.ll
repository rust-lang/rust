
; RUN: %opt < %s %loadEnzyme -enzyme-rust-type -print-type-analysis -type-analysis-func=callee -o /dev/null | FileCheck %s

%T = type { [2 x i32] }

declare void @llvm.dbg.declare(metadata, metadata, metadata)

define internal void @callee(i8* %arg) {
start:
  %t = bitcast i8* %arg to %T*
  call void @llvm.dbg.declare(metadata %T* %t, metadata !334, metadata !DIExpression()), !dbg !335
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
!18 = distinct !DICompileUnit(language: DW_LANG_Rust, file: !19, producer: "clang LLVM (rustc version 1.54.0 (a178d0322 2021-07-26))", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !20, globals: !37)
!19 = !DIFile(filename: "rustunion.rs", directory: "/home/nomanous/Space/Tmp/EnzymeTest")
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
!81 = !DIBasicType(name: "i32", size: 32, encoding: DW_ATE_signed)
!156 = !DIBasicType(name: "f32", size: 32, encoding: DW_ATE_float)
!316 = distinct !DISubprogram(name: "callee", linkageName: "callee", scope: !318, file: !317, line: 18, type: !319, scopeLine: 18, flags: DIFlagPrototyped, unit: !18, templateParams: !4, retainedNodes: !333)
!317 = !DIFile(filename: "rustunion.rs", directory: "/home/nomanous/Space/Tmp/EnzymeTest", checksumkind: CSK_MD5, checksum: "c4cbdbc7b77a9275d5c8cacf582f049b")
!318 = !DINamespace(name: "rustunion", scope: null)
!319 = !DISubroutineType(types: !320)
!320 = !{!156, !321}
!321 = !DICompositeType(tag: DW_TAG_union_type, name: "T", scope: !318, file: !2, size: 64, align: 32, elements: !322, templateParams: !4, identifier: "5ba7121423471e9fe57f5fe846dc6003")
!322 = !{!323, !328}
!323 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !321, file: !2, baseType: !324, size: 64, align: 32)
!324 = !DICompositeType(tag: DW_TAG_structure_type, name: "U1", scope: !318, file: !2, size: 64, align: 32, elements: !325, templateParams: !4, identifier: "8409f3525edb1423db1f91d1d80ebe5e")
!325 = !{!326, !327}
!326 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !324, file: !2, baseType: !156, size: 32, align: 32)
!327 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !324, file: !2, baseType: !81, size: 32, align: 32, offset: 32)
!328 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !321, file: !2, baseType: !329, size: 64, align: 32)
!329 = !DICompositeType(tag: DW_TAG_structure_type, name: "U2", scope: !318, file: !2, size: 64, align: 32, elements: !330, templateParams: !4, identifier: "9a6f179d41ffa3284bb4a822cd154cbf")
!330 = !{!331, !332}
!331 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !329, file: !2, baseType: !156, size: 32, align: 32)
!332 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !329, file: !2, baseType: !156, size: 32, align: 32, offset: 32)
!333 = !{!334}
!334 = !DILocalVariable(name: "t", arg: 1, scope: !316, file: !317, line: 18, type: !321)
!335 = !DILocation(line: 18, column: 18, scope: !316)

; CHECK: callee - {} |{[-1]:Pointer}:{} 
; CHECK-NEXT: i8* %arg: {[-1]:Pointer, [-1,0]:Float@float}
; CHECK-NEXT: start
; CHECK-NEXT:   %t = bitcast i8* %arg to %T*: {[-1]:Pointer, [-1,0]:Float@float}
; CHECK-NEXT:   call void @llvm.dbg.declare(metadata %T* %t, metadata !38, metadata !DIExpression()), !dbg !59: {}
; CHECK-NEXT:   ret void: {}
