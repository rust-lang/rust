; RUN: %opt < %s %loadEnzyme -enzyme-rust-type -print-type-analysis -type-analysis-func=callee -o /dev/null | FileCheck %s


%"std::vec::Vec<f32>" = type { { i32*, i64 }, i64 }

declare void @llvm.dbg.declare(metadata, metadata, metadata)

define internal void @callee(i8* %arg) {
start:
  %t = bitcast i8* %arg to %"std::vec::Vec<f32>"*
  call void @llvm.dbg.declare(metadata %"std::vec::Vec<f32>"* %t, metadata !1568, metadata !DIExpression()), !dbg !1569
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
!19 = !DIFile(filename: "rustvec.rs", directory: "/home/nomanous/Space/Tmp/Enzyme")
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
!42 = !DINamespace(name: "ptr", scope: !23)
!52 = !DINamespace(name: "unique", scope: !42)
!57 = !DINamespace(name: "marker", scope: !23)
!84 = !DIBasicType(name: "usize", size: 64, encoding: DW_ATE_unsigned)
!241 = !DIBasicType(name: "f32", size: 32, encoding: DW_ATE_float)
!248 = !{!249}
!249 = !DITemplateTypeParameter(name: "T", type: !241)
!392 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "*const f32", baseType: !241, size: 64, align: 64, dwarfAddressSpace: 0)
!460 = !DICompositeType(tag: DW_TAG_structure_type, name: "Vec<f32, alloc::alloc::Global>", scope: !461, file: !2, size: 192, align: 64, elements: !463, templateParams: !478, identifier: "c35725c530882b702a86d0f2bdc58bb0")
!461 = !DINamespace(name: "vec", scope: !462)
!462 = !DINamespace(name: "alloc", scope: null)
!463 = !{!464, !480}
!464 = !DIDerivedType(tag: DW_TAG_member, name: "buf", scope: !460, file: !2, baseType: !465, size: 128, align: 64)
!465 = !DICompositeType(tag: DW_TAG_structure_type, name: "RawVec<f32, alloc::alloc::Global>", scope: !466, file: !2, size: 128, align: 64, elements: !467, templateParams: !478, identifier: "73e42376b1c14713ccaf5990ac80da8c")
!466 = !DINamespace(name: "raw_vec", scope: !462)
!467 = !{!468, !474, !475}
!468 = !DIDerivedType(tag: DW_TAG_member, name: "ptr", scope: !465, file: !2, baseType: !469, size: 64, align: 64)
!469 = !DICompositeType(tag: DW_TAG_structure_type, name: "Unique<f32>", scope: !52, file: !2, size: 64, align: 64, elements: !470, templateParams: !248, identifier: "ef9c766bc5851286d5bbc6ad0619c0da")
!470 = !{!471, !472}
!471 = !DIDerivedType(tag: DW_TAG_member, name: "pointer", scope: !469, file: !2, baseType: !392, size: 64, align: 64)
!472 = !DIDerivedType(tag: DW_TAG_member, name: "_marker", scope: !469, file: !2, baseType: !473, align: 8)
!473 = !DICompositeType(tag: DW_TAG_structure_type, name: "PhantomData<f32>", scope: !57, file: !2, align: 8, elements: !4, templateParams: !248, identifier: "4069b0e897fcfe863aa50b83cce43b8a")
!474 = !DIDerivedType(tag: DW_TAG_member, name: "cap", scope: !465, file: !2, baseType: !84, size: 64, align: 64, offset: 64)
!475 = !DIDerivedType(tag: DW_TAG_member, name: "alloc", scope: !465, file: !2, baseType: !476, align: 8)
!476 = !DICompositeType(tag: DW_TAG_structure_type, name: "Global", scope: !477, file: !2, align: 8, elements: !4, templateParams: !4, identifier: "c2e1c48749aceb2535bbc8720e314c12")
!477 = !DINamespace(name: "alloc", scope: !462)
!478 = !{!249, !479}
!479 = !DITemplateTypeParameter(name: "A", type: !476)
!480 = !DIDerivedType(tag: DW_TAG_member, name: "len", scope: !460, file: !2, baseType: !84, size: 64, align: 64, offset: 128)
!1562 = distinct !DISubprogram(name: "callee", linkageName: "_ZN7rustvec6callee17ha569ab73a448e015E", scope: !1564, file: !1563, line: 1, type: !1565, scopeLine: 1, flags: DIFlagPrototyped, unit: !18, templateParams: !4, retainedNodes: !1567)
!1563 = !DIFile(filename: "rustvec.rs", directory: "/home/nomanous/Space/Tmp/Enzyme", checksumkind: CSK_MD5, checksum: "cea41987509dc75ba27672e273ebd3a3")
!1564 = !DINamespace(name: "rustvec", scope: null)
!1565 = !DISubroutineType(types: !1566)
!1566 = !{!241, !460}
!1567 = !{!1568}
!1568 = !DILocalVariable(name: "t", arg: 1, scope: !1562, file: !1563, line: 1, type: !460)
!1569 = !DILocation(line: 1, column: 11, scope: !1562)

; CHECK: callee - {} |{[-1]:Pointer}:{} 
; CHECK-NEXT: i8* %arg: {[-1]:Pointer, [-1,0]:Pointer, [-1,0,-1]:Float@float, [-1,8]:Integer, [-1,16]:Integer}
; CHECK-NEXT: start
; CHECK-NEXT:   %t = bitcast i8* %arg to %"std::vec::Vec<f32>"*: {[-1]:Pointer, [-1,0]:Pointer, [-1,0,-1]:Float@float, [-1,8]:Integer, [-1,16]:Integer}
; CHECK-NEXT:   call void @llvm.dbg.declare(metadata %"std::vec::Vec<f32>"* %t, metadata !38, metadata !DIExpression()), !dbg !74: {}
; CHECK-NEXT:   ret void: {}
