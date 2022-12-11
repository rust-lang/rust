; RUN: if [ %llvmver -ge 8 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 8 ]; then %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s; fi

source_filename = "examples/solids/problems/finite-strain-neo-hookean-initial-ad.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define double @caller(double %div13) {
entry:
  %res = call double (i8*, ...) @__enzyme_fwddiff(i8* bitcast (double (double)* @computeS to i8*), double %div13, double 1.000000e+00)
  ret double %res
}

define internal double @computeS(double %E2work) {
entry:
  %cmp.i = fcmp olt double %E2work, 0xBFD2BEC333018866
  br i1 %cmp.i, label %if.then.i, label %if.else.i

if.then.i:                                        ; preds = %entry
  %mul.i128 = fmul double %E2work, 2.000000e+00
  br label %log1p_series_shifted.exit

if.else.i:                                        ; preds = %entry
  %cmp6.i = fcmp ogt double %E2work, 0x3FDA827999FCEF34
  br label %log1p_series_shifted.exit

log1p_series_shifted.exit:                        ; preds = %if.else.i, %if.then.i
  %x.addr.0.i = phi double [ %mul.i128, %if.then.i ], [ %E2work, %if.else.i ]
  call void @llvm.dbg.value(metadata double %x.addr.0.i, metadata !178, metadata !DIExpression()), !dbg !188
  ret double %x.addr.0.i
}

declare double @__enzyme_fwddiff(i8*, ...)

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!172, !173, !174, !175, !176}
!llvm.ident = !{!177}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 13.0.1 (git@github.com:llvm/llvm-project cf15ccdeb6d5254ee7d46c7535c29200003a3880)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !42, globals: !78, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "examples/solids/problems/finite-strain-neo-hookean-initial-ad.c", directory: "/home/wmoses/git/Enzyme/enzyme/sc/libCEED")
!2 = !{!3, !9, !14, !25, !31, !36}
!3 = !DICompositeType(tag: DW_TAG_enumeration_type, file: !4, line: 489, baseType: !5, size: 32, elements: !6)
!4 = !DIFile(filename: "include/ceed/ceed.h", directory: "/home/wmoses/git/Enzyme/enzyme/sc/libCEED")
!5 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!6 = !{!7, !8}
!7 = !DIEnumerator(name: "CEED_GAUSS", value: 0)
!8 = !DIEnumerator(name: "CEED_GAUSS_LOBATTO", value: 1)
!9 = !DICompositeType(tag: DW_TAG_enumeration_type, file: !10, line: 170, baseType: !5, size: 32, elements: !11)
!10 = !DIFile(filename: "petsc/include/petscsystypes.h", directory: "/home/wmoses/git/Enzyme/enzyme/sc/libCEED")
!11 = !{!12, !13}
!12 = !DIEnumerator(name: "PETSC_FALSE", value: 0)
!13 = !DIEnumerator(name: "PETSC_TRUE", value: 1)
!14 = !DICompositeType(tag: DW_TAG_enumeration_type, file: !15, line: 5, baseType: !5, size: 32, elements: !16)
!15 = !DIFile(filename: "examples/solids/problems/../problems/cl-problems.h", directory: "/home/wmoses/git/Enzyme/enzyme/sc/libCEED")
!16 = !{!17, !18, !19, !20, !21, !22, !23, !24}
!17 = !DIEnumerator(name: "ELAS_LINEAR", value: 0)
!18 = !DIEnumerator(name: "ELAS_SS_NH", value: 1)
!19 = !DIEnumerator(name: "ELAS_FSInitial_NH1", value: 2)
!20 = !DIEnumerator(name: "ELAS_FSInitial_NH2", value: 3)
!21 = !DIEnumerator(name: "ELAS_FSInitial_NH_AD", value: 4)
!22 = !DIEnumerator(name: "ELAS_FSCurrent_NH1", value: 5)
!23 = !DIEnumerator(name: "ELAS_FSCurrent_NH2", value: 6)
!24 = !DIEnumerator(name: "ELAS_FSInitial_MR1", value: 7)
!25 = !DICompositeType(tag: DW_TAG_enumeration_type, file: !26, line: 12, baseType: !5, size: 32, elements: !27)
!26 = !DIFile(filename: "examples/solids/problems/../include/../include/structs.h", directory: "/home/wmoses/git/Enzyme/enzyme/sc/libCEED")
!27 = !{!28, !29, !30}
!28 = !DIEnumerator(name: "FORCE_NONE", value: 0)
!29 = !DIEnumerator(name: "FORCE_CONST", value: 1)
!30 = !DIEnumerator(name: "FORCE_MMS", value: 2)
!31 = !DICompositeType(tag: DW_TAG_enumeration_type, file: !26, line: 26, baseType: !5, size: 32, elements: !32)
!32 = !{!33, !34, !35}
!33 = !DIEnumerator(name: "MULTIGRID_LOGARITHMIC", value: 0)
!34 = !DIEnumerator(name: "MULTIGRID_UNIFORM", value: 1)
!35 = !DIEnumerator(name: "MULTIGRID_NONE", value: 2)
!36 = !DICompositeType(tag: DW_TAG_enumeration_type, file: !37, line: 683, baseType: !5, size: 32, elements: !38)
!37 = !DIFile(filename: "petsc/include/petscerror.h", directory: "/home/wmoses/git/Enzyme/enzyme/sc/libCEED")
!38 = !{!39, !40, !41}
!39 = !DIEnumerator(name: "PETSC_ERROR_INITIAL", value: 0)
!40 = !DIEnumerator(name: "PETSC_ERROR_REPEAT", value: 1)
!41 = !DIEnumerator(name: "PETSC_ERROR_IN_CXX", value: 2)
!42 = !{!43, !46, !51, !60, !63, !65, !67, !69, !68, !76, !77}
!43 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !44, size: 64)
!44 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !45)
!45 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!46 = !DIDerivedType(tag: DW_TAG_typedef, name: "PetscVoidFunction", file: !47, line: 1475, baseType: !48)
!47 = !DIFile(filename: "petsc/include/petscsys.h", directory: "/home/wmoses/git/Enzyme/enzyme/sc/libCEED")
!48 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !49, size: 64)
!49 = !DISubroutineType(types: !50)
!50 = !{null}
!51 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !52, size: 64)
!52 = !DICompositeType(tag: DW_TAG_array_type, baseType: !53, elements: !57)
!53 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !54)
!54 = !DIDerivedType(tag: DW_TAG_typedef, name: "CeedScalar", file: !55, line: 26, baseType: !56)
!55 = !DIFile(filename: "include/ceed/ceed-f64.h", directory: "/home/wmoses/git/Enzyme/enzyme/sc/libCEED")
!56 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
!57 = !{!58, !59}
!58 = !DISubrange(count: 3)
!59 = !DISubrange(count: -1)
!60 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !61, size: 64)
!61 = !DICompositeType(tag: DW_TAG_array_type, baseType: !54, elements: !62)
!62 = !{!59}
!63 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !64, size: 64)
!64 = !DICompositeType(tag: DW_TAG_array_type, baseType: !53, elements: !62)
!65 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !66, size: 64)
!66 = !DICompositeType(tag: DW_TAG_array_type, baseType: !54, elements: !57)
!67 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !68, size: 64)
!68 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!69 = !DIDerivedType(tag: DW_TAG_typedef, name: "Physics", file: !70, line: 29, baseType: !71)
!70 = !DIFile(filename: "examples/solids/problems/../qfunctions/finite-strain-neo-hookean.h", directory: "/home/wmoses/git/Enzyme/enzyme/sc/libCEED")
!71 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !72, size: 64)
!72 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Physics_private", file: !70, line: 30, size: 128, elements: !73)
!73 = !{!74, !75}
!74 = !DIDerivedType(tag: DW_TAG_member, name: "nu", scope: !72, file: !70, line: 31, baseType: !54, size: 64)
!75 = !DIDerivedType(tag: DW_TAG_member, name: "E", scope: !72, file: !70, line: 32, baseType: !54, size: 64, offset: 64)
!76 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !56, size: 64)
!77 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !54, size: 64)
!78 = !{!79, !123, !126, !128, !130, !132, !134, !140, !145, !150, !155, !160, !165, !169}
!79 = !DIGlobalVariableExpression(var: !80, expr: !DIExpression())
!80 = distinct !DIGlobalVariable(name: "finite_strain_neo_Hookean_initial_ad", scope: !0, file: !1, line: 14, type: !81, isLocal: false, isDefinition: true)
!81 = !DIDerivedType(tag: DW_TAG_typedef, name: "ProblemData", file: !26, line: 156, baseType: !82)
!82 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !26, line: 147, size: 1152, elements: !83)
!83 = !{!84, !101, !102, !103, !104, !105, !106, !107, !108, !109, !110, !111, !112, !113, !114, !116, !117, !118, !120}
!84 = !DIDerivedType(tag: DW_TAG_member, name: "setup_geo", scope: !82, file: !26, line: 148, baseType: !85, size: 64)
!85 = !DIDerivedType(tag: DW_TAG_typedef, name: "CeedQFunctionUser", file: !4, line: 599, baseType: !86)
!86 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !87, size: 64)
!87 = !DISubroutineType(types: !88)
!88 = !{!89, !68, !90, !96, !99}
!89 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!90 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !91)
!91 = !DIDerivedType(tag: DW_TAG_typedef, name: "CeedInt", file: !4, line: 129, baseType: !92)
!92 = !DIDerivedType(tag: DW_TAG_typedef, name: "int32_t", file: !93, line: 26, baseType: !94)
!93 = !DIFile(filename: "/usr/include/x86_64-linux-gnu/bits/stdint-intn.h", directory: "")
!94 = !DIDerivedType(tag: DW_TAG_typedef, name: "__int32_t", file: !95, line: 40, baseType: !89)
!95 = !DIFile(filename: "/usr/include/x86_64-linux-gnu/bits/types.h", directory: "")
!96 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !97, size: 64)
!97 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !98)
!98 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !53, size: 64)
!99 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !100, size: 64)
!100 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !77)
!101 = !DIDerivedType(tag: DW_TAG_member, name: "residual", scope: !82, file: !26, line: 148, baseType: !85, size: 64, offset: 64)
!102 = !DIDerivedType(tag: DW_TAG_member, name: "jacobian", scope: !82, file: !26, line: 148, baseType: !85, size: 64, offset: 128)
!103 = !DIDerivedType(tag: DW_TAG_member, name: "energy", scope: !82, file: !26, line: 148, baseType: !85, size: 64, offset: 192)
!104 = !DIDerivedType(tag: DW_TAG_member, name: "diagnostic", scope: !82, file: !26, line: 149, baseType: !85, size: 64, offset: 256)
!105 = !DIDerivedType(tag: DW_TAG_member, name: "true_soln", scope: !82, file: !26, line: 149, baseType: !85, size: 64, offset: 320)
!106 = !DIDerivedType(tag: DW_TAG_member, name: "tape", scope: !82, file: !26, line: 149, baseType: !85, size: 64, offset: 384)
!107 = !DIDerivedType(tag: DW_TAG_member, name: "setup_geo_loc", scope: !82, file: !26, line: 150, baseType: !43, size: 64, offset: 448)
!108 = !DIDerivedType(tag: DW_TAG_member, name: "residual_loc", scope: !82, file: !26, line: 150, baseType: !43, size: 64, offset: 512)
!109 = !DIDerivedType(tag: DW_TAG_member, name: "jacobian_loc", scope: !82, file: !26, line: 150, baseType: !43, size: 64, offset: 576)
!110 = !DIDerivedType(tag: DW_TAG_member, name: "energy_loc", scope: !82, file: !26, line: 150, baseType: !43, size: 64, offset: 640)
!111 = !DIDerivedType(tag: DW_TAG_member, name: "diagnostic_loc", scope: !82, file: !26, line: 151, baseType: !43, size: 64, offset: 704)
!112 = !DIDerivedType(tag: DW_TAG_member, name: "true_soln_loc", scope: !82, file: !26, line: 151, baseType: !43, size: 64, offset: 768)
!113 = !DIDerivedType(tag: DW_TAG_member, name: "tape_loc", scope: !82, file: !26, line: 151, baseType: !43, size: 64, offset: 832)
!114 = !DIDerivedType(tag: DW_TAG_member, name: "quadrature_mode", scope: !82, file: !26, line: 152, baseType: !115, size: 32, offset: 896)
!115 = !DIDerivedType(tag: DW_TAG_typedef, name: "CeedQuadMode", file: !4, line: 494, baseType: !3)
!116 = !DIDerivedType(tag: DW_TAG_member, name: "q_data_size", scope: !82, file: !26, line: 153, baseType: !91, size: 32, offset: 928)
!117 = !DIDerivedType(tag: DW_TAG_member, name: "number_fields_stored", scope: !82, file: !26, line: 153, baseType: !91, size: 32, offset: 960)
!118 = !DIDerivedType(tag: DW_TAG_member, name: "field_sizes", scope: !82, file: !26, line: 154, baseType: !119, size: 64, offset: 1024)
!119 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !91, size: 64)
!120 = !DIDerivedType(tag: DW_TAG_member, name: "field_names", scope: !82, file: !26, line: 155, baseType: !121, size: 64, offset: 1088)
!121 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !122, size: 64)
!122 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !43)
!123 = !DIGlobalVariableExpression(var: !124, expr: !DIExpression())
!124 = distinct !DIGlobalVariable(name: "enzyme_dup", scope: !0, file: !125, line: 177, type: !89, isLocal: false, isDefinition: true)
!125 = !DIFile(filename: "examples/solids/problems/../qfunctions/finite-strain-neo-hookean-initial-ad.h", directory: "/home/wmoses/git/Enzyme/enzyme/sc/libCEED")
!126 = !DIGlobalVariableExpression(var: !127, expr: !DIExpression())
!127 = distinct !DIGlobalVariable(name: "enzyme_tape", scope: !0, file: !125, line: 177, type: !89, isLocal: false, isDefinition: true)
!128 = !DIGlobalVariableExpression(var: !129, expr: !DIExpression())
!129 = distinct !DIGlobalVariable(name: "enzyme_const", scope: !0, file: !125, line: 177, type: !89, isLocal: false, isDefinition: true)
!130 = !DIGlobalVariableExpression(var: !131, expr: !DIExpression())
!131 = distinct !DIGlobalVariable(name: "enzyme_nofree", scope: !0, file: !125, line: 177, type: !89, isLocal: false, isDefinition: true)
!132 = !DIGlobalVariableExpression(var: !133, expr: !DIExpression())
!133 = distinct !DIGlobalVariable(name: "enzyme_allocated", scope: !0, file: !125, line: 177, type: !89, isLocal: false, isDefinition: true)
!134 = !DIGlobalVariableExpression(var: !135, expr: !DIExpression())
!135 = distinct !DIGlobalVariable(name: "SetupGeo_loc", scope: !0, file: !136, line: 47, type: !137, isLocal: true, isDefinition: true)
!136 = !DIFile(filename: "examples/solids/problems/../qfunctions/common.h", directory: "/home/wmoses/git/Enzyme/enzyme/sc/libCEED")
!137 = !DICompositeType(tag: DW_TAG_array_type, baseType: !44, size: 456, elements: !138)
!138 = !{!139}
!139 = !DISubrange(count: 57)
!140 = !DIGlobalVariableExpression(var: !141, expr: !DIExpression())
!141 = distinct !DIGlobalVariable(name: "ElasFSInitialNHF_AD_loc", scope: !0, file: !125, line: 216, type: !142, isLocal: true, isDefinition: true)
!142 = !DICompositeType(tag: DW_TAG_array_type, baseType: !44, size: 784, elements: !143)
!143 = !{!144}
!144 = !DISubrange(count: 98)
!145 = !DIGlobalVariableExpression(var: !146, expr: !DIExpression())
!146 = distinct !DIGlobalVariable(name: "ElasFSInitialNHdF_AD_loc", scope: !0, file: !125, line: 379, type: !147, isLocal: true, isDefinition: true)
!147 = !DICompositeType(tag: DW_TAG_array_type, baseType: !44, size: 792, elements: !148)
!148 = !{!149}
!149 = !DISubrange(count: 99)
!150 = !DIGlobalVariableExpression(var: !151, expr: !DIExpression())
!151 = distinct !DIGlobalVariable(name: "ElasFSNHEnergy_loc", scope: !0, file: !70, line: 91, type: !152, isLocal: true, isDefinition: true)
!152 = !DICompositeType(tag: DW_TAG_array_type, baseType: !44, size: 656, elements: !153)
!153 = !{!154}
!154 = !DISubrange(count: 82)
!155 = !DIGlobalVariableExpression(var: !156, expr: !DIExpression())
!156 = distinct !DIGlobalVariable(name: "ElasFSNHDiagnostic_loc", scope: !0, file: !70, line: 181, type: !157, isLocal: true, isDefinition: true)
!157 = !DICompositeType(tag: DW_TAG_array_type, baseType: !44, size: 688, elements: !158)
!158 = !{!159}
!159 = !DISubrange(count: 86)
!160 = !DIGlobalVariableExpression(var: !161, expr: !DIExpression())
!161 = distinct !DIGlobalVariable(name: "ElasFSInitialNHFree_AD_loc", scope: !0, file: !125, line: 515, type: !162, isLocal: true, isDefinition: true)
!162 = !DICompositeType(tag: DW_TAG_array_type, baseType: !44, size: 808, elements: !163)
!163 = !{!164}
!164 = !DISubrange(count: 101)
!165 = !DIGlobalVariableExpression(var: !166, expr: !DIExpression())
!166 = distinct !DIGlobalVariable(name: "field_sizes", scope: !0, file: !1, line: 12, type: !167, isLocal: true, isDefinition: true)
!167 = !DICompositeType(tag: DW_TAG_array_type, baseType: !91, size: 96, elements: !168)
!168 = !{!58}
!169 = !DIGlobalVariableExpression(var: !170, expr: !DIExpression())
!170 = distinct !DIGlobalVariable(name: "field_names", scope: !0, file: !1, line: 11, type: !171, isLocal: true, isDefinition: true)
!171 = !DICompositeType(tag: DW_TAG_array_type, baseType: !122, size: 192, elements: !168)
!172 = !{i32 7, !"Dwarf Version", i32 4}
!173 = !{i32 2, !"Debug Info Version", i32 3}
!174 = !{i32 1, !"wchar_size", i32 4}
!175 = !{i32 7, !"PIC Level", i32 2}
!176 = !{i32 7, !"uwtable", i32 1}
!177 = !{!"clang version 13.0.1 (git@github.com:llvm/llvm-project cf15ccdeb6d5254ee7d46c7535c29200003a3880)"}
!178 = !DILocalVariable(name: "x", arg: 1, scope: !179, file: !70, line: 47, type: !54)
!179 = distinct !DISubprogram(name: "log1p_series_shifted", scope: !70, file: !70, line: 47, type: !180, scopeLine: 47, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !182)
!180 = !DISubroutineType(types: !181)
!181 = !{!54, !54}
!182 = !{!178, !183, !184, !185, !186, !187}
!183 = !DILocalVariable(name: "left", scope: !179, file: !70, line: 48, type: !53)
!184 = !DILocalVariable(name: "right", scope: !179, file: !70, line: 48, type: !53)
!185 = !DILocalVariable(name: "sum", scope: !179, file: !70, line: 49, type: !54)
!186 = !DILocalVariable(name: "y", scope: !179, file: !70, line: 59, type: !54)
!187 = !DILocalVariable(name: "y2", scope: !179, file: !70, line: 60, type: !53)
!188 = !DILocation(line: 0, scope: !179, inlinedAt: !189)
!189 = distinct !DILocation(line: 159, column: 27, scope: !190)
!190 = distinct !DISubprogram(name: "computeS", scope: !125, file: !125, line: 127, type: !191, scopeLine: 128, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !193)
!191 = !DISubroutineType(types: !192)
!192 = !{!89, !77, !77, !53, !53}
!193 = !{!194, !195, !196, !197, !198, !201, !203, !207, !208, !209, !211, !212, !213, !215}
!194 = !DILocalVariable(name: "Swork", arg: 1, scope: !190, file: !125, line: 127, type: !77)
!195 = !DILocalVariable(name: "E2work", arg: 2, scope: !190, file: !125, line: 127, type: !77)
!196 = !DILocalVariable(name: "lambda", arg: 3, scope: !190, file: !125, line: 128, type: !53)
!197 = !DILocalVariable(name: "mu", arg: 4, scope: !190, file: !125, line: 128, type: !53)
!198 = !DILocalVariable(name: "E2", scope: !190, file: !125, line: 130, type: !199)
!199 = !DICompositeType(tag: DW_TAG_array_type, baseType: !54, size: 576, elements: !200)
!200 = !{!58, !58}
!201 = !DILocalVariable(name: "C", scope: !190, file: !125, line: 139, type: !202)
!202 = !DICompositeType(tag: DW_TAG_array_type, baseType: !53, size: 576, elements: !200)
!203 = !DILocalVariable(name: "Cinvwork", scope: !190, file: !125, line: 146, type: !204)
!204 = !DICompositeType(tag: DW_TAG_array_type, baseType: !54, size: 384, elements: !205)
!205 = !{!206}
!206 = !DISubrange(count: 6)
!207 = !DILocalVariable(name: "detCm1", scope: !190, file: !125, line: 147, type: !53)
!208 = !DILocalVariable(name: "C_inv", scope: !190, file: !125, line: 151, type: !202)
!209 = !DILocalVariable(name: "indj", scope: !190, file: !125, line: 158, type: !210)
!210 = !DICompositeType(tag: DW_TAG_array_type, baseType: !90, size: 192, elements: !205)
!211 = !DILocalVariable(name: "indk", scope: !190, file: !125, line: 158, type: !210)
!212 = !DILocalVariable(name: "logJ", scope: !190, file: !125, line: 159, type: !53)
!213 = !DILocalVariable(name: "m", scope: !214, file: !125, line: 161, type: !91)
!214 = distinct !DILexicalBlock(scope: !190, file: !125, line: 161, column: 3)
!215 = !DILocalVariable(name: "n", scope: !216, file: !125, line: 163, type: !91)
!216 = distinct !DILexicalBlock(scope: !217, file: !125, line: 163, column: 5)
!217 = distinct !DILexicalBlock(scope: !218, file: !125, line: 161, column: 35)
!218 = distinct !DILexicalBlock(scope: !214, file: !125, line: 161, column: 3)

; CHECK: define internal double @fwddiffecomputeS(double %E2work, double %"E2work'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %cmp.i = fcmp olt double %E2work, 0xBFD2BEC333018866
; CHECK-NEXT:   br i1 %cmp.i, label %if.then.i, label %if.else.i

; CHECK: if.then.i:                                        ; preds = %entry
; CHECK-NEXT:   %0 = fmul fast double %"E2work'", 2.000000e+00
; CHECK-NEXT:   br label %log1p_series_shifted.exit

; CHECK: if.else.i:                                        ; preds = %entry
; CHECK-NEXT:   br label %log1p_series_shifted.exit

; CHECK: log1p_series_shifted.exit:                        ; preds = %if.else.i, %if.then.i
; CHECK-NEXT:   %[[res:.+]] = phi {{(fast )?}}double [ %0, %if.then.i ], [ %"E2work'", %if.else.i ]
; CHECK-NEXT:   ret double %[[res]]
; CHECK-NEXT: }
