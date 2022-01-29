; RUN: if [ %llvmver -ge 9 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi

; ModuleID = '/workspaces/Enzyme/enzyme/test/Integration/ReverseMode/dbginfo2.c'
source_filename = "/workspaces/Enzyme/enzyme/test/Integration/ReverseMode/dbginfo2.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, %struct._IO_codecvt*, %struct._IO_wide_data*, %struct._IO_FILE*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type opaque
%struct._IO_codecvt = type opaque
%struct._IO_wide_data = type opaque
%struct.Data = type { double, double, double }

@stderr = external dso_local local_unnamed_addr global %struct._IO_FILE*, align 8
@.str = private unnamed_addr constant [68 x i8] c"Assertion Failed: fabs( [%s = %g] - [%s = %g] ) > %g at %s:%d (%s)\0A\00", align 1
@.str.1 = private unnamed_addr constant [4 x i8] c"res\00", align 1
@.str.2 = private unnamed_addr constant [4 x i8] c"exp\00", align 1
@.str.3 = private unnamed_addr constant [66 x i8] c"/workspaces/Enzyme/enzyme/test/Integration/ReverseMode/dbginfo2.c\00", align 1
@__PRETTY_FUNCTION__.main = private unnamed_addr constant [11 x i8] c"int main()\00", align 1

; Function Attrs: nofree norecurse nounwind uwtable writeonly
define dso_local void @foo(double %x, %struct.Data* nocapture %data) local_unnamed_addr #0 !dbg !13 {
entry:
  call void @llvm.dbg.value(metadata double %x, metadata !25, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.value(metadata %struct.Data* %data, metadata !26, metadata !DIExpression()), !dbg !27
  %mul = fmul double %x, 2.000000e+00, !dbg !28
  %x1 = getelementptr inbounds %struct.Data, %struct.Data* %data, i64 0, i32 0, !dbg !29
  store double %mul, double* %x1, align 8, !dbg !30, !tbaa !31
  %add = fadd double %x, 3.000000e+00, !dbg !36
  %x2 = getelementptr inbounds %struct.Data, %struct.Data* %data, i64 0, i32 1, !dbg !37
  store double %add, double* %x2, align 8, !dbg !38, !tbaa !39
  %mul3 = fmul double %mul, %add, !dbg !40
  %res = getelementptr inbounds %struct.Data, %struct.Data* %data, i64 0, i32 2, !dbg !41
  store double %mul3, double* %res, align 8, !dbg !42, !tbaa !43
  ret void, !dbg !44
}

; Function Attrs: nounwind uwtable
define dso_local double @call(double %x) #1 !dbg !45 {
entry:
  call void @llvm.dbg.value(metadata double %x, metadata !49, metadata !DIExpression()), !dbg !52
  %call = call noalias dereferenceable_or_null(24) i8* @malloc(i64 24) #7, !dbg !53
  %0 = bitcast i8* %call to %struct.Data*, !dbg !53
  call void @llvm.dbg.value(metadata %struct.Data* %0, metadata !50, metadata !DIExpression()), !dbg !52
  call void @foo(double %x, %struct.Data* %0), !dbg !54
  %res1 = getelementptr inbounds i8, i8* %call, i64 16, !dbg !55
  %1 = bitcast i8* %res1 to double*, !dbg !55
  %2 = load double, double* %1, align 8, !dbg !55, !tbaa !43
  call void @llvm.dbg.value(metadata double %2, metadata !51, metadata !DIExpression()), !dbg !52
  call void @free(i8* %call) #7, !dbg !56
  ret double %2, !dbg !57
}

; Function Attrs: nofree nounwind
declare dso_local noalias i8* @malloc(i64) local_unnamed_addr #2

; Function Attrs: nounwind
declare !dbg !4 dso_local void @free(i8* nocapture) local_unnamed_addr #3

; Function Attrs: nounwind uwtable
define dso_local double @dcall(double %x) local_unnamed_addr #1 !dbg !58 {
entry:
  call void @llvm.dbg.value(metadata double %x, metadata !60, metadata !DIExpression()), !dbg !61
  %call = call double @__enzyme_autodiff(i8* bitcast (double (double)* @call to i8*), double %x) #7, !dbg !62
  ret double %call, !dbg !63
}

declare dso_local double @__enzyme_autodiff(i8*, double) local_unnamed_addr #4

; Function Attrs: nounwind uwtable
define dso_local i32 @main() local_unnamed_addr #1 !dbg !64 {
entry:
  %call = call double @dcall(double 2.450000e+01), !dbg !71
  call void @llvm.dbg.value(metadata double %call, metadata !69, metadata !DIExpression()), !dbg !72
  call void @llvm.dbg.value(metadata double -1.040000e+02, metadata !70, metadata !DIExpression()), !dbg !72
  %sub = fadd double %call, 1.040000e+02, !dbg !73
  %0 = call double @llvm.fabs.f64(double %sub), !dbg !73
  %cmp = fcmp ogt double %0, 1.000000e-10, !dbg !73
  br i1 %cmp, label %if.then, label %if.end, !dbg !76

if.then:                                          ; preds = %entry
  %1 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !dbg !77, !tbaa !79
  %call1 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %1, i8* getelementptr inbounds ([68 x i8], [68 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.1, i64 0, i64 0), double %call, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.2, i64 0, i64 0), double -1.040000e+02, double 1.000000e-10, i8* getelementptr inbounds ([66 x i8], [66 x i8]* @.str.3, i64 0, i64 0), i32 38, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @__PRETTY_FUNCTION__.main, i64 0, i64 0)) #8, !dbg !77
  call void @abort() #9, !dbg !77
  unreachable, !dbg !77

if.end:                                           ; preds = %entry
  ret i32 0, !dbg !81
}

; Function Attrs: nounwind readnone speculatable willreturn
declare double @llvm.fabs.f64(double) #5

; Function Attrs: nofree nounwind
declare dso_local i32 @fprintf(%struct._IO_FILE* nocapture, i8* nocapture readonly, ...) local_unnamed_addr #2

; Function Attrs: noreturn nounwind
declare dso_local void @abort() local_unnamed_addr #6

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #5

attributes #0 = { nofree norecurse nounwind uwtable writeonly "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nofree nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { nounwind readnone speculatable willreturn }
attributes #6 = { noreturn nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #7 = { nounwind }
attributes #8 = { cold }
attributes #9 = { noreturn nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9, !10, !11}
!llvm.ident = !{!12}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "Ubuntu clang version 10.0.1-++20210313014558+ef32c611aa21-1~exp1~20210313125203.63", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "/workspaces/Enzyme/enzyme/test/Integration/ReverseMode/dbginfo2.c", directory: "/workspaces/Enzyme/enzyme/build")
!2 = !{}
!3 = !{!4, !8}
!4 = !DISubprogram(name: "free", scope: !5, file: !5, line: 565, type: !6, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !2)
!5 = !DIFile(filename: "/usr/include/stdlib.h", directory: "")
!6 = !DISubroutineType(types: !7)
!7 = !{null, !8}
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!9 = !{i32 7, !"Dwarf Version", i32 4}
!10 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !{i32 1, !"wchar_size", i32 4}
!12 = !{!"Ubuntu clang version 10.0.1-++20210313014558+ef32c611aa21-1~exp1~20210313125203.63"}
!13 = distinct !DISubprogram(name: "foo", scope: !14, file: !14, line: 16, type: !15, scopeLine: 16, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !24)
!14 = !DIFile(filename: "test/Integration/ReverseMode/dbginfo2.c", directory: "/workspaces/Enzyme/enzyme")
!15 = !DISubroutineType(types: !16)
!16 = !{null, !17, !18}
!17 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
!18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !19, size: 64)
!19 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Data", file: !14, line: 10, size: 192, elements: !20)
!20 = !{!21, !22, !23}
!21 = !DIDerivedType(tag: DW_TAG_member, name: "x1", scope: !19, file: !14, line: 11, baseType: !17, size: 64)
!22 = !DIDerivedType(tag: DW_TAG_member, name: "x2", scope: !19, file: !14, line: 12, baseType: !17, size: 64, offset: 64)
!23 = !DIDerivedType(tag: DW_TAG_member, name: "res", scope: !19, file: !14, line: 13, baseType: !17, size: 64, offset: 128)
!24 = !{!25, !26}
!25 = !DILocalVariable(name: "x", arg: 1, scope: !13, file: !14, line: 16, type: !17)
!26 = !DILocalVariable(name: "data", arg: 2, scope: !13, file: !14, line: 16, type: !18)
!27 = !DILocation(line: 0, scope: !13)
!28 = !DILocation(line: 17, column: 18, scope: !13)
!29 = !DILocation(line: 17, column: 11, scope: !13)
!30 = !DILocation(line: 17, column: 14, scope: !13)
!31 = !{!32, !33, i64 0}
!32 = !{!"Data", !33, i64 0, !33, i64 8, !33, i64 16}
!33 = !{!"double", !34, i64 0}
!34 = !{!"omnipotent char", !35, i64 0}
!35 = !{!"Simple C/C++ TBAA"}
!36 = !DILocation(line: 18, column: 18, scope: !13)
!37 = !DILocation(line: 18, column: 11, scope: !13)
!38 = !DILocation(line: 18, column: 14, scope: !13)
!39 = !{!32, !33, i64 8}
!40 = !DILocation(line: 19, column: 26, scope: !13)
!41 = !DILocation(line: 19, column: 11, scope: !13)
!42 = !DILocation(line: 19, column: 15, scope: !13)
!43 = !{!32, !33, i64 16}
!44 = !DILocation(line: 20, column: 5, scope: !13)
!45 = distinct !DISubprogram(name: "call", scope: !14, file: !14, line: 23, type: !46, scopeLine: 23, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !48)
!46 = !DISubroutineType(types: !47)
!47 = !{!17, !17}
!48 = !{!49, !50, !51}
!49 = !DILocalVariable(name: "x", arg: 1, scope: !45, file: !14, line: 23, type: !17)
!50 = !DILocalVariable(name: "data", scope: !45, file: !14, line: 24, type: !18)
!51 = !DILocalVariable(name: "res", scope: !45, file: !14, line: 26, type: !17)
!52 = !DILocation(line: 0, scope: !45)
!53 = !DILocation(line: 24, column: 25, scope: !45)
!54 = !DILocation(line: 25, column: 5, scope: !45)
!55 = !DILocation(line: 26, column: 24, scope: !45)
!56 = !DILocation(line: 27, column: 5, scope: !45)
!57 = !DILocation(line: 28, column: 5, scope: !45)
!58 = distinct !DISubprogram(name: "dcall", scope: !14, file: !14, line: 31, type: !46, scopeLine: 31, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !59)
!59 = !{!60}
!60 = !DILocalVariable(name: "x", arg: 1, scope: !58, file: !14, line: 31, type: !17)
!61 = !DILocation(line: 0, scope: !58)
!62 = !DILocation(line: 33, column: 12, scope: !58)
!63 = !DILocation(line: 33, column: 5, scope: !58)
!64 = distinct !DISubprogram(name: "main", scope: !14, file: !14, line: 35, type: !65, scopeLine: 35, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !68)
!65 = !DISubroutineType(types: !66)
!66 = !{!67}
!67 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!68 = !{!69, !70}
!69 = !DILocalVariable(name: "res", scope: !64, file: !14, line: 36, type: !17)
!70 = !DILocalVariable(name: "exp", scope: !64, file: !14, line: 37, type: !17)
!71 = !DILocation(line: 36, column: 18, scope: !64)
!72 = !DILocation(line: 0, scope: !64)
!73 = !DILocation(line: 38, column: 5, scope: !74)
!74 = distinct !DILexicalBlock(scope: !75, file: !14, line: 38, column: 5)
!75 = distinct !DILexicalBlock(scope: !64, file: !14, line: 38, column: 5)
!76 = !DILocation(line: 38, column: 5, scope: !75)
!77 = !DILocation(line: 38, column: 5, scope: !78)
!78 = distinct !DILexicalBlock(scope: !74, file: !14, line: 38, column: 5)
!79 = !{!80, !80, i64 0}
!80 = !{!"any pointer", !34, i64 0}
!81 = !DILocation(line: 39, column: 1, scope: !64)


; CHECK: define internal {{(dso_local )?}}{ double } @diffecall(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %call = call noalias nonnull dereferenceable(24) dereferenceable_or_null(24) i8* @malloc(i64 24) #{{.*}}, !dbg ![[DBG:[0-9]+]]
; CHECK-NEXT:   %"call'mi" = call noalias nonnull dereferenceable(24) dereferenceable_or_null(24) i8* @malloc(i64 24) #{{.*}}, !dbg !{{.*}}[[DBG]]
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull dereferenceable(24) dereferenceable_or_null(24) %"call'mi", i8 0, i64 24, i1 false)
; CHECK-NEXT:   %"'ipc2" = bitcast i8* %"call'mi" to %struct.Data*
; CHECK-NEXT:   %0 = bitcast i8* %call to %struct.Data*
; CHECK-NEXT:   %"res1'ipg" = getelementptr inbounds i8, i8* %"call'mi", i64 16
; CHECK-NEXT:   %"'ipc" = bitcast i8* %"res1'ipg" to double*
; CHECK-NEXT:   %1 = load double, double* %"'ipc", align 8
; CHECK-NEXT:   %2 = fadd fast double %1, %differeturn
; CHECK-NEXT:   store double %2, double* %"'ipc", align 8
; CHECK-NEXT:   %3 = call { double } @diffefoo(double %x, %struct.Data* %0, %struct.Data* %"'ipc2")
; CHECK-NEXT:   tail call void @free(i8* nonnull %"call'mi"), !dbg !{{.*}}[[DBG]]
; CHECK-NEXT:   tail call void @free(i8* nonnull %call), !dbg !{{.*}}[[DBG]]
; CHECK-NEXT:   ret { double } %3
; CHECK-NEXT: }
