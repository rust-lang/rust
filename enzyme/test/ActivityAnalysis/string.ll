; RUN: %opt < %s %loadEnzyme -print-activity-analysis -activity-analysis-func=_Z2fnv -o /dev/null | FileCheck %s

%"class.std::__cxx11::basic_string" = type { %"struct.std::__cxx11::basic_string<char>::_Alloc_hider", i64, %union.anon }
%"struct.std::__cxx11::basic_string<char>::_Alloc_hider" = type { i8* }
%union.anon = type { i64, [8 x i8] }
%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }

@.str = private unnamed_addr constant [12 x i8] c"test string\00", align 1
@.str.2 = private unnamed_addr constant [4 x i8] c"%f\0A\00", align 1
@str = private unnamed_addr constant [5 x i8] c"Home\00", align 1

define dso_local void @_Z2fnv() {
entry:
  %s = alloca %"class.std::__cxx11::basic_string", align 8
  %0 = bitcast %"class.std::__cxx11::basic_string"* %s to i8*
  %a1 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %s, i64 0, i32 2
  %a2 = bitcast %"class.std::__cxx11::basic_string"* %s to %union.anon**
  store %union.anon* %a1, %union.anon** %a2, align 8, !tbaa !2
  %a3 = bitcast %union.anon* %a1 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(11) %a3, i8* nonnull align 1 dereferenceable(11) getelementptr inbounds ([12 x i8], [12 x i8]* @.str, i64 0, i64 0), i64 11, i1 false) #7
  %_M_p.i.i.i.i.i = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %s, i64 0, i32 0, i32 0
  %_M_string_length.i.i.i.i.i.i = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %s, i64 0, i32 1
  store i64 11, i64* %_M_string_length.i.i.i.i.i.i, align 8, !tbaa !7
  %a4 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %s, i64 0, i32 2, i32 1, i64 3
  store i8 0, i8* %a4, align 1, !tbaa !10
  %call1 = call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) %a3)
  %a5 = load i8*, i8** %_M_p.i.i.i.i.i, align 8, !tbaa !11
  %cmp.i.i.i = icmp eq i8* %a5, %a3
  br i1 %cmp.i.i.i, label %_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED2Ev.exit, label %if.then.i.i

if.then.i.i:                                      ; preds = %entry
  call void @_ZdlPv(i8* %a5) #7
  br label %_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED2Ev.exit

_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED2Ev.exit: ; preds = %entry, %if.then.i.i
  ret void
}

; Function Attrs: nofree nounwind
declare dso_local i32 @printf(i8* nocapture readonly, ...) 

; Function Attrs: nobuiltin nounwind
declare dso_local void @_ZdlPv(i8*) local_unnamed_addr #5

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1)

attributes #0 = { nounwind uwtable "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { nobuiltin nounwind }
attributes #7 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 12.0.1 (git@github.com:llvm/llvm-project 4973ce53ca8abfc14233a3d8b3045673e0e8543c)"}
!2 = !{!3, !4, i64 0}
!3 = !{!"_ZTSNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_Alloc_hiderE", !4, i64 0}
!4 = !{!"any pointer", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}
!7 = !{!8, !9, i64 8}
!8 = !{!"_ZTSNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE", !3, i64 0, !9, i64 8, !5, i64 16}
!9 = !{!"long", !5, i64 0}
!10 = !{!5, !5, i64 0}
!11 = !{!8, !4, i64 0}

; CHECK: entry
; CHECK-NEXT:   %s = alloca %"class.std::__cxx11::basic_string", align 8: icv:1 ici:1
; CHECK-NEXT:   %0 = bitcast %"class.std::__cxx11::basic_string"* %s to i8*: icv:1 ici:1
; CHECK-NEXT:   %a1 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %s, i64 0, i32 2: icv:1 ici:1
; CHECK-NEXT:   %a2 = bitcast %"class.std::__cxx11::basic_string"* %s to %union.anon**: icv:1 ici:1
; CHECK-NEXT:   store %union.anon* %a1, %union.anon** %a2, align 8, !tbaa !2: icv:1 ici:1
; CHECK-NEXT:   %a3 = bitcast %union.anon* %a1 to i8*: icv:1 ici:1
; CHECK-NEXT:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(11) %a3, i8* nonnull align 1 dereferenceable(11) getelementptr inbounds ([12 x i8], [12 x i8]* @.str, i64 0, i64 0), i64 11, i1 false) #2: icv:1 ici:1
; CHECK-NEXT:   %_M_p.i.i.i.i.i = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %s, i64 0, i32 0, i32 0: icv:1 ici:1
; CHECK-NEXT:   %_M_string_length.i.i.i.i.i.i = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %s, i64 0, i32 1: icv:1 ici:1
; CHECK-NEXT:   store i64 11, i64* %_M_string_length.i.i.i.i.i.i, align 8, !tbaa !7: icv:1 ici:1
; CHECK-NEXT:   %a4 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %s, i64 0, i32 2, i32 1, i64 3: icv:1 ici:1
; CHECK-NEXT:   store i8 0, i8* %a4, align 1, !tbaa !10: icv:1 ici:1
; CHECK-NEXT:   %call1 = call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) %a3): icv:1 ici:1
; CHECK-NEXT:   %a5 = load i8*, i8** %_M_p.i.i.i.i.i, align 8, !tbaa !11: icv:1 ici:1
; CHECK-NEXT:   %cmp.i.i.i = icmp eq i8* %a5, %a3: icv:1 ici:1
; CHECK-NEXT:   br i1 %cmp.i.i.i, label %_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED2Ev.exit, label %if.then.i.i: icv:1 ici:1
; CHECK-NEXT: if.then.i.i
; CHECK-NEXT:   call void @_ZdlPv(i8* %a5) #2: icv:1 ici:1
; CHECK-NEXT:   br label %_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED2Ev.exit: icv:1 ici:1
; CHECK-NEXT: _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED2Ev.exit
; CHECK-NEXT:   ret void: icv:1 ici:1
