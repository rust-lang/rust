; RUN: %opt < %s %loadEnzyme -print-activity-analysis -activity-analysis-func=matvec -activity-analysis-inactive-args -o /dev/null | FileCheck %s


%"class.std::__cxx11::basic_string" = type { %"struct.std::__cxx11::basic_string<char>::_Alloc_hider", i64, %union.anon }
%"struct.std::__cxx11::basic_string<char>::_Alloc_hider" = type { i8* }
%union.anon = type { i64, [8 x i8] }

@.str = private unnamed_addr constant [3 x i8] c"g1\00", align 1

declare double* @makedouble()

define void @matvec(%"class.std::__cxx11::basic_string"* %a1) {
entry:
  %a0 = call double* @makedouble()
  %a3 = tail call i32 @_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7compareEPKc(%"class.std::__cxx11::basic_string"* nonnull %a1, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i64 0, i64 0))
  %a4 = icmp eq i32 %a3, 0
  br i1 %a4, label %add, label %exit

add:                                                ; preds = %2
  %a6 = load double, double* %a0, align 8
  %a7 = fadd double %a6, 1.000000e+00
  store double %a7, double* %a0, align 8
  br label %exit

exit:                                                ; preds = %5, %2
  ret void
}

declare dso_local i32 @_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7compareEPKc(%"class.std::__cxx11::basic_string"* nonnull, i8*) 

; CHECK: %"class.std::__cxx11::basic_string"* %a1: icv:1
; CHECK-NEXT: entry
; CHECK-NEXT:   %a0 = call double* @makedouble(): icv:0 ici:1
; CHECK-NEXT:   %a3 = tail call i32 @_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7compareEPKc(%"class.std::__cxx11::basic_string"* nonnull %a1, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i64 0, i64 0)): icv:1 ici:1
; CHECK-NEXT:   %a4 = icmp eq i32 %a3, 0: icv:1 ici:1
; CHECK-NEXT:   br i1 %a4, label %add, label %exit: icv:1 ici:1
; CHECK-NEXT: add
; CHECK-NEXT:   %a6 = load double, double* %a0, align 8: icv:0 ici:0
; CHECK-NEXT:   %a7 = fadd double %a6, 1.000000e+00: icv:0 ici:0
; CHECK-NEXT:   store double %a7, double* %a0, align 8: icv:1 ici:0
; CHECK-NEXT:   br label %exit: icv:1 ici:1
; CHECK-NEXT: exit
; CHECK-NEXT:   ret void: icv:1 ici:1
