; RUN: %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=caller -o /dev/null | FileCheck %s

define void @caller(double* %p) {
entry:
  %q = alloca [2 x double], align 8
  %gep0 = getelementptr [2 x double], [2 x double]* %q, i32 0, i32 0
  %gep = getelementptr [2 x double], [2 x double]* %q, i32 0, i32 1
  store double 0.000000e+00, double* %gep, align 8, !tbaa !2
  %o = call double* @max(double* %p, double* %gep0, i1 true)
  ret void
}

define double* @max(double* %a, double* %b, i1 %cmp) {
entry:
  %retval = alloca double*, align 8
  %av = load double, double* %a, align 8
  %bv = load double, double* %b, align 8
  br i1 %cmp, label %if.then, label %if.end

if.then:
  store double* %a, double** %retval, align 8
  br label %return

if.end:
  store double* %b, double** %retval, align 8
  br label %return

return:
  %res = load double*, double** %retval, align 8
  ret double* %res
}

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0, i64 8}
!3 = !{!4, i64 8, !"long"}
!4 = !{!5, i64 1, !"omnipotent char"}
!5 = !{!"Simple C++ TBAA"}


; CHECK: caller - {} |{[-1]:Pointer, [-1,-1]:Float@double}:{}
; CHECK-NEXT: double* %p: {[-1]:Pointer, [-1,-1]:Float@double}
; CHECK-NEXT: entry
; CHECK-NEXT:   %q = alloca [2 x double], align 8: {[-1]:Pointer, [-1,0]:Float@double, [-1,8]:Integer, [-1,9]:Integer, [-1,10]:Integer, [-1,11]:Integer, [-1,12]:Integer, [-1,13]:Integer, [-1,14]:Integer, [-1,15]:Integer}
; CHECK-NEXT:   %gep0 = getelementptr [2 x double], [2 x double]* %q, i32 0, i32 0: {[-1]:Pointer, [-1,0]:Float@double}
; CHECK-NEXT:   %gep = getelementptr [2 x double], [2 x double]* %q, i32 0, i32 1: {[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer, [-1,7]:Integer}
; CHECK-NEXT:   store double 0.000000e+00, double* %gep, align 8, !tbaa !2: {}
; CHECK-NEXT:   %o = call double* @max(double* %p, double* %gep0, i1 true): {[-1]:Pointer, [-1,0]:Float@double}
; CHECK-NEXT:   ret void: {}
