; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

%struct.Gradients = type { double, double, double }

@global = external dso_local local_unnamed_addr global double, align 8, !enzyme_shadow !{[3 x double]* @dglobal }
@dglobal = external dso_local local_unnamed_addr global [3 x double], align 8

declare dso_local %struct.Gradients @_Z22__enzyme_fwddiffPFddEz(double (double)*, ...)

define dso_local double @_Z9mulglobald(double %x) {
entry:
  %0 = load double, double* @global, align 8
  %mul = fmul double %0, %x
  ret double %mul
}

define dso_local void @_Z10derivatived(double %x) {
entry:
  call %struct.Gradients (double (double)*, ...) @_Z22__enzyme_fwddiffPFddEz(double (double)* nonnull @_Z9mulglobald, metadata !"enzyme_width", i64 3, double %x, double 1.0 , double 2.0, double 3.0)
  ret void
}


; CHECK: define internal [3 x double] @fwddiffe3_Z9mulglobald(double %x, [3 x double] %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:  %0 = load double, double* @global, align 8
; CHECK-NEXT:  %1 = load double, double* getelementptr inbounds ([3 x double], [3 x double]* @dglobal, i32 0, i32 0), align 8
; CHECK-NEXT:  %2 = load double, double* getelementptr inbounds ([3 x double], [3 x double]* @dglobal, i32 0, i32 1), align 8
; CHECK-NEXT:  %3 = load double, double* getelementptr inbounds ([3 x double], [3 x double]* @dglobal, i32 0, i32 2), align 8
; CHECK-NEXT:  %4 = extractvalue [3 x double] %"x'", 0
; CHECK-NEXT:  %5 = fmul fast double %1, %x
; CHECK-NEXT:  %6 = fmul fast double %4, %0
; CHECK-NEXT:  %7 = fadd fast double %5, %6
; CHECK-NEXT:  %8 = insertvalue [3 x double] undef, double %7, 0
; CHECK-NEXT:  %9 = extractvalue [3 x double] %"x'", 1
; CHECK-NEXT:  %10 = fmul fast double %2, %x
; CHECK-NEXT:  %11 = fmul fast double %9, %0
; CHECK-NEXT:  %12 = fadd fast double %10, %11
; CHECK-NEXT:  %13 = insertvalue [3 x double] %8, double %12, 1
; CHECK-NEXT:  %14 = extractvalue [3 x double] %"x'", 2
; CHECK-NEXT:  %15 = fmul fast double %3, %x
; CHECK-NEXT:  %16 = fmul fast double %14, %0
; CHECK-NEXT:  %17 = fadd fast double %15, %16
; CHECK-NEXT:  %18 = insertvalue [3 x double] %13, double %17, 2
; CHECK-NEXT:  ret [3 x double] %18
; CHECK-NEXT:}