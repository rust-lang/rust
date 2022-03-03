; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-lower-globals -mem2reg -sroa -simplifycfg -instsimplify -S | FileCheck %s

%struct.Gradients = type { double, double, double }

; Function Attrs: nounwind
declare %struct.Gradients @__enzyme_fwddiff(double (double)*, ...)

@global = external dso_local local_unnamed_addr global double, align 8

; Function Attrs: noinline norecurse nounwind readonly uwtable
define double @mulglobal(double %x) {
entry:
  %l1 = load double, double* @global, align 8
  %mul = fmul fast double %l1, %x
  store double %mul, double* @global, align 8
  %l2 = load double, double* @global, align 8
  %mul2 = fmul fast double %l2, %l2
  store double %mul2, double* @global, align 8
  %l3 = load double, double* @global, align 8
  ret double %l3
}

; Function Attrs: noinline nounwind uwtable
define %struct.Gradients @derivative(double %x) {
entry:
  %0 = tail call %struct.Gradients (double (double)*, ...) @__enzyme_fwddiff(double (double)* nonnull @mulglobal, metadata !"enzyme_width", i64 3, double %x, double 1.0, double 2.0, double 3.0)
  ret %struct.Gradients %0
}


; CHECK: define internal [3 x double] @fwddiffe3mulglobal(double %x, [3 x double] %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"global'ipa" = alloca double, align 8
; CHECK-NEXT:   %"global'ipa1" = alloca double, align 8
; CHECK-NEXT:   %"global'ipa2" = alloca double, align 8
; CHECK-NEXT:   %0 = bitcast double* %"global'ipa" to i8*
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull align 8 %0, i8 0, i64 8, i1 false)
; CHECK-NEXT:   %1 = bitcast double* %"global'ipa1" to i8*
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull align 8 %1, i8 0, i64 8, i1 false)
; CHECK-NEXT:   %2 = bitcast double* %"global'ipa2" to i8*
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull align 8 %2, i8 0, i64 8, i1 false)
; CHECK-NEXT:   %global_local.0.copyload = load double, double* @global, align 8
; CHECK-NEXT:   %mul = fmul fast double %global_local.0.copyload, %x
; CHECK-NEXT:   %3 = extractvalue [3 x double] %"x'", 0
; CHECK-NEXT:   %4 = fmul fast double %3, %global_local.0.copyload
; CHECK-NEXT:   %5 = extractvalue [3 x double] %"x'", 1
; CHECK-NEXT:   %6 = fmul fast double %5, %global_local.0.copyload
; CHECK-NEXT:   %7 = extractvalue [3 x double] %"x'", 2
; CHECK-NEXT:   %8 = fmul fast double %7, %global_local.0.copyload
; CHECK-NEXT:   %mul2 = fmul fast double %mul, %mul
; CHECK-NEXT:   %9 = fmul fast double %4, %mul
; CHECK-NEXT:   %10 = fmul fast double %4, %mul
; CHECK-NEXT:   %11 = fadd fast double %9, %10
; CHECK-NEXT:   %12 = insertvalue [3 x double] undef, double %11, 0
; CHECK-NEXT:   %13 = fmul fast double %6, %mul
; CHECK-NEXT:   %14 = fmul fast double %6, %mul
; CHECK-NEXT:   %15 = fadd fast double %13, %14
; CHECK-NEXT:   %16 = insertvalue [3 x double] %12, double %15, 1
; CHECK-NEXT:   %17 = fmul fast double %8, %mul
; CHECK-NEXT:   %18 = fmul fast double %8, %mul
; CHECK-NEXT:   %19 = fadd fast double %17, %18
; CHECK-NEXT:   %20 = insertvalue [3 x double] %16, double %19, 2
; CHECK-NEXT:   store double %mul2, double* @global, align 8
; CHECK-NEXT:   store double %11, double* %"global'ipa", align 8
; CHECK-NEXT:   store double %15, double* %"global'ipa1", align 8
; CHECK-NEXT:   store double %19, double* %"global'ipa2", align 8
; CHECK-NEXT:   ret [3 x double] %20
; CHECK-NEXT: }