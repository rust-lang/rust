; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-lower-globals -sroa -instsimplify -adce -S | FileCheck %s
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(sroa,instsimplify,adce)" -enzyme-preopt=false -enzyme-lower-globals -S | FileCheck %s

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
define double @derivative(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_fwddiff(double (double)* nonnull @mulglobal, double %x, double 1.0)
  ret double %0
}

; Function Attrs: nounwind
declare double @__enzyme_fwddiff(double (double)*, ...)

; CHECK: define internal double @fwddiffemulglobal(double %x, double %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %global_local.0.copyload = load double, double* @global, align 8
; CHECK-NEXT:   %mul = fmul fast double %global_local.0.copyload, %x
; CHECK-NEXT:   %0 = fmul fast double %"x'", %global_local.0.copyload
; CHECK-NEXT:   %mul2 = fmul fast double %mul, %mul
; CHECK-NEXT:   %1 = fmul fast double %0, %mul
; CHECK-NEXT:   %2 = fmul fast double %0, %mul
; CHECK-NEXT:   %3 = fadd fast double %1, %2
; CHECK-NEXT:   store double %mul2, double* @global, align 8
; CHECK-NEXT:   ret double %3
; CHECK-NEXT: }
