; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -gvn -dse -S | FileCheck %s
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(gvn,dse)" -enzyme-preopt=false -S | FileCheck %s

; Function Attrs: noinline norecurse nounwind uwtable
define dso_local double @f(double* noalias nocapture %out, double %x) #0 {
entry:
  store double %x, double* %out, align 8
  store double 0.000000e+00, double* %out, align 8
  %res = load double, double* %out
  ret double %res
}

; Function Attrs: noinline nounwind uwtable
define dso_local double @dsumsquare(double* %x, double* %xp, double %inp, double %in2) local_unnamed_addr #1 {
entry:
  %call = tail call fast double @__enzyme_fwddiff(i8* bitcast (double (double*, double)* @f to i8*), double* %x, double* %xp, double %inp, double 1.0)
  ret double %call
}

declare dso_local double @__enzyme_fwddiff(i8*, double*, double*, double, double) local_unnamed_addr

attributes #0 = { noinline norecurse nounwind uwtable }
attributes #1 = { noinline nounwind uwtable }

; CHECK: define internal double @fwddiffef(double* noalias nocapture %out, double* nocapture %"out'", double %x, double %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   store double 0.000000e+00, double* %out, align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"out'", align 8
; CHECK-NEXT:   ret double 0.000000e+00
; CHECK-NEXT: }
