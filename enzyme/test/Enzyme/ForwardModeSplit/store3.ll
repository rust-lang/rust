; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -adce -correlated-propagation -simplifycfg -gvn -dse -S | FileCheck %s

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
  %call = tail call fast double @__enzyme_fwdsplit(i8* bitcast (double (double*, double)* @f to i8*), double* %x, double* %xp, double %inp, double 1.0, i8* null)
  ret double %call
}

declare dso_local double @__enzyme_fwdsplit(i8*, double*, double*, double, double, i8*) local_unnamed_addr

attributes #0 = { noinline norecurse nounwind uwtable }
attributes #1 = { noinline nounwind uwtable }

; CHECK: define internal double @fwddiffef(double* noalias nocapture %out, double* nocapture %"out'", double %x, double %"x'", i8* %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   tail call void @free(i8* nonnull %tapeArg)
; CHECK-NEXT:   store double 0.000000e+00, double* %"out'", align 8
; CHECK-NEXT:   ret double 0.000000e+00
; CHECK-NEXT: }
