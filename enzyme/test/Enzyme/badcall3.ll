; RUN: opt < %s %loadEnzyme -enzyme -enzyme_preopt=false -mem2reg -instsimplify -adce -correlated-propagation -simplifycfg -S | FileCheck %s

; Function Attrs: noinline norecurse nounwind uwtable
define dso_local zeroext i1 @metasubf(double* nocapture %x) local_unnamed_addr #0 {
entry:
  %arrayidx = getelementptr inbounds double, double* %x, i64 1
  store double 3.000000e+00, double* %arrayidx, align 8
  %0 = load double, double* %x, align 8
  %cmp = fcmp fast oeq double %0, 2.000000e+00
  ret i1 %cmp
}

; Function Attrs: noinline norecurse nounwind uwtable
define dso_local zeroext i1 @othermetasubf(double* nocapture %x) local_unnamed_addr #0 {
entry:
  %arrayidx = getelementptr inbounds double, double* %x, i64 1
  store double 4.000000e+00, double* %arrayidx, align 8
  %0 = load double, double* %x, align 8
  %cmp = fcmp fast oeq double %0, 3.000000e+00
  ret i1 %cmp
}

; Function Attrs: noinline norecurse nounwind uwtable
define dso_local void @subf(double* nocapture %x) local_unnamed_addr #0 {
entry:
  %0 = load double, double* %x, align 8
  %mul = fmul fast double %0, 2.000000e+00
  store double %mul, double* %x, align 8
  %call = tail call zeroext i1 @metasubf(double* %x)
  %call1 = tail call zeroext i1 @othermetasubf(double* %x)
  ret void
}

; Function Attrs: noinline norecurse nounwind uwtable
define dso_local void @f(double* nocapture %x) #0 {
entry:
  tail call void @subf(double* %x)
  store double 2.000000e+00, double* %x, align 8
  ret void
}

; Function Attrs: noinline nounwind uwtable
define dso_local double @dsumsquare(double* %x, double* %xp) local_unnamed_addr #1 {
entry:
  %call = tail call fast double @__enzyme_autodiff(i8* bitcast (void (double*)* @f to i8*), double* %x, double* %xp)
  ret double %call
}

declare dso_local double @__enzyme_autodiff(i8*, double*, double*) local_unnamed_addr

; CHECK: define internal {} @diffef(double* nocapture %x, double* %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call { { {}, {} } } @augmented_subf(double* %x, double* %"x'")
; CHECK-NEXT:   store double 2.000000e+00, double* %x, align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"x'", align 8
; CHECK-NEXT:   %1 = call {} @diffesubf(double* nonnull %x, double* %"x'", { {}, {} } undef)
; CHECK-NEXT:   ret {} undef
; CHECK-NEXT: }

; CHECK: define internal { {} } @augmented_othermetasubf(double* nocapture %x, double* %"x'") 
; CHECK-NEXT: entry:
; CHECK-NEXT:   %arrayidx = getelementptr inbounds double, double* %x, i64 1
; CHECK-NEXT:   store double 4.000000e+00, double* %arrayidx, align 8
; CHECK-NEXT:   ret { {} } undef
; CHECK-NEXT: }

; CHECK: define internal { {} } @augmented_metasubf(double* nocapture %x, double* %"x'") 
; CHECK-NEXT: entry:
; CHECK-NEXT:   %arrayidx = getelementptr inbounds double, double* %x, i64 1
; CHECK-NEXT:   store double 3.000000e+00, double* %arrayidx, align 8
; CHECK-NEXT:   ret { {} } undef
; CHECK-NEXT: }

; CHECK: define internal { { {}, {} } } @augmented_subf(double* nocapture %x, double* %"x'") 
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = load double, double* %x, align 8
; CHECK-NEXT:   %mul = fmul fast double %0, 2.000000e+00
; CHECK-NEXT:   store double %mul, double* %x, align 8
; CHECK-NEXT:   %1 = call { {} } @augmented_metasubf(double* %x, double* %"x'")
; CHECK-NEXT:   %2 = call { {} } @augmented_othermetasubf(double* %x, double* %"x'")
; CHECK-NEXT:   ret { { {}, {} } } undef
; CHECK-NEXT: }

; CHECK: define internal {} @diffesubf(double* nocapture %x, double* %"x'", { {}, {} } %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call {} @diffeothermetasubf(double* %x, double* %"x'", {} undef)
; CHECK-NEXT:   %1 = call {} @diffemetasubf(double* %x, double* %"x'", {} undef)
; CHECK-NEXT:   %2 = load double, double* %"x'"
; CHECK-NEXT:   store double 0.000000e+00, double* %"x'"
; CHECK-NEXT:   %m0diffe = fmul fast double %2, 2.000000e+00
; CHECK-NEXT:   %3 = load double, double* %"x'"
; CHECK-NEXT:   %4 = fadd fast double %3, %m0diffe
; CHECK-NEXT:   store double %4, double* %"x'"
; CHECK-NEXT:   ret {} undef
; CHECK-NEXT: }

; CHECK: define internal {} @diffeothermetasubf(double* nocapture %x, double* %"x'", {} %tapeArg) 
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[tostore:.+]] = getelementptr inbounds double, double* %"x'", i64 1
; CHECK-NEXT:   store double 0.000000e+00, double* %[[tostore]], align 8
; CHECK-NEXT:   ret {} undef
; CHECK-NEXT: }

; CHECK: define internal {} @diffemetasubf(double* nocapture %x, double* %"x'", {} %tapeArg) 
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[tostore2:.+]] = getelementptr inbounds double, double* %"x'", i64 1
; CHECK-NEXT:   store double 0.000000e+00, double* %[[tostore2]], align 8
; CHECK-NEXT:   ret {} undef
; CHECK-NEXT: }
