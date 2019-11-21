; RUN: %opt < %s %loadEnzyme -enzyme -enzyme_preopt=false -mem2reg -instsimplify -adce -correlated-propagation -simplifycfg -S | FileCheck %s

; Function Attrs: noinline norecurse nounwind uwtable
define dso_local zeroext i1 @metasubf(double* nocapture %x) local_unnamed_addr #0 {
entry:
  %arrayidx = getelementptr inbounds double, double* %x, i64 1
  store double 3.000000e+00, double* %arrayidx, align 8
  %0 = load double, double* %x, align 8
  %cmp = fcmp fast oeq double %0, 2.000000e+00
  ret i1 %cmp
}

define dso_local zeroext i1 @omegasubf(double* nocapture %x) local_unnamed_addr #0 {
entry:
  %arrayidx = getelementptr inbounds double, double* %x, i64 1
  store double 3.000000e+00, double* %arrayidx, align 8
  %0 = load double, double* %x, align 8
  %cmp = fcmp fast oeq double %0, 2.000000e+00
  ret i1 %cmp
}

; Function Attrs: noinline norecurse nounwind uwtable
define dso_local zeroext i1 @subf(double* nocapture %x) local_unnamed_addr #0 {
entry:
  %0 = load double, double* %x, align 8
  %mul = fmul fast double %0, 2.000000e+00
  store double %mul, double* %x, align 8
  %call = tail call zeroext i1 @omegasubf(double* %x)
  %call2 = tail call zeroext i1 @metasubf(double* %x)
  ret i1 %call2
}

; Function Attrs: noinline norecurse nounwind uwtable
define dso_local void @f(double* nocapture %x) #0 {
entry:
  %call = tail call zeroext i1 @subf(double* %x)
  %sel = select i1 %call, double 2.000000e+00, double 3.000000e+00
  store double %sel, double* %x, align 8
  ret void
}

; Function Attrs: noinline nounwind uwtable
define dso_local double @dsumsquare(double* %x, double* %xp) local_unnamed_addr #1 {
entry:
  %call = tail call fast double @__enzyme_autodiff(i8* bitcast (void (double*)* @f to i8*), double* %x, double* %xp)
  ret double %call
}

declare dso_local double @__enzyme_autodiff(i8*, double*, double*) local_unnamed_addr

attributes #0 = { noinline norecurse nounwind uwtable }
attributes #1 = { noinline nounwind uwtable }

; CHECK: define internal {{(dso_local )?}}{} @diffef(double* nocapture %x, double* %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[augsubf:.+]] = call { { {}, {}, double }, i1 } @augmented_subf(double* %x, double* %"x'")
; CHECK-NEXT:   %[[augcache:.+]] = extractvalue { { {}, {}, double }, i1 } %[[augsubf]], 0
; CHECK-NEXT:   %[[augret:.+]] = extractvalue { { {}, {}, double }, i1 } %[[augsubf]], 1
; CHECK-NEXT:   %sel = select i1 %[[augret]], double 2.000000e+00, double 3.000000e+00
; CHECK-NEXT:   store double %sel, double* %x, align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"x'"
; CHECK-NEXT:   %[[dsubf:.+]] = call {} @diffesubf(double* nonnull %x, double* %"x'", { {}, {}, double } %[[augcache]])
; CHECK-NEXT:   ret {} undef
; CHECK-NEXT: }

; CHECK: define internal {{(dso_local )?}}{ {}, i1 } @augmented_metasubf(double* nocapture %x, double* %"x'") 
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = alloca { {}, i1 }
; CHECK-NEXT:   %arrayidx = getelementptr inbounds double, double* %x, i64 1
; CHECK-NEXT:   store double 3.000000e+00, double* %arrayidx, align 8
; CHECK-NEXT:   %1 = load double, double* %x, align 8
; CHECK-NEXT:   %cmp = fcmp fast oeq double %1, 2.000000e+00
; CHECK-NEXT:   %2 = getelementptr { {}, i1 }, { {}, i1 }* %0, i32 0, i32 1
; CHECK-NEXT:   store i1 %cmp, i1* %2
; CHECK-NEXT:   %3 = load { {}, i1 }, { {}, i1 }* %0
; CHECK-NEXT:   ret { {}, i1 } %3
; CHECK-NEXT: }

; CHECK: define internal {{(dso_local )?}}{ {} } @augmented_omegasubf(double* nocapture %x, double* %"x'") 
; CHECK-NEXT: entry:
; CHECK-NEXT:   %arrayidx = getelementptr inbounds double, double* %x, i64 1
; CHECK-NEXT:   store double 3.000000e+00, double* %arrayidx, align 8
; CHECK-NEXT:   ret { {} } undef
; CHECK-NEXT: }

; CHECK: define internal {{(dso_local )?}}{ { {}, {}, double }, i1 } @augmented_subf(double* nocapture %x, double* %"x'") 
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = alloca { { {}, {}, double }, i1 }
; CHECK-NEXT:   %1 = getelementptr { { {}, {}, double }, i1 }, { { {}, {}, double }, i1 }* %0, i32 0, i32 0
; CHECK-NEXT:   %2 = load double, double* %x, align 8
; CHECK-NEXT:   %3 = getelementptr { {}, {}, double }, { {}, {}, double }* %1, i32 0, i32 2
; CHECK-NEXT:   store double %2, double* %3
; CHECK-NEXT:   %mul = fmul fast double %2, 2.000000e+00
; CHECK-NEXT:   store double %mul, double* %x, align 8
; CHECK-NEXT:   %[[augsubf:.+]] = call { {} } @augmented_omegasubf(double* %x, double* %"x'")
; CHECK-NEXT:   %[[augmetasubf:.+]] = call { {}, i1 } @augmented_metasubf(double* %x, double* %"x'")
; CHECK-NEXT:   %[[metasubf:.+]] = extractvalue { {}, i1 } %[[augmetasubf]], 1
; CHECK-NEXT:   %[[gep:.+]] = getelementptr { { {}, {}, double }, i1 }, { { {}, {}, double }, i1 }* %0, i32 0, i32 1
; CHECK-NEXT:   store i1 %[[metasubf]], i1* %[[gep]]
; CHECK-NEXT:   %[[toret:.+]] = load { { {}, {}, double }, i1 }, { { {}, {}, double }, i1 }* %0
; CHECK-NEXT:   ret { { {}, {}, double }, i1 } %[[toret]]
; CHECK-NEXT: }

; CHECK: define internal {{(dso_local )?}}{} @diffesubf(double* nocapture %x, double* %"x'", { {}, {}, double } %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call {} @diffemetasubf(double* %x, double* %"x'", {} undef)
; CHECK-NEXT:   %1 = call {} @diffeomegasubf(double* %x, double* %"x'", {} undef)
; CHECK-NEXT:   %2 = load double, double* %"x'"
; CHECK-NEXT:   store double 0.000000e+00, double* %"x'"
; CHECK-NEXT:   %m0diffe = fmul fast double %2, 2.000000e+00
; CHECK-NEXT:   %3 = load double, double* %"x'"
; CHECK-NEXT:   %4 = fadd fast double %3, %m0diffe
; CHECK-NEXT:   store double %4, double* %"x'"
; CHECK-NEXT:   ret {} undef
; CHECK-NEXT: }

; CHECK: define internal {{(dso_local )?}}{} @diffemetasubf(double* nocapture %x, double* %"x'", {} %tapeArg) 
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[tostore:.+]] = getelementptr inbounds double, double* %"x'", i64 1
; CHECK-NEXT:   store double 0.000000e+00, double* %[[tostore]], align 8
; CHECK-NEXT:   ret {} undef
; CHECK-NEXT: }

; CHECK: define internal {{(dso_local )?}}{} @diffeomegasubf(double* nocapture %x, double* %"x'", {} %tapeArg) 
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[tostore:.+]] = getelementptr inbounds double, double* %"x'", i64 1
; CHECK-NEXT:   store double 0.000000e+00, double* %[[tostore]], align 8
; CHECK-NEXT:   ret {} undef
; CHECK-NEXT: }
