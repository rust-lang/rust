; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

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
define dso_local zeroext i1 @subf(double* nocapture %x) local_unnamed_addr #0 {
entry:
  %0 = load double, double* %x, align 8
  %mul = fmul fast double %0, 2.000000e+00
  store double %mul, double* %x, align 8
  %call = tail call zeroext i1 @metasubf(double* %x)
  ret i1 %call
}

; Function Attrs: noinline norecurse nounwind uwtable
define dso_local void @f(double* nocapture %x) #0 {
entry:
  %call = tail call zeroext i1 @subf(double* %x)
  store double 2.000000e+00, double* %x, align 8
  ret void
}

; Function Attrs: noinline nounwind uwtable
define dso_local double @dsumsquare(double* %x, double* %xp) local_unnamed_addr #1 {
entry:
  %call = tail call fast double @__enzyme_fwddiff(i8* bitcast (void (double*)* @f to i8*), double* %x, double* %xp)
  ret double %call
}

declare dso_local double @__enzyme_fwddiff(i8*, double*, double*) local_unnamed_addr

attributes #0 = { noinline norecurse nounwind uwtable }
attributes #1 = { noinline nounwind uwtable }

; CHECK: define internal {{(dso_local )?}}void @fwddiffef(double* nocapture %x, double* nocapture %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:	call void @fwddiffesubf(double* %x, double* %"x'")
; CHECK-NEXT:	store double 2.000000e+00, double* %x
; CHECK-NEXT:	store double 0.000000e+00, double* %"x'"
; CHECK-NEXT:	ret void
; CHECK-NEXT: }

; CHECK: define internal {{(dso_local )?}}void @fwddiffesubf(double* nocapture %x, double* nocapture %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[i1:.+]] = load double, double* %"x'"
; CHECK-NEXT:   %[[i0:.+]] = load double, double* %x
; CHECK-NEXT:   %mul = fmul fast double %[[i0]], 2.000000e+00
; CHECK-NEXT:   %[[i2:.+]] = fmul fast double %[[i1]], 2.000000e+00
; CHECK-NEXT:   store double %mul, double* %x
; CHECK-NEXT:   store double %[[i2]], double* %"x'"
; CHECK-NEXT:   call void @fwddiffemetasubf(double* %x, double* %"x'")
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal {{(dso_local )?}}void @fwddiffemetasubf(double* nocapture %x, double* nocapture %"x'") 
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"arrayidx'ipg" = getelementptr inbounds double, double* %"x'", i64 1
; CHECK-NEXT:   %arrayidx = getelementptr inbounds double, double* %x, i64 1
; CHECK-NEXT:   store double 3.000000e+00, double* %arrayidx
; CHECK-NEXT:   store double 0.000000e+00, double* %"arrayidx'ipg"
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
