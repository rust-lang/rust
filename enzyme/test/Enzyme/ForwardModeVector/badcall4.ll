; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -instsimplify -adce -correlated-propagation -simplifycfg -S | FileCheck %s

%struct.Gradients = type { double*, double*, double* }

; Function Attrs: nounwind
declare %struct.Gradients @__enzyme_fwddiff(i8*, ...)


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
define dso_local zeroext i1 @subf(double* nocapture %x) local_unnamed_addr #0 {
entry:
  %0 = load double, double* %x, align 8
  %mul = fmul fast double %0, 2.000000e+00
  store double %mul, double* %x, align 8
  %call = tail call zeroext i1 @metasubf(double* %x)
  %call1 = tail call zeroext i1 @othermetasubf(double* %x)
  %res = and i1 %call, %call1
  ret i1 %res
}

; Function Attrs: noinline norecurse nounwind uwtable
define dso_local void @f(double* nocapture %x) #0 {
entry:
  %call = tail call zeroext i1 @subf(double* %x)
  store double 2.000000e+00, double* %x, align 8
  ret void
}

; Function Attrs: noinline nounwind uwtable
define dso_local %struct.Gradients @dsumsquare(double* %x, double* %xp1, double* %xp2, double* %xp3) local_unnamed_addr #1 {
entry:
  %call = tail call %struct.Gradients (i8*, ...) @__enzyme_fwddiff(i8* bitcast (void (double*)* @f to i8*), metadata !"enzyme_width", i64 3, double* %x, double* %xp1, double* %xp2, double* %xp3)
  ret %struct.Gradients %call
}


; CHECK: define internal void @fwddiffe3f(double* nocapture %x, [3 x double*] %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @fwddiffe3subf(double* %x, [3 x double*] %"x'")
; CHECK-NEXT:   store double 2.000000e+00, double* %x, align 8
; CHECK-NEXT:   %0 = extractvalue [3 x double*] %"x'", 0
; CHECK-NEXT:   store double 0.000000e+00, double* %0, align 8
; CHECK-NEXT:   %1 = extractvalue [3 x double*] %"x'", 1
; CHECK-NEXT:   store double 0.000000e+00, double* %1, align 8
; CHECK-NEXT:   %2 = extractvalue [3 x double*] %"x'", 2
; CHECK-NEXT:   store double 0.000000e+00, double* %2, align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @fwddiffe3subf(double* nocapture %x, [3 x double*] %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = load double, double* %x, align 8
; CHECK-NEXT:   %1 = extractvalue [3 x double*] %"x'", 0
; CHECK-NEXT:   %2 = load double, double* %1, align 8
; CHECK-NEXT:   %3 = extractvalue [3 x double*] %"x'", 1
; CHECK-NEXT:   %4 = load double, double* %3, align 8
; CHECK-NEXT:   %5 = extractvalue [3 x double*] %"x'", 2
; CHECK-NEXT:   %6 = load double, double* %5, align 8
; CHECK-NEXT:   %mul = fmul fast double %0, 2.000000e+00
; CHECK-NEXT:   %7 = fmul fast double %2, 2.000000e+00
; CHECK-NEXT:   %8 = fmul fast double %4, 2.000000e+00
; CHECK-NEXT:   %9 = fmul fast double %6, 2.000000e+00
; CHECK-NEXT:   store double %mul, double* %x, align 8
; CHECK-NEXT:   %10 = extractvalue [3 x double*] %"x'", 0
; CHECK-NEXT:   store double %7, double* %10, align 8
; CHECK-NEXT:   %11 = extractvalue [3 x double*] %"x'", 1
; CHECK-NEXT:   store double %8, double* %11, align 8
; CHECK-NEXT:   %12 = extractvalue [3 x double*] %"x'", 2
; CHECK-NEXT:   store double %9, double* %12, align 8
; CHECK-NEXT:   call void @fwddiffe3metasubf(double* %x, [3 x double*] %"x'")
; CHECK-NEXT:   call void @fwddiffe3othermetasubf(double* %x, [3 x double*] %"x'")
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @fwddiffe3metasubf(double* nocapture %x, [3 x double*] %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = extractvalue [3 x double*] %"x'", 0
; CHECK-NEXT:   %"arrayidx'ipg" = getelementptr inbounds double, double* %0, i64 1
; CHECK-NEXT:   %1 = extractvalue [3 x double*] %"x'", 1
; CHECK-NEXT:   %"arrayidx'ipg1" = getelementptr inbounds double, double* %1, i64 1
; CHECK-NEXT:   %2 = extractvalue [3 x double*] %"x'", 2
; CHECK-NEXT:   %"arrayidx'ipg2" = getelementptr inbounds double, double* %2, i64 1
; CHECK-NEXT:   %arrayidx = getelementptr inbounds double, double* %x, i64 1
; CHECK-NEXT:   store double 3.000000e+00, double* %arrayidx, align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"arrayidx'ipg", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"arrayidx'ipg1", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"arrayidx'ipg2", align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @fwddiffe3othermetasubf(double* nocapture %x, [3 x double*] %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = extractvalue [3 x double*] %"x'", 0
; CHECK-NEXT:   %"arrayidx'ipg" = getelementptr inbounds double, double* %0, i64 1
; CHECK-NEXT:   %1 = extractvalue [3 x double*] %"x'", 1
; CHECK-NEXT:   %"arrayidx'ipg1" = getelementptr inbounds double, double* %1, i64 1
; CHECK-NEXT:   %2 = extractvalue [3 x double*] %"x'", 2
; CHECK-NEXT:   %"arrayidx'ipg2" = getelementptr inbounds double, double* %2, i64 1
; CHECK-NEXT:   %arrayidx = getelementptr inbounds double, double* %x, i64 1
; CHECK-NEXT:   store double 4.000000e+00, double* %arrayidx, align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"arrayidx'ipg", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"arrayidx'ipg1", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"arrayidx'ipg2", align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }