; RUN: %opt < %s %loadEnzyme -enzyme -enzyme_preopt=false -mem2reg -sroa -instsimplify -adce -correlated-propagation -simplifycfg -S | FileCheck %s

; Function Attrs: noinline norecurse nounwind uwtable
define dso_local void @subf(i1 zeroext %z, double* nocapture %x) local_unnamed_addr #0 {
entry:
  br i1 %z, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %0 = load double, double* %x, align 8
  %mul = fmul fast double %0, %0
  store double %mul, double* %x, align 8
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

; Function Attrs: noinline norecurse nounwind uwtable
define dso_local void @f(i1 zeroext %z, double* nocapture %x) #0 {
entry:
  tail call void @subf(i1 zeroext %z, double* %x)
  %arrayidx = getelementptr inbounds double, double* %x, i64 1
  store double 2.000000e+00, double* %arrayidx, align 8
  ret void
}

; Function Attrs: noinline nounwind uwtable
define dso_local double @dsumsquare(i1 zeroext %z, double* %x, double* %xp) local_unnamed_addr #1 {
entry:
  %call = tail call fast double @__enzyme_autodiff(i8* bitcast (void (i1, double*)* @f to i8*), i1 zeroext %z, double* %x, double* %xp)
  ret double %call
}

declare dso_local double @__enzyme_autodiff(i8*, i1 zeroext, double*, double*)

; CHECK: define internal {} @diffef(i1 zeroext %z, double* nocapture %x, double* %"x'") {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[augsubf:.+]] = call { { double } } @augmented_subf(i1 %z, double* %x, double* %"x'")
; CHECK-NEXT:   %[[subf:.+]] = extractvalue { { double } } %[[augsubf]], 0
; CHECK-NEXT:   %"arrayidx'ipge" = getelementptr inbounds double, double* %"x'", i64 1
; CHECK-NEXT:   %arrayidx = getelementptr inbounds double, double* %x, i64 1
; CHECK-NEXT:   store double 2.000000e+00, double* %arrayidx, align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"arrayidx'ipge", align 8
; CHECK-NEXT:   %[[dsubf:.+]] = call {} @diffesubf(i1 %z, double* nonnull %x, double* %"x'", { double } %[[subf]])
; CHECK-NEXT:   ret {} undef
; CHECK-NEXT: }

; CHECK: define internal { { double } } @augmented_subf(i1 zeroext %z, double* nocapture %x, double* %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   br i1 %z, label %if.then, label %if.end

; CHECK: if.then:                                          ; preds = %entry
; CHECK-NEXT:   %0 = load double, double* %x, align 8
; CHECK-NEXT:   %mul = fmul fast double %0, %0
; CHECK-NEXT:   store double %mul, double* %x, align 8
; CHECK-NEXT:   br label %if.end

; CHECK: if.end:                                           ; preds = %if.then, %entry
; CHECK-NEXT:   %[[val:.+]] = phi double [ %0, %if.then ], [ undef, %entry ]
; CHECK-NEXT:   %[[toret:.+]] = insertvalue { { double } } undef, double %[[val]], 0, 0 
; CHECK-NEXT:   ret { { double } } %[[toret]]
; CHECK-NEXT: }

; CHECK: define internal {} @diffesubf(i1 zeroext %z, double* nocapture %x, double* %"x'", { double } %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   br i1 %z, label %invertif.then, label %invertentry

; CHECK: invertentry:                                      ; preds = %entry, %invertif.then
; CHECK-NEXT:   ret {} undef

; CHECK: invertif.then:                                    ; preds = %entry
; CHECK-NEXT:   %0 = load double, double* %"x'"
; CHECK-NEXT:   store double 0.000000e+00, double* %"x'", align 8
; CHECK-NEXT:   %[[uw:.+]] = extractvalue { double } %tapeArg, 0
; CHECK-NEXT:   %m0diffe = fmul fast double %0, %[[uw]]
; CHECK-NEXT:   %m1diffe = fmul fast double %0, %[[uw]]
; CHECK-NEXT:   %1 = fadd fast double %m0diffe, %m1diffe
; CHECK-NEXT:   %2 = load double, double* %"x'"
; CHECK-NEXT:   %3 = fadd fast double %2, %1
; CHECK-NEXT:   store double %3, double* %"x'"
; CHECK-NEXT:   br label %invertentry
; CHECK-NEXT: }
