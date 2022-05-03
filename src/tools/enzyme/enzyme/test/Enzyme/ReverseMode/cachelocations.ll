; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -instsimplify -adce -correlated-propagation -simplifycfg -S | FileCheck %s

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

; CHECK: define internal void @diffef(i1 zeroext %z, double* nocapture %x, double* nocapture %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[augsubf:.+]] = call fast double @augmented_subf(i1 %z, double* %x, double* %"x'")
; CHECK-NEXT:   %[[arrayidxipge:.+]] = getelementptr inbounds double, double* %"x'", i64 1
; CHECK-NEXT:   %arrayidx = getelementptr inbounds double, double* %x, i64 1
; CHECK-NEXT:   store double 2.000000e+00, double* %arrayidx, align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %[[arrayidxipge]], align 8
; CHECK-NEXT:   call void @diffesubf(i1 %z, double* nonnull %x, double* %"x'", double %[[augsubf]])
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal double @augmented_subf(i1 zeroext %z, double* nocapture %x, double* nocapture %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   br i1 %z, label %if.then, label %if.end

; CHECK: if.then:                                          ; preds = %entry
; CHECK-NEXT:   %0 = load double, double* %x, align 8
; CHECK-NEXT:   %mul = fmul fast double %0, %0
; CHECK-NEXT:   store double %mul, double* %x, align 8
; CHECK-NEXT:   br label %if.end

; CHECK: if.end:                                           ; preds = %if.then, %entry
; CHECK-NEXT:   %[[val:.+]] = phi double [ %0, %if.then ], [ undef, %entry ]
; CHECK-NEXT:   ret double %[[val]]
; CHECK-NEXT: }

; CHECK: define internal void @diffesubf(i1 zeroext %z, double* nocapture %x, double* nocapture %"x'", double
; CHECK-NEXT: entry:
; CHECK-NEXT:   br i1 %z, label %invertif.then, label %invertentry

; CHECK: invertentry:                                      ; preds = %entry, %invertif.then
; CHECK-NEXT:   ret void

; CHECK: invertif.then:                                    ; preds = %entry
; CHECK-NEXT:   %[[px:.+]] = load double, double* %"x'"
; CHECK-NEXT:   store double 0.000000e+00, double* %"x'", align 8
; CHECK-NEXT:   %m0diffe = fmul fast double %[[px]], %0
; CHECK-NEXT:   %m1diffe = fmul fast double %[[px]], %0
; CHECK-NEXT:   %[[de:.+]] = fadd fast double %m0diffe, %m1diffe
; CHECK-NEXT:   %[[ppx:.+]] = load double, double* %"x'"
; CHECK-NEXT:   %[[postx:.+]] = fadd fast double %[[ppx]], %[[de]]
; CHECK-NEXT:   store double %[[postx]], double* %"x'"
; CHECK-NEXT:   br label %invertentry
; CHECK-NEXT: }
