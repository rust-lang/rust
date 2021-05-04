; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -adce -correlated-propagation -simplifycfg -S | FileCheck %s

; XFAIL: *
; a function returning a ptr with no arguments is mistakenly marked as constant in spite of accessing a global

@global = external dso_local local_unnamed_addr global double*, align 8, !enzyme_shadow !{double** @dglobal}
@dglobal = external dso_local local_unnamed_addr global double*, align 8

; Function Attrs: noinline norecurse nounwind readonly uwtable
define dso_local double* @myglobal() local_unnamed_addr #0 {
entry:
  %ptr = load double*, double** @global, align 8
  ret double* %ptr
}

; Function Attrs: noinline norecurse nounwind readonly uwtable
define dso_local double @mulglobal(double %x) #0 {
entry:
  %call = tail call double* @myglobal()
  %arrayidx = getelementptr inbounds double, double* %call, i64 2
  %0 = load double, double* %arrayidx, align 8
  %mul = fmul fast double %0, %x
  ret double %mul
}

; Function Attrs: noinline nounwind uwtable
define dso_local double @derivative(double %x) local_unnamed_addr #1 {
entry:
  %0 = tail call double (...) @__enzyme_autodiff.f64(double (double)* nonnull @mulglobal, double %x) #2
  ret double %0
}

declare double @__enzyme_autodiff.f64(...) local_unnamed_addr

attributes #0 = { noinline norecurse nounwind readonly uwtable }
attributes #1 = { noinline nounwind uwtable }
attributes #2 = { nounwind }

; CHECK: define internal { double } @diffemulglobal(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %call_augmented = call { double*, double*, double* } @augmented_myglobal()
; CHECK-NEXT:   %call = extractvalue { double*, double*, double* } %call_augmented, 1
; CHECK-NEXT:   %"call'ac" = extractvalue { double*, double*, double* } %call_augmented, 2
; CHECK-NEXT:   %"arrayidx'ipg" = getelementptr inbounds double, double* %"call'ac", i64 2
; CHECK-NEXT:   %arrayidx = getelementptr inbounds double, double* %call, i64 2
; CHECK-NEXT:   %0 = load double, double* %arrayidx, align 8
; CHECK-NEXT:   %m0diffe = fmul fast double %differeturn, %x
; CHECK-NEXT:   %m1diffex = fmul fast double %differeturn, %0
; CHECK-NEXT:   %1 = load double, double* %"arrayidx'ipg", align 8
; CHECK-NEXT:   %2 = fadd fast double %1, %m0diffe
; CHECK-NEXT:   store double %2, double* %"arrayidx'ipg", align 8
; CHECK-NEXT:   %3 = insertvalue { double } undef, double %m1diffex, 0
; CHECK-NEXT:   ret { double } %3
; CHECK-NEXT: }

; CHECK: define internal { double*, double*, double* } @augmented_myglobal()
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = alloca { double*, double*, double* }
; CHECK-NEXT:   %1 = getelementptr inbounds { double*, double*, double* }, { double*, double*, double* }* %0, i32 0, i32 0
; CHECK-NEXT:   %"ptr'ipl" = load double*, double** @dglobal, align 8
; CHECK-NEXT:   store double* %"ptr'ipl", double** %1
; CHECK-NEXT:   %ptr = load double*, double** @global, align 8
; CHECK-NEXT:   %2 = getelementptr inbounds { double*, double*, double* }, { double*, double*, double* }* %0, i32 0, i32 1
; CHECK-NEXT:   store double* %ptr, double** %2
; CHECK-NEXT:   %3 = getelementptr inbounds { double*, double*, double* }, { double*, double*, double* }* %0, i32 0, i32 2
; CHECK-NEXT:   store double* %"ptr'ipl", double** %3
; CHECK-NEXT:   %4 = load { double*, double*, double* }, { double*, double*, double* }* %0
; CHECK-NEXT:   ret { double*, double*, double* } %4
; CHECK-NEXT: }

; CHECK: define internal void @diffemyglobal(double* %"ptr'il_phi")
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret void
; CHECK-NEXT: }