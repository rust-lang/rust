; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -adce -correlated-propagation -simplifycfg -S | FileCheck %s

; XFAIL: *
; a function returning a float with no arguments is mistakenly marked as constant in spite of accessing a global

@global = external dso_local local_unnamed_addr global double, align 8, !enzyme_shadow !{double* @dglobal}
@dglobal = external dso_local local_unnamed_addr global double, align 8

; Function Attrs: noinline norecurse nounwind readonly uwtable
define dso_local double @myglobal() local_unnamed_addr #0 {
entry:
  %flt = load double, double* @global, align 8
  ret double %flt
}

; Function Attrs: noinline norecurse nounwind readonly uwtable
define dso_local double @mulglobal(double %x) #0 {
entry:
  %call = tail call double @myglobal()
  %mul = fmul fast double %call, %x
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
; CHECK-NEXT:   %call = tail call double @myglobal()
; CHECK-NEXT:   %m0diffe = fmul fast double %differeturn, %x
; CHECK-NEXT:   %m1diffex = fmul fast double %differeturn, %0
; CHECK-NEXT:   call void @diffe_myglobal(double %m0diffe)
; CHECK-NEXT:   %3 = insertvalue { double } undef, double %m1diffex, 0
; CHECK-NEXT:   ret { double } %3
; CHECK-NEXT: }