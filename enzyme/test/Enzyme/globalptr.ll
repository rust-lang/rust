; RUN: opt < %s %loadEnzyme -enzyme -enzyme_preopt=false -mem2reg -instsimplify -adce -correlated-propagation -simplifycfg -S | FileCheck %s

; XFAIL: *
; a function returning a pointer/float with no arguments is mistakenly marked as constant in spite of accessing a global

@global = external dso_local local_unnamed_addr global double*, align 8

; Function Attrs: noinline norecurse nounwind readonly uwtable
define dso_local double* @myglobal() local_unnamed_addr #0 {
entry:
  %0 = load double*, double** @global, align 8
  ret double* %0
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
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %call = tail call double* @myglobal()
; CHECK-NEXT:    %arrayidx = getelementptr inbounds double, double* %call, i64 2
; CHECK-NEXT:    %0 = load double, double* %arrayidx, align 8
; CHECK-NEXT:    %[[tmul:.+]] = fmul fast double %0, %x
; CHECK-NEXT:    %[[tcall:.+]] = call {} @diffemyglobal(double %x)
