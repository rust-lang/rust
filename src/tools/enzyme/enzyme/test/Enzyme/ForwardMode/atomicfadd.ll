; RUN: if [ %llvmver -ge 9 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 9 ]; then %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s; fi

; Function Attrs: norecurse nounwind readonly uwtable
define dso_local double @sum(double* nocapture %n, double %x) #0 {
entry:
  %res = atomicrmw fadd double* %n, double %x monotonic
  ret double %res
}

; Function Attrs: nounwind uwtable
define dso_local void @dsum(double* %x, double* %xp, double %n) local_unnamed_addr #1 {
entry:
  %0 = tail call double (double (double*, double)*, ...) @__enzyme_fwddiff(double (double*, double)* nonnull @sum, double* %x, double* %xp, double %n, double 1.000000e+00)
  ret void
}

; Function Attrs: nounwind
declare double @__enzyme_fwddiff(double (double*, double)*, ...) #2

attributes #0 = { norecurse nounwind readonly uwtable }
attributes #1 = { nounwind uwtable }
attributes #2 = { nounwind }

; CHECK: define internal double @fwddiffesum(double* nocapture %n, double* nocapture %"n'", double %x, double %"x'") 
; CHECK-NEXT: entry:
; CHECK-NEXT:   %res = atomicrmw fadd double* %n, double %x monotonic
; CHECK-NEXT:   %0 = atomicrmw fadd double* %"n'", double %"x'" monotonic
; CHECK-NEXT:   ret double %0
; CHECK-NEXT: }
