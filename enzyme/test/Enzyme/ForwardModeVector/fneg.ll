; RUN: if [ %llvmver -ge 10 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi

%struct.Gradients = type { double, double, double }

declare %struct.Gradients @__enzyme_fwddiff(double (double)*, ...)


define dso_local double @fneg(double %x) {
entry:
  %fneg = fneg double %x
  ret double %fneg
}

define dso_local void @fnegd(double %x) {
entry:
  %0 = call %struct.Gradients (double (double)*, ...) @__enzyme_fwddiff(double (double)* nonnull @fneg,  metadata !"enzyme_width", i64 3, double %x, double 1.0, double 2.5, double 3.0)
  ret void
}


; CHECK: define internal [3 x double] @fwddiffe3fneg(double %x, [3 x double] %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = extractvalue [3 x double] %"x'", 0
; CHECK-NEXT:   %1 = fneg fast double %0
; CHECK-NEXT:   %2 = insertvalue [3 x double] undef, double %1, 0
; CHECK-NEXT:   %3 = extractvalue [3 x double] %"x'", 1
; CHECK-NEXT:   %4 = fneg fast double %3
; CHECK-NEXT:   %5 = insertvalue [3 x double] %2, double %4, 1
; CHECK-NEXT:   %6 = extractvalue [3 x double] %"x'", 2
; CHECK-NEXT:   %7 = fneg fast double %6
; CHECK-NEXT:   %8 = insertvalue [3 x double] %5, double %7, 2
; CHECK-NEXT:   ret [3 x double] %8
; CHECK-NEXT: }