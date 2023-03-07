; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

; __attribute__((noinline))
; double f(double x) {
;     return x;
; }
; 
; double relu(double x) {
;     return (x > 0) ? f(x) : 0;
; }
; 
; double drelu(double x) {
;     return __builtin_autodiff(relu, x);
; }

define dso_local double @f(double %x) #1 {
entry:
  ret double %x
}

define dso_local double @relu(double %x) {
entry:
  %cmp = fcmp fast ogt double %x, 0.000000e+00
  br i1 %cmp, label %cond.true, label %cond.end

cond.true:                                        ; preds = %entry
  %call = tail call fast double @f(double %x)
  br label %cond.end

cond.end:                                         ; preds = %entry, %cond.true
  %cond = phi double [ %call, %cond.true ], [ 0.000000e+00, %entry ]
  ret double %cond
}

define dso_local double @drelu(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_fwddiff(double (double)* nonnull @relu, double %x, double 1.0)
  ret double %0
}

declare double @__enzyme_fwddiff(double (double)*, ...) #0

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone noinline }

; CHECK: define internal double @fwddifferelu(double %x, double %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %cmp = fcmp fast ogt double %x, 0.000000e+00
; CHECK-NEXT:   br i1 %cmp, label %cond.true, label %cond.end
; CHECK: cond.true:                                ; preds = %entry
; CHECK-NEXT:   %0 = call fast double @fwddiffef(double %x, double %"x'")
; CHECK-NEXT:   br label %cond.end
; CHECK: cond.end:
; CHECK-NEXT:   %[[cond:.+]] = phi{{( fast)?}} double [ %0, %cond.true ], [ 0.000000e+00, %entry ]
; CHECK-NEXT:   ret double %[[cond]]
; CHECK-NEXT: }


; CHECK: define internal {{(dso_local )?}}double @fwddiffef(double %x, double %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret double %"x'"
; CHECK-NEXT: }
