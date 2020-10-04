; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -inline -early-cse -instcombine -simplifycfg -S | FileCheck %s

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
  %0 = tail call double (double (double)*, ...) @__enzyme_autodiff(double (double)* nonnull @relu, double %x)
  ret double %0
}

declare double @__enzyme_autodiff(double (double)*, ...) #0

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone noinline }

; CHECK: define dso_local double @drelu(double %x)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %cmp.i = fcmp fast ogt double %x, 0.000000e+00
; CHECK-NEXT:   br i1 %cmp.i, label %invertcond.true.i, label %differelu.exit
; CHECK: invertcond.true.i:                                ; preds = %entry
; CHECK-NEXT:   %[[diffef:.+]] = call { double } @diffef(double %x, double 1.000000e+00)
; CHECK-NEXT:   %[[result:.+]] = extractvalue { double } %[[diffef]], 0
; CHECK-NEXT:   br label %differelu.exit
; CHECK: differelu.exit:                                   ; preds = %entry, %invertcond.true.i
; CHECK-NEXT:   %"x'de.0.i" = phi double [ %[[result]], %invertcond.true.i ], [ 0.000000e+00, %entry ]
; CHECK-NEXT:   ret double %"x'de.0.i"
; CHECK-NEXT: }

; CHECK: define internal {{(dso_local )?}}{ double } @diffef(double %x, double %[[differet:.+]])
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[result:.+]] = insertvalue { double } undef, double %[[differet]], 0
; CHECK-NEXT:   ret { double } %[[result]]
; CHECK-NEXT: }
