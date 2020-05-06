; RUN: %opt < %s %loadEnzyme -enzyme -enzyme_preopt=false -inline -mem2reg -instcombine -early-cse -adce -S | FileCheck %s

; #include <math.h>
; 
; double sqrelu(double x) {
;     return (x > 0) ? sqrt(x * sin(x)) : 0;
; }
; 
; double dsqrelu(double x) {
;     return __builtin_autodiff(sqrelu, x);
; }

; Function Attrs: nounwind readnone uwtable
define dso_local double @sqrelu(double %x) #0 {
entry:
  %cmp = fcmp fast ogt double %x, 0.000000e+00
  br i1 %cmp, label %cond.true, label %cond.end

cond.true:                                        ; preds = %entry
  %0 = tail call fast double @llvm.sin.f64(double %x)
  %mul = fmul fast double %0, %x
  %1 = tail call fast double @llvm.sqrt.f64(double %mul)
  br label %cond.end

cond.end:                                         ; preds = %entry, %cond.true
  %cond = phi double [ %1, %cond.true ], [ 0.000000e+00, %entry ]
  ret double %cond
}

; Function Attrs: nounwind readnone speculatable
declare double @llvm.sin.f64(double) #1

; Function Attrs: nounwind readnone speculatable
declare double @llvm.sqrt.f64(double) #1

; Function Attrs: nounwind uwtable
define dso_local double @dsqrelu(double %x) local_unnamed_addr #2 {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_autodiff(double (double)* nonnull @sqrelu, double %x)
  ret double %0
}

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double)*, ...) #3

attributes #0 = { nounwind readnone uwtable }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { nounwind uwtable }
attributes #3 = { nounwind }

; CHECK: define dso_local double @dsqrelu(double %x) local_unnamed_addr
; CHECK-NEXT: entry:
; CHECK-NEXT:   %cmp.i = fcmp fast ogt double %x, 0.000000e+00
; CHECK-NEXT:   br i1 %cmp.i, label %cond.true.i, label %cond.end.i
; CHECK: cond.true.i:                                      ; preds = %entry
; CHECK-NEXT:   %0 = call fast double @llvm.sin.f64(double %x)
; CHECK-NEXT:   %mul.i = fmul fast double %0, %x
; CHECK-NEXT:   %1 = call fast double @llvm.sqrt.f64(double %mul.i)
; CHECK-NEXT:   br label %cond.end.i
; CHECK: cond.end.i:
; CHECK-NEXT:   %_cache.0.i = phi double [ %1, %cond.true.i ], [ undef, %entry ]
; CHECK-NEXT:   br i1 %cmp.i, label %invertcond.true.i, label %diffesqrelu.exit
; CHECK: invertcond.true.i:
; CHECK-NEXT:   %[[div:.+]] = fdiv fast double 5.000000e-01, %_cache.0.i
; CHECK-NEXT:   %[[dmul0:.+]] = fmul fast double %[[div]], %x
; CHECK-NEXT:   %[[dsin:.+]] = call fast double @llvm.sin.f64(double %x)
; CHECK-NEXT:   %[[dmul1:.+]] = fmul fast double %[[div]], %[[dsin]]
; CHECK-NEXT:   %[[dcos:.+]] = call fast double @llvm.cos.f64(double %x)
; CHECK-NEXT:   %[[fmul:.+]] = fmul fast double %[[dmul0]], %[[dcos]]
; CHECK-NEXT:   %[[res:.+]] = fadd fast double %[[dmul1]], %[[fmul]]
; CHECK-NEXT:   br label %diffesqrelu.exit
; CHECK: diffesqrelu.exit: 
; CHECK-NEXT:   %"x'de.0.i" = phi double [ %[[res]], %invertcond.true.i ], [ 0.000000e+00, %cond.end.i ]
; CHECK-NEXT:   ret double %"x'de.0.i"
; CHECK-NEXT: }
