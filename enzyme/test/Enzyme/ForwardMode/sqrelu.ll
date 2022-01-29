; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -inline -mem2reg -instcombine -early-cse -adce -S | FileCheck %s

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
  %0 = tail call double (double (double)*, ...) @__enzyme_fwddiff(double (double)* nonnull @sqrelu, double %x, double 1.0)
  ret double %0
}

; Function Attrs: nounwind
declare double @__enzyme_fwddiff(double (double)*, ...) #3

attributes #0 = { nounwind readnone uwtable }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { nounwind uwtable }
attributes #3 = { nounwind }

; CHECK: define dso_local double @dsqrelu(double %x) local_unnamed_addr
; CHECK-NEXT: entry:
; CHECK-NEXT:   %cmp.i = fcmp fast ogt double %x, 0.000000e+00
; CHECK-NEXT:   br i1 %cmp.i, label %cond.true.i, label %fwddiffesqrelu.exit

; CHECK: cond.true.i:
; CHECK-NEXT:   %0 = call fast double @llvm.sin.f64(double %x)
; CHECK-NEXT:   %1 = call fast double @llvm.cos.f64(double %x)
; CHECK-NEXT:   %mul.i = fmul fast double %0, %x
; CHECK-NEXT:   %2 = fmul fast double %1, %x
; CHECK-NEXT:   %3 = fadd fast double %2, %0
; CHECK-NEXT:   %4 = call fast double @llvm.sqrt.f64(double %mul.i)
; CHECK-NEXT:   %5 = fmul fast double %3, 5.000000e-01
; CHECK-NEXT:   %6 = fdiv fast double %5, %4
; CHECK-NEXT:   %7 = fcmp fast oeq double %mul.i, 0.000000e+00
; CHECK-NEXT:   %8 = select  {{(fast )?}}i1 %7, double 0.000000e+00, double %6
; CHECK-NEXT:   br label %fwddiffesqrelu.exit

; CHECK: fwddiffesqrelu.exit: 
; CHECK-NEXT:   %"cond'.i" = phi  {{(fast )?}}double [ %8, %cond.true.i ], [ 0.000000e+00, %entry ]
; CHECK-NEXT:   ret double %"cond'.i"
; CHECK-NEXT: }
