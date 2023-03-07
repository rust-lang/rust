; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

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

; CHECK: define internal double @fwddiffesqrelu(double %x, double %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %cmp = fcmp fast ogt double %x, 0.000000e+00
; CHECK-NEXT:   br i1 %cmp, label %cond.true, label %cond.end

; CHECK: cond.true:
; CHECK-NEXT:   %0 = tail call fast double @llvm.sin.f64(double %x)
; CHECK-NEXT:   %1 = call fast double @llvm.cos.f64(double %x)
; CHECK-NEXT:   %2 = fmul fast double %"x'", %1
; CHECK-NEXT:   %mul = fmul fast double %0, %x
; CHECK-NEXT:   %3 = fmul fast double %2, %x
; CHECK-NEXT:   %4 = fmul fast double %"x'", %0
; CHECK-NEXT:   %5 = fadd fast double %3, %4
; CHECK-NEXT:   %6 = call fast double @llvm.sqrt.f64(double %mul)
; CHECK-NEXT:   %7 = fmul fast double 5.000000e-01, %5
; CHECK-NEXT:   %8 = fdiv fast double %7, %6
; CHECK-NEXT:   %9 = fcmp fast oeq double %mul, 0.000000e+00
; CHECK-NEXT:   %10 = select  {{(fast )?}}i1 %9, double 0.000000e+00, double %8
; CHECK-NEXT:   br label %cond.end

; CHECK: cond.end: 
; CHECK-NEXT:   %[[condi:.+]] = phi  {{(fast )?}}double [ %10, %cond.true ], [ 0.000000e+00, %entry ]
; CHECK-NEXT:   ret double %[[condi]]
; CHECK-NEXT: }
