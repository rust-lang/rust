; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -inline -mem2reg -simplifycfg -adce -S | FileCheck %s

%struct.Gradients = type { double, double }

; Function Attrs: nounwind
declare %struct.Gradients @__enzyme_fwddiff(double (double)*, ...)

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
define dso_local %struct.Gradients @dsqrelu(double %x) local_unnamed_addr #2 {
entry:
  %0 = tail call %struct.Gradients (double (double)*, ...) @__enzyme_fwddiff(double (double)* nonnull @sqrelu, metadata !"enzyme_width", i64 2, double %x, double 1.0, double 1.5)
  ret %struct.Gradients %0
}

attributes #0 = { nounwind readnone uwtable }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { nounwind uwtable }
attributes #3 = { nounwind }


; CHECK: define dso_local %struct.Gradients @dsqrelu(double %x)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %cmp.i = fcmp fast ogt double %x, 0.000000e+00
; CHECK-NEXT:   br i1 %cmp.i, label %cond.true.i, label %fwddiffe2sqrelu.exit

; CHECK: cond.true.i:                                      ; preds = %entry
; CHECK-NEXT:   %0 = call fast double @llvm.sin.f64(double %x)
; CHECK-NEXT:   %1 = call fast double @llvm.cos.f64(double %x)
; CHECK-NEXT:   %2 = fmul fast double 1.500000e+00, %1
; CHECK-NEXT:   %mul.i = fmul fast double %0, %x
; CHECK-NEXT:   %3 = fmul fast double %1, %x
; CHECK-NEXT:   %4 = fadd fast double %3, %0
; CHECK-NEXT:   %5 = fmul fast double %2, %x
; CHECK-NEXT:   %6 = fmul fast double 1.500000e+00, %0
; CHECK-NEXT:   %7 = fadd fast double %5, %6
; CHECK-NEXT:   %8 = call fast double @llvm.sqrt.f64(double %mul.i)
; CHECK-NEXT:   %9 = fmul fast double 5.000000e-01, %4
; CHECK-NEXT:   %10 = fdiv fast double %9, %8
; CHECK-NEXT:   %11 = fcmp fast oeq double %mul.i, 0.000000e+00
; CHECK-NEXT:   %12 = select {{(fast )?}}i1 %11, double 0.000000e+00, double %10
; CHECK-NEXT:   %13 = insertvalue [2 x double] undef, double %12, 0
; CHECK-NEXT:   %14 = call fast double @llvm.sqrt.f64(double %mul.i)
; CHECK-NEXT:   %15 = fmul fast double 5.000000e-01, %7
; CHECK-NEXT:   %16 = fdiv fast double %15, %14
; CHECK-NEXT:   %17 = fcmp fast oeq double %mul.i, 0.000000e+00
; CHECK-NEXT:   %18 = select {{(fast )?}}i1 %17, double 0.000000e+00, double %16
; CHECK-NEXT:   %19 = insertvalue [2 x double] %13, double %18, 1
; CHECK-NEXT:   br label %fwddiffe2sqrelu.exit

; CHECK: fwddiffe2sqrelu.exit:                             ; preds = %entry, %cond.true.i
; CHECK-NEXT:   %"cond'.i" = phi {{(fast )?}}[2 x double] [ %19, %cond.true.i ], [ zeroinitializer, %entry ]
; CHECK-NEXT:   %20 = extractvalue [2 x double] %"cond'.i", 0
; CHECK-NEXT:   %21 = insertvalue %struct.Gradients zeroinitializer, double %20, 0
; CHECK-NEXT:   %22 = extractvalue [2 x double] %"cond'.i", 1
; CHECK-NEXT:   %23 = insertvalue %struct.Gradients %21, double %22, 1
; CHECK-NEXT:   ret %struct.Gradients %23
; CHECK-NEXT: }