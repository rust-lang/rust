; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -inline -early-cse -instcombine -simplifycfg -S | FileCheck %s

%struct.Gradients = type { double, double }

; Function Attrs: nounwind
declare %struct.Gradients @__enzyme_fwddiff(double (double)*, ...)


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

define dso_local %struct.Gradients @drelu(double %x) {
entry:
  %0 = tail call %struct.Gradients (double (double)*, ...) @__enzyme_fwddiff(double (double)* nonnull @relu, metadata !"enzyme_width", i64 2, double %x, double 0.0, double 1.0)
  ret %struct.Gradients %0
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone noinline }


; CHECK: define dso_local %struct.Gradients @drelu(double %x)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %cmp.i = fcmp fast ogt double %x, 0.000000e+00
; CHECK-NEXT:   br i1 %cmp.i, label %cond.true.i, label %fwddiffe2relu.exit

; CHECK: cond.true.i:                                      ; preds = %entry
; CHECK-NEXT:   %0 = call {{(fast )?}}[2 x double] @fwddiffe2f(double %x, [2 x double] [double 0.000000e+00, double 1.000000e+00])
; CHECK-NEXT:   br label %fwddiffe2relu.exit

; CHECK: fwddiffe2relu.exit:                          ; preds = %entry, %cond.true.i
; CHECK-NEXT:   %"cond'.i" = phi {{(fast )?}}[2 x double] [ %0, %cond.true.i ], [ zeroinitializer, %entry ]
; CHECK-NEXT:   %1 = extractvalue [2 x double] %"cond'.i", 0
; CHECK-NEXT:   %2 = insertvalue %struct.Gradients zeroinitializer, double %1, 0
; CHECK-NEXT:   %3 = extractvalue [2 x double] %"cond'.i", 1
; CHECK-NEXT:   %4 = insertvalue %struct.Gradients %2, double %3, 1
; CHECK-NEXT:   ret %struct.Gradients %4
; CHECK-NEXT: }

; CHECK: define internal [2 x double] @fwddiffe2f(double %x, [2 x double] %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret [2 x double] %"x'"
; CHECK-NEXT: }