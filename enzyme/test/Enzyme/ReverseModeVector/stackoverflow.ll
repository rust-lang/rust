; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -simplifycfg -S | FileCheck %s

define double @caller(double* %data, i64* %a4) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %next1, %loop ]
  %res = phi double [ 0.000000e+00, %entry ], [ %res, %loop ]
  %next1 = add nuw nsw i64 %i, 1
  %a19 = load i64, i64* %a4
  %gepk3 = getelementptr double, double* %data, i64 %i
  %datak = load double, double* %gepk3
  %add = fadd fast double %datak, %res
  %exitcond3 = icmp eq i64 %next1, %a19
  br i1 %exitcond3, label %exit, label %loop

exit:
    ret double %add
}

define dso_local void @derivative(double* %this, double* %dthis, i64* %xpr) {
  %call11 = call [1650 x double] (i8*, ...) @__enzyme_autodiff(i8* bitcast (double (double*, i64*)* @caller to i8*),  metadata !"enzyme_width", i64 1650, metadata !"enzyme_dupv", i32 8, double* %this, double* %dthis, i64* %xpr)
  ret void
}

declare dso_local [1650 x double] @__enzyme_autodiff(i8*,...)


; CHECK: define dso_local void @derivative(double* %this, double* %dthis, i64* %xpr)
