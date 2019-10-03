; RUN: opt < %s %loadEnzyme -enzyme -enzyme_preopt=false -inline -mem2reg -simplifycfg -instsimplify -S | FileCheck %s


; Function Attrs: noinline nounwind uwtable
define dso_local double @f(double %x) #0 {
entry:
  %retval = alloca double, align 8
  %res = load double, double* %retval, align 8
  ret double %res
}

; Function Attrs: noinline nounwind uwtable
define dso_local double @dsumsquare(double %x) #0 {
entry:
  %call = call fast double @__enzyme_autodiff(i8* bitcast (double (double)* @f to i8*), double %x)
  ret double %call
}

declare dso_local double @__enzyme_autodiff(i8*, double)

attributes #0 = { noinline nounwind uwtable }

; CHECK: define internal {{(dso_local )?}}{ double } @diffef(double %x, double %[[differet:.+]])
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret { double } zeroinitializer
; CHECK-NEXT: }
