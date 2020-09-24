; RUN: %opt < %s %loadEnzyme -enzyme -enzyme_preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s
; XFAIL: *
; TODO, this currently fails because as Enzyme we don't run DSE

declare noalias i8* @malloc(i64)

define double @sub(double %x, i64 %y) {
entry:
  %malloccall = tail call i8* @malloc(i64 8)
  %bc = bitcast i8* %malloccall to i64*
  store i64 %y, i64* %bc, align 8
  ret double %x
}

define double @caller(double %x) {
entry:
  %call = tail call double @sub(double %x, i64 0)
  ret double %call
}

define dso_local double @dcaller(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_autodiff(double (double)* nonnull @caller, double %x)
  ret double %0
}

declare double @__enzyme_autodiff(double (double)*, ...)

; CHECK: define internal {{(dso_local )?}}{ double } @diffeadd4(double %x, double %[[differet:.+]])
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call { double } @diffeadd2(double %x, double %[[differet]])
; CHECK-NEXT:   ret { double } %0
; CHECK-NEXT: }

; CHECK: define internal {{(dso_local )?}}{ double } @diffeadd2(double %x, double %[[differet:.+]])
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[result2:.+]] = insertvalue { double } undef, double %[[differet]], 0
; CHECK-NEXT:   ret { double } %[[result2]]
; CHECK-NEXT: }
