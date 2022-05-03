; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

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

; CHECK: define internal { double } @diffecaller(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call { double } @diffesub(double %x, i64 0, double %differeturn)
; CHECK-NEXT:   ret { double } %0
; CHECK-NEXT: }

; CHECK: define internal { double } @diffesub(double %x, i64 %y, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %malloccall = tail call i8* @malloc(i64 8)
; CHECK-NEXT:   %bc = bitcast i8* %malloccall to i64*
; CHECK-NEXT:   store i64 %y, i64* %bc, align 8
; CHECK-NEXT:   %0 = insertvalue { double } undef, double %differeturn, 0
; CHECK-NEXT:   ret { double } %0
; CHECK-NEXT: }
