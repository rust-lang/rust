; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

declare { double, double } @__muldc3(double, double, double, double)
declare { double, double } @__enzyme_autodiff(i8*, ...)

define { double, double } @square(double %x.coerce0, double %x.coerce1) {
entry:
  %call = call { double, double } @__muldc3(double %x.coerce0, double %x.coerce1, double %x.coerce0, double %x.coerce1) #2
  ret { double, double } %call
}

define { double, double } @dsquare(double %x.coerce0, double %x.coerce1) {
entry:
  %call = call { double, double } (i8*, ...) @__enzyme_autodiff(i8* bitcast ({ double, double } (double, double)* @square to i8*), double %x.coerce0, double %x.coerce1) #2
  ret { double, double } %call
}


; CHECK: define internal { double, double } @diffesquare(double %x.coerce0, double %x.coerce1, { double, double } %differeturn)
; CHECK-NEXT: entry:
; CHECK-DAG:    %[[a0:.+]] = extractvalue { double, double } %differeturn, 0
; CHECK-DAG:    %[[a1:.+]] = extractvalue { double, double } %differeturn, 1
; CHECK-DAG:    %2 = call { double, double } @__muldc3(double %[[a0]], double %[[a1]], double %x.coerce0, double %x.coerce1)
; CHECK-DAG:    %3 = call { double, double } @__muldc3(double %x.coerce0, double %x.coerce1, double %[[a0]], double %[[a1]])
; CHECK-DAG:    %[[a4:.+]] = extractvalue { double, double } %2, 0
; CHECK-DAG:    %[[a5:.+]] = extractvalue { double, double } %2, 1
; CHECK-DAG:    %[[a6:.+]] = extractvalue { double, double } %3, 0
; CHECK-DAG:    %7 = fadd fast double %[[a4]], %[[a6]]
; CHECK-DAG:    %[[a8:.+]] = extractvalue { double, double } %3, 1
; CHECK-DAG:    %9 = fadd fast double %[[a5]], %[[a8]]
; CHECK-NEXT:   %10 = insertvalue { double, double } undef, double %7, 0
; CHECK-NEXT:   %11 = insertvalue { double, double } %10, double %9, 1
; CHECK-NEXT:   ret { double, double } %11
; CHECK-NEXT: }