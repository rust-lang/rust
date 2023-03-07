; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

declare { double, double } @__muldc3(double, double, double, double)
declare { double, double } @__enzyme_fwddiff(i8*, ...)

define { double, double } @square(double %x.coerce0, double %x.coerce1) {
entry:
  %call = call { double, double } @__muldc3(double %x.coerce0, double %x.coerce1, double %x.coerce0, double %x.coerce1)
  ret { double, double } %call
}

define { double, double } @dsquare(double %x.coerce0, double %x.coerce1) {
entry:
  %call = call { double, double } (i8*, ...) @__enzyme_fwddiff(i8* bitcast ({ double, double } (double, double)* @square to i8*), double %x.coerce0, double %x.coerce1, double 1.000000e+00, double 0.000000e+00)
  ret { double, double } %call
}


; CHECK: define internal { double, double } @fwddiffesquare(double %x.coerce0, double %"x.coerce0'", double %x.coerce1, double %"x.coerce1'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call { double, double } @__muldc3(double %"x.coerce0'", double %"x.coerce1'", double %x.coerce0, double %x.coerce1)
; CHECK-NEXT:   %1 = call { double, double } @__muldc3(double %x.coerce0, double %x.coerce1, double %"x.coerce0'", double %"x.coerce1'")
; CHECK-DAG:    %[[a2:.+]] = extractvalue { double, double } %0, 0
; CHECK-DAG:    %[[a3:.+]] = extractvalue { double, double } %1, 0
; CHECK-DAG:    %4 = fadd fast double %[[a2]], %[[a3]]
; CHECK-DAG:    %[[a5:.+]] = extractvalue { double, double } %0, 1
; CHECK-DAG:    %[[a6:.+]] = extractvalue { double, double } %1, 1
; CHECK-DAG:    %7 = fadd fast double %[[a5]], %[[a6]]
; CHECK-DAG:    %[[a8:.+]] = insertvalue { double, double } undef, double %4, 0
; CHECK-DAG:    %[[a9:.+]] = insertvalue { double, double } %[[a8]], double %7, 1
; CHECK-DAG:    ret { double, double } %[[a9]]
; CHECK-NEXT: }
