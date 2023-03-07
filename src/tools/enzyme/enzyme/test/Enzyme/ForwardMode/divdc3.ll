; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

declare dso_local { double, double } @__divdc3(double, double, double, double)
declare { double, double } @__enzyme_fwddiff(i8*, ...)

define { double, double } @tester(double %x.coerce0, double %x.coerce1, double %y.coerce0, double %y.coerce1) {
entry:
  %call = call { double, double } @__divdc3(double %x.coerce0, double %x.coerce1, double %x.coerce0, double %x.coerce1)
  ret { double, double } %call
}

define dso_local { double, double } @test_derivative(double %x.coerce0, double %x.coerce1, double %y.coerce0, double %y.coerce1) {
entry:
  %call = call { double, double } (i8*, ...) @__enzyme_fwddiff(i8* bitcast ({ double, double } (double, double, double, double)* @tester to i8*), double %x.coerce0, double %x.coerce1, double 1.000000e+00, double 0.000000e+00, double %y.coerce0, double %y.coerce1, double 1.000000e+00, double 0.000000e+00)
  ret { double, double } %call
}


; CHECK: define internal { double, double } @fwddiffetester(double %x.coerce0, double %"x.coerce0'", double %x.coerce1, double %"x.coerce1'", double %y.coerce0, double %"y.coerce0'", double %y.coerce1, double %"y.coerce1'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call { double, double } @__muldc3(double %"x.coerce0'", double %"x.coerce1'", double %x.coerce0, double %x.coerce1)
; CHECK-NEXT:   %1 = call { double, double } @__muldc3(double %x.coerce0, double %x.coerce1, double %"x.coerce0'", double %"x.coerce1'")
; CHECK-NEXT:   %2 = call { double, double } @__muldc3(double %x.coerce0, double %x.coerce1, double %x.coerce0, double %x.coerce1)
; CHECK-DAG:    %[[a3:.+]] = extractvalue { double, double } %0, 0
; CHECK-DAG:    %[[a4:.+]] = extractvalue { double, double } %1, 0
; CHECK-DAG:    %5 = fsub fast double %[[a3]], %[[a4]]
; CHECK-DAG:    %[[a6:.+]] = extractvalue { double, double } %0, 1
; CHECK-DAG:    %[[a7:.+]] = extractvalue { double, double } %1, 1
; CHECK-DAG:    %8 = fsub fast double %[[a6]], %[[a7]]
; CHECK-DAG:    %[[a9:.+]] = extractvalue { double, double } %2, 0
; CHECK-DAG:    %[[a10:.+]] = extractvalue { double, double } %2, 1
; CHECK-DAG:    %11 = call { double, double } @__divdc3(double %5, double %8, double %[[a9]], double %[[a10]])
; CHECK-NEXT:   ret { double, double } %11
; CHECK-NEXT: }
