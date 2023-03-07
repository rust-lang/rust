; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

define { double, double } @test(double %x.coerce0, double %x.coerce1, double %y.coerce0, double %y.coerce1) {
entry:
  %call = call { double, double } @__divdc3(double %x.coerce0, double %x.coerce1, double %y.coerce0, double %y.coerce1)
  ret { double, double } %call
}

declare { double, double } @__divdc3(double, double, double, double)

define { double, double, double, double} @dtest(double %x.coerce0, double %x.coerce1, double %y.coerce0, double %y.coerce1) {
entry:
  %call = call { double, double, double, double } (i8*, ...) @__enzyme_autodiff(i8* bitcast ({ double, double } (double, double, double, double)* @test to i8*), double %x.coerce0, double %x.coerce1, double %y.coerce0, double %y.coerce1)
  ret { double, double, double, double} %call
}

declare { double, double, double, double } @__enzyme_autodiff(i8*, ...)


; CHECK: define internal { double, double, double, double } @diffetest(double %x.coerce0, double %x.coerce1, double %y.coerce0, double %y.coerce1, { double, double } %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %call = call { double, double } @__divdc3(double %x.coerce0, double %x.coerce1, double %y.coerce0, double %y.coerce1)
; CHECK-DAG:    %[[a0:.+]] = extractvalue { double, double } %differeturn, 0
; CHECK-DAG:    %[[a1:.+]] = extractvalue { double, double } %differeturn, 1
; CHECK-DAG:    %2 = call { double, double } @__divdc3(double %[[a0]], double %[[a1]], double %y.coerce0, double %y.coerce1)
; CHECK-DAG:    %3 = call { double, double } @__divdc3(double %[[a0]], double %[[a1]], double %x.coerce1, double %y.coerce0)
; CHECK-DAG:    %[[a4:.+]] = extractvalue { double, double } %call, 0
; CHECK-DAG:    %5 = {{(fneg fast double)|(fsub fast double \-0.000000e\+00,)}} %[[a4]]
; CHECK-DAG:    %[[a6:.+]] = extractvalue { double, double } %call, 1
; CHECK-DAG:    %7 = {{(fneg fast double)|(fsub fast double \-0.000000e\+00,)}} %[[a6]]
; CHECK-DAG:    %[[a8:.+]] = extractvalue { double, double } %3, 0
; CHECK-DAG:    %[[a9:.+]] = extractvalue { double, double } %3, 1
; CHECK-DAG:    %10 = call { double, double } @__muldc3(double %5, double %7, double %[[a8]], double %[[a9]])
; CHECK-DAG:    %[[a11:.+]] = extractvalue { double, double } %2, 0
; CHECK-DAG:    %[[a12:.+]] = extractvalue { double, double } %2, 1
; CHECK-DAG:    %[[a13:.+]] = extractvalue { double, double } %10, 0
; CHECK-DAG:    %[[a14:.+]] = extractvalue { double, double } %10, 1
; CHECK-DAG:    %15 = insertvalue { double, double, double, double } undef, double %[[a11]], 0
; CHECK-DAG:    %16 = insertvalue { double, double, double, double } %15, double %[[a12]], 1
; CHECK-DAG:    %17 = insertvalue { double, double, double, double } %16, double %[[a13]], 2
; CHECK-DAG:    %18 = insertvalue { double, double, double, double } %17, double %[[a14]], 3
; CHECK-NEXT:   ret { double, double, double, double } %18
; CHECK-NEXT: }