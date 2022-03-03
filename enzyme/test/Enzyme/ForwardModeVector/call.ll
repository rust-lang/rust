; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

%struct.Gradients = type { double, double, double }

define double @_Z4add2d(double %x) {
entry:
  %add = fadd double %x, 2.000000e+00
  ret double %add
}

define double @_Z4add4d(double %x) {
entry:
  %call = call double @_Z4add2d(double %x)
  %add = fadd double %call, 2.000000e+00
  ret double %add
}

define %struct.Gradients @_Z5dadd4d(double %x) {
entry:
  %call = call %struct.Gradients (double (double)*, ...) @_Z22__enzyme_fwddiffPFddEz(double (double)* nonnull @_Z4add4d, metadata !"enzyme_width", i64 3, double %x, double 1.000000e+00, double 2.000000e+00, double 3.000000e+00)
  ret %struct.Gradients %call
}

declare %struct.Gradients @_Z22__enzyme_fwddiffPFddEz(double (double)*, ...)


; CHECK: define internal [3 x double] @fwddiffe3_Z4add4d(double %x, [3 x double] %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call {{(fast )?}}[3 x double] @fwddiffe3_Z4add2d(double %x, [3 x double] %"x'")
; CHECK-NEXT:   ret [3 x double] %0
; CHECK-NEXT: }

; CHECK: define internal [3 x double] @fwddiffe3_Z4add2d(double %x, [3 x double] %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret [3 x double] %"x'"
; CHECK-NEXT: }