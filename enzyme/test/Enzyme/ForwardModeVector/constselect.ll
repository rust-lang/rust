; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -instsimplify -simplifycfg -S | FileCheck %s

%struct.Gradients = type { double, double }

; Function Attrs: nounwind
declare %struct.Gradients @__enzyme_fwddiff(double (double)*, ...)

; Function Attrs: norecurse nounwind readnone uwtable
define double @fun2(double %x) {
entry:
  %cmp.inv = fcmp oge double %x, 0.000000e+00
  %.x = select i1 %cmp.inv, double %x, double 0.000000e+00
  ret double %.x
}

; Function Attrs: nounwind uwtable
define i32 @main() {
entry:
  %call3.4 = tail call %struct.Gradients (double (double)*, ...) @__enzyme_fwddiff(double (double)* nonnull @fun2, metadata !"enzyme_width", i64 2, double 2.0, double 0.000000e+00, double 1.000000e+00)
  ret i32 0
}


; CHECK: define internal [2 x double] @fwddiffe2fun2(double %x, [2 x double] %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %cmp.inv = fcmp oge double %x, 0.000000e+00
; CHECK-NEXT:   %0 = select {{(fast )?}}i1 %cmp.inv, [2 x double] %"x'", [2 x double] zeroinitializer
; CHECK-NEXT:   ret [2 x double] %0
; CHECK-NEXT: }