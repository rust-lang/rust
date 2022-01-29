; RUN: %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=d -o /dev/null | FileCheck %s

; ModuleID = 'inp.ll'
source_filename = "/mnt/Data/git/Enzyme/enzyme/test/Integration/integrateexp.cpp"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define dso_local double @m(double %div) {
entry:
  %call = call fast double @__enzyme_autodiff(i8* bitcast (double (double)* @_Z6foobard to i8*), double %div) #3
  ret double %call
}

declare dso_local double @__enzyme_autodiff(i8*, double) local_unnamed_addr #0

; Function Attrs: nounwind uwtable
define dso_local double @_Z6foobard(double %t) #1 {
entry:
  %alloc = alloca i64
  store i64 ptrtoint (i1 (double*)* @d to i64), i64* %alloc
  ret double 1.000000e-02
}

; Function Attrs: norecurse nounwind uwtable
define linkonce_odr dso_local zeroext i1 @d(double* %x) #2 align 2 {
entry:
  %call = tail call zeroext i1 @g(double* %x)
  ret i1 %call
}

; Function Attrs: norecurse nounwind uwtable
define linkonce_odr dso_local zeroext i1 @g(double* %x) local_unnamed_addr #2 {
entry:
  ret i1 false
}

attributes #0 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #2 = { norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #3 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}

; CHECK: d - {[-1]:Integer} |{[-1]:Pointer, [-1,-1]:Float@double}:{} 
; CHECK-NEXT: double* %x: {[-1]:Pointer, [-1,-1]:Float@double}
; CHECK-NEXT: entry
; CHECK-NEXT:   %call = tail call zeroext i1 @g(double* %x): {[-1]:Integer}
; CHECK-NEXT:   ret i1 %call: {}

; CHECK-NOT: g
