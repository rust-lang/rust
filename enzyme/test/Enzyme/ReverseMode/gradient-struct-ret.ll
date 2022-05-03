; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -early-cse -instsimplify -simplifycfg -S | FileCheck %s

%struct.Gradients = type { double, double }

define dso_local double @muldd(double %x, double %y) #0 {
entry:
  %mul = fmul double %x, %y
  ret double %mul
}

define dso_local %struct.Gradients @dmuldd(double %x, double %y) local_unnamed_addr #1 {
entry:
  %call = call %struct.Gradients (i8*, ...) @_Z17__enzyme_autodiffPvz(i8* bitcast (double (double, double)* @muldd to i8*), double %x, double %y)
  ret %struct.Gradients %call
}

declare dso_local %struct.Gradients @_Z17__enzyme_autodiffPvz(i8*, ...) local_unnamed_addr #2

attributes #0 = { norecurse nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="non-leaf" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+neon" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="non-leaf" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+neon" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="non-leaf" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+neon" "unsafe-fp-math"="false" "use-soft-float"="false" }

; CHECK: define internal {{(dso_local )?}}{ double, double } @diffemuldd(double %x, double %y, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %m0diffex = fmul fast double %differeturn, %y
; CHECK-NEXT:   %m1diffey = fmul fast double %differeturn, %x
; CHECK-NEXT:   %0 = insertvalue { double, double } undef, double %m0diffex, 0
; CHECK-NEXT:   %1 = insertvalue { double, double } %0, double %m1diffey, 1
; CHECK-NEXT:   ret { double, double } %1
; CHECK-NEXT: }