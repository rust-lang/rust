; RUN: if [ %llvmver -ge 12 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -simplifycfg -early-cse -S | FileCheck %s ; fi

;
; #include <array>
; 
; using namespace std;
;
; struct Diffe {
;     double ddx;
;     double ddy;
;     double ddz;
; };
; 
; extern Diffe __enzyme_autodiff(void*, ...);
; 
; double test(double x, double y, double z) {
;     return x * y * z;
; }
; Diffe dtest(double x, double y, double z) {
;     return __enzyme_autodiff((void*)test, x, y, z);
; }


%struct.Diffe = type { double, double, double }

define dso_local double @_Z4testddd(double %x, double %y, double %z) #0 {
entry:
  %mul = fmul double %x, %y
  %mul1 = fmul double %mul, %z
  ret double %mul1
}

define dso_local void @_Z5dtestddd(%struct.Diffe* noalias sret(%struct.Diffe) align 8 %agg.result, double %x, double %y, double %z) local_unnamed_addr #1 {
entry:
  tail call void (%struct.Diffe*, i8*, ...) @_Z17__enzyme_autodiffPvz(%struct.Diffe* sret(%struct.Diffe) align 8 %agg.result, i8* bitcast (double (double, double, double)* @_Z4testddd to i8*), double %x, double %y, double %z)
  ret void
}

declare dso_local void @_Z17__enzyme_autodiffPvz(%struct.Diffe* sret(%struct.Diffe) align 8, i8*, ...) local_unnamed_addr #2

attributes #0 = { norecurse nounwind readnone uwtable willreturn mustprogress "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { uwtable mustprogress "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }




; CHECK: define {{(dso_local )?}}void @_Z5dtestddd(%struct.Diffe* noalias sret(%struct.Diffe) align 8 %agg.result, double %x, double %y, double %z)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call { double, double, double } @diffe_Z4testddd(double %x, double %y, double %z, double 1.000000e+00)
; CHECK-DAG:    [[REG1:%[0-9]+]] = getelementptr inbounds %struct.Diffe, %struct.Diffe* %agg.result, i32 0, i32 0
; CHECK-DAG:    [[REG2:%[0-9]+]] = extractvalue { double, double, double } %0, 0
; CHECK-NEXT:   store double [[REG2]], double* [[REG1]], align 8
; CHECK-DAG:    [[REG3:%[0-9]+]] = getelementptr inbounds %struct.Diffe, %struct.Diffe* %agg.result, i32 0, i32 1
; CHECK-DAG:    [[REG4:%[0-9]+]] = extractvalue { double, double, double } %0, 1
; CHECK-NEXT:   store double [[REG4]], double* [[REG3]], align 8
; CHECK-DAG:    [[REG5:%[0-9]+]] = getelementptr inbounds %struct.Diffe, %struct.Diffe* %agg.result, i32 0, i32 2
; CHECK-DAG:    [[REG6:%[0-9]+]] = extractvalue { double, double, double } %0, 2
; CHECK-NEXT:   store double [[REG6]], double* [[REG5]], align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }


; CHECK:  define internal { double, double, double } @diffe_Z4testddd(double %x, double %y, double %z, double %differeturn)
; CHECK-NEXT: entry:  
; CHECK-NEXT:   %mul = fmul double %x, %y
; CHECK-NEXT:   %m0diffemul = fmul fast double %differeturn, %z
; CHECK-NEXT:   %m1diffez = fmul fast double %differeturn, %mul
; CHECK-NEXT:   %m0diffex = fmul fast double %m0diffemul, %y
; CHECK-NEXT:   %m1diffey = fmul fast double %m0diffemul, %x
; CHECK-NEXT:   %0 = insertvalue { double, double, double } undef, double %m0diffex, 0
; CHECK-NEXT:   %1 = insertvalue { double, double, double } %0, double %m1diffey, 1
; CHECK-NEXT:   %2 = insertvalue { double, double, double } %1, double %m1diffez, 2
; CHECK-NEXT:   ret { double, double, double } %2
; CHECK-NEXT: }