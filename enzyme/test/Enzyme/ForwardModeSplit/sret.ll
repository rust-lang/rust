; RUN: if [ %llvmver -lt 12 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -simplifycfg -early-cse -S | FileCheck %s ; fi


; #include <stdio.h>
; #include <array>

; using namespace std;

; extern array<double,3> __enzyme_fwdsplit(void*, ...);

; array<double,3> square(double x) {
;     return {x * x, x * x * x, x};
; }
; array<double,3> dsquare(double x) {
;     // This returns the derivative of square or 2 * x
;     return __enzyme_fwdsplit((void*)square, x, 1.0);
; }
; int main() {
;     printf("%f \n", dsquare(3)[0]);
; }


%"struct.std::array" = type { [3 x double] }

@.str = private unnamed_addr constant [5 x i8] c"%f \0A\00", align 1

define dso_local void @_Z6squared(%"struct.std::array"* noalias nocapture sret %agg.result, double %x) #0 {
entry:
  %arrayinit.begin = getelementptr inbounds %"struct.std::array", %"struct.std::array"* %agg.result, i64 0, i32 0, i64 0
  %mul = fmul double %x, %x
  store double %mul, double* %arrayinit.begin, align 8
  %arrayinit.element = getelementptr inbounds %"struct.std::array", %"struct.std::array"* %agg.result, i64 0, i32 0, i64 1
  %mul2 = fmul double %mul, %x
  store double %mul2, double* %arrayinit.element, align 8
  %arrayinit.element3 = getelementptr inbounds %"struct.std::array", %"struct.std::array"* %agg.result, i64 0, i32 0, i64 2
  store double %x, double* %arrayinit.element3, align 8
  ret void
}

define dso_local void @_Z7dsquared(%"struct.std::array"* noalias sret %agg.result, double %x) local_unnamed_addr #1 {
entry:
  tail call void (%"struct.std::array"*, i8*, ...) @_Z16__enzyme_fwdsplitPvz(%"struct.std::array"* sret %agg.result, i8* bitcast (void (%"struct.std::array"*, double)* @_Z6squared to i8*), double %x, double 1.000000e+00, i8* null)
  ret void
}

declare dso_local void @_Z16__enzyme_fwdsplitPvz(%"struct.std::array"* sret, i8*, ...) local_unnamed_addr #2

define dso_local i32 @main() local_unnamed_addr #3 {
entry:
  %ref.tmp = alloca %"struct.std::array", align 8
  %0 = bitcast %"struct.std::array"* %ref.tmp to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %0) #6
  call void (%"struct.std::array"*, i8*, ...) @_Z16__enzyme_fwdsplitPvz(%"struct.std::array"* nonnull sret %ref.tmp, i8* bitcast (void (%"struct.std::array"*, double)* @_Z6squared to i8*), double 3.000000e+00, double 1.000000e+00, i8* null)
  %arrayidx.i.i = getelementptr inbounds %"struct.std::array", %"struct.std::array"* %ref.tmp, i64 0, i32 0, i64 0
  %1 = load double, double* %arrayidx.i.i, align 8
  %call1 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str, i64 0, i64 0), double %1)
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %0) #6
  ret i32 0
}

declare dso_local i32 @printf(i8* nocapture readonly, ...) local_unnamed_addr #4

declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #5

declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #5

attributes #0 = { norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { norecurse uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { argmemonly nounwind }
attributes #6 = { nounwind }


; CHECK: define dso_local void @_Z7dsquared(%"struct.std::array"* noalias sret %agg.result, double %x)
; CHECK-NEXT: entry:  
; CHECK-NEXT:   %0 = alloca %"struct.std::array"
; CHECK-NEXT:   call void @fwddiffe_Z6squared(%"struct.std::array"* %0, %"struct.std::array"* %agg.result, double %x, double 1.000000e+00, i8* null)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }


; CHECK: define internal void @fwddiffe_Z6squared(%"struct.std::array"* noalias nocapture "enzyme_sret" %agg.result, %"struct.std::array"* nocapture "enzyme_sret" %"agg.result'", double %x, double %"x'", i8* %tapeArg)
; CHECK-NEXT: entry:  
; CHECK-NEXT:   %"arrayinit.begin'ipg" = getelementptr inbounds %"struct.std::array", %"struct.std::array"* %"agg.result'", i64 0, i32 0, i64 0
; CHECK-NEXT:   %mul = fmul double %x, %x
; CHECK-NEXT:   %0 = fmul fast double %"x'", %x
; CHECK-NEXT:   %1 = fadd fast double %0, %0
; CHECK-NEXT:   store double %1, double* %"arrayinit.begin'ipg", align 8
; CHECK-NEXT:   %"arrayinit.element'ipg" = getelementptr inbounds %"struct.std::array", %"struct.std::array"* %"agg.result'", i64 0, i32 0, i64 1
; CHECK-NEXT:   %2 = fmul fast double %1, %x
; CHECK-NEXT:   %3 = fmul fast double %"x'", %mul
; CHECK-NEXT:   %4 = fadd fast double %2, %3
; CHECK-NEXT:   store double %4, double* %"arrayinit.element'ipg", align 8
; CHECK-NEXT:   %"arrayinit.element3'ipg" = getelementptr inbounds %"struct.std::array", %"struct.std::array"* %"agg.result'", i64 0, i32 0, i64 2
; CHECK-NEXT:   store double %"x'", double* %"arrayinit.element3'ipg", align 8
; CHECK-NEXT:  ret void
; CHECK-NEXT: }
