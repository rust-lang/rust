; RUN: if [ %llvmver -ge 12 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -simplifycfg -early-cse -S | FileCheck %s ; fi


; #include <stdio.h>
; #include <array>

; using namespace std;

; extern array<double,3> __enzyme_fwddiff(void*, ...);

; array<double,3> square(double x) {
;     return {x * x, x * x * x, x};
; }
; array<double,3> dsquare(double x) {
;     // This returns the derivative of square or 2 * x
;     return __enzyme_fwddiff((void*)square, x, 1.0);
; }
; int main() {
;     printf("%f \n", dsquare(3)[0]);
; }


%"struct.std::array" = type { [3 x double] }

@.str = private unnamed_addr constant [5 x i8] c"%f \0A\00", align 1

define dso_local void @_Z6squared(%"struct.std::array"* noalias nocapture sret(%"struct.std::array") align 8 %agg.result, double %x) #0 {
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

define dso_local void @_Z7dsquared(%"struct.std::array"* noalias sret(%"struct.std::array") align 8 %agg.result, double %x) local_unnamed_addr #1 {
entry:
  tail call void (%"struct.std::array"*, i8*, ...) @_Z16__enzyme_fwddiffPvz(%"struct.std::array"* sret(%"struct.std::array") align 8 %agg.result, i8* bitcast (void (%"struct.std::array"*, double)* @_Z6squared to i8*), double %x, double 1.000000e+00)
  ret void
}

declare dso_local void @_Z16__enzyme_fwddiffPvz(%"struct.std::array"* sret(%"struct.std::array") align 8, i8*, ...) local_unnamed_addr #2

define dso_local i32 @main() local_unnamed_addr #3 {
entry:
  %ref.tmp = alloca %"struct.std::array", align 8
  %0 = bitcast %"struct.std::array"* %ref.tmp to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %0) #6
  call void (%"struct.std::array"*, i8*, ...) @_Z16__enzyme_fwddiffPvz(%"struct.std::array"* nonnull sret(%"struct.std::array") align 8 %ref.tmp, i8* bitcast (void (%"struct.std::array"*, double)* @_Z6squared to i8*), double 3.000000e+00, double 1.000000e+00)
  %arrayidx.i.i = getelementptr inbounds %"struct.std::array", %"struct.std::array"* %ref.tmp, i64 0, i32 0, i64 0
  %1 = load double, double* %arrayidx.i.i, align 8
  %call1 = call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([5 x i8], [5 x i8]* @.str, i64 0, i64 0), double %1)
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %0) #6
  ret i32 0
}

declare dso_local noundef i32 @printf(i8* nocapture noundef readonly, ...) local_unnamed_addr #4

declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #5

declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #5

attributes #0 = { nofree norecurse nounwind uwtable willreturn writeonly mustprogress "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { uwtable mustprogress "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { norecurse uwtable mustprogress "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nofree nounwind "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { argmemonly nofree nosync nounwind willreturn }
attributes #6 = { nounwind }


; CHECK: define dso_local void @_Z7dsquared(%"struct.std::array"* noalias sret(%"struct.std::array") align 8 %agg.result, double %x)
; CHECK-NEXT: entry:  
; CHECK-NEXT:   %0 = alloca %"struct.std::array"
; CHECK-NEXT:   call void @fwddiffe_Z6squared(%"struct.std::array"* %0, %"struct.std::array"* %agg.result, double %x, double 1.000000e+00)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }


; CHECK: define internal void @fwddiffe_Z6squared(%"struct.std::array"* noalias nocapture align 8 "enzyme_sret" %agg.result, %"struct.std::array"* nocapture "enzyme_sret" %"agg.result'", double %x, double %"x'") #0 {
; CHECK-NEXT: entry:  
; CHECK-NEXT:   %"arrayinit.begin'ipg" = getelementptr inbounds %"struct.std::array", %"struct.std::array"* %"agg.result'", i64 0, i32 0, i64 0
; CHECK-NEXT:  %arrayinit.begin = getelementptr inbounds %"struct.std::array", %"struct.std::array"* %agg.result, i64 0, i32 0, i64 0
; CHECK-NEXT:   %mul = fmul double %x, %x
; CHECK-NEXT:   %0 = fmul fast double %"x'", %x
; CHECK-NEXT:   %1 = fadd fast double %0, %0
; CHECK-NEXT:   store double %mul, double* %arrayinit.begin, align 8
; CHECK-NEXT:   store double %1, double* %"arrayinit.begin'ipg", align 8
; CHECK-NEXT:   %"arrayinit.element'ipg" = getelementptr inbounds %"struct.std::array", %"struct.std::array"* %"agg.result'", i64 0, i32 0, i64 1
; CHECK-NEXT:   %arrayinit.element = getelementptr inbounds %"struct.std::array", %"struct.std::array"* %agg.result, i64 0, i32 0, i64 1
; CHECK-NEXT:   %mul2 = fmul double %mul, %x
; CHECK-NEXT:   %2 = fmul fast double %1, %x
; CHECK-NEXT:   %3 = fmul fast double %"x'", %mul
; CHECK-NEXT:   %4 = fadd fast double %2, %3
; CHECK-NEXT:   store double %mul2, double* %arrayinit.element, align 8
; CHECK-NEXT:   store double %4, double* %"arrayinit.element'ipg", align 8
; CHECK-NEXT:   %"arrayinit.element3'ipg" = getelementptr inbounds %"struct.std::array", %"struct.std::array"* %"agg.result'", i64 0, i32 0, i64 2
; CHECK-NEXT:   %arrayinit.element3 = getelementptr inbounds %"struct.std::array", %"struct.std::array"* %agg.result, i64 0, i32 0, i64 2
; CHECK-NEXT:   store double %x, double* %arrayinit.element3, align 8
; CHECK-NEXT:   store double %"x'", double* %"arrayinit.element3'ipg", align 8
; CHECK-NEXT:  ret void
; CHECK-NEXT: }
