; RUN: if [ %llvmver -ge 12 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -simplifycfg -early-cse -S | FileCheck %s ; fi

; #include <stdio.h>
; #include <array>

; using namespace std;

; extern int enzyme_width;

; struct Gradients {
;     array<double,3> dx1, dx2, dx3;
; };

; extern Gradients __enzyme_fwddiff(void*, ...);

; array<double,3> square(double x) {
;     return {x * x, x * x * x, x};
; }
; Gradients dsquare(double x) {
;     // This returns the derivative of square or 2 * x
;     return __enzyme_fwddiff((void*)square, enzyme_width, 3, x, 1.0, 2.0, 3.0);
; }
; int main() {
;     printf("%f \n", dsquare(3).dx1[0]);
; }


%"struct.std::array" = type { [3 x double] }
%struct.Gradients = type { %"struct.std::array", %"struct.std::array", %"struct.std::array" }

$_ZNSt5arrayIdLm3EEixEm = comdat any

$_ZNSt14__array_traitsIdLm3EE6_S_refERA3_Kdm = comdat any

@enzyme_width = external dso_local local_unnamed_addr global i32, align 4
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

define dso_local void @_Z7dsquared(%struct.Gradients* noalias sret(%struct.Gradients) align 8 %agg.result, double %x) local_unnamed_addr #1 {
entry:
  %0 = load i32, i32* @enzyme_width, align 4
  call void (%struct.Gradients*, i8*, ...) @_Z16__enzyme_fwddiffPvz(%struct.Gradients* sret(%struct.Gradients) align 8 %agg.result, i8* bitcast (void (%"struct.std::array"*, double)* @_Z6squared to i8*), i32 %0, i32 3, double %x, double 1.000000e+00, double 2.000000e+00, double 3.000000e+00)
  ret void
}

declare dso_local void @_Z16__enzyme_fwddiffPvz(%struct.Gradients* sret(%struct.Gradients) align 8, i8*, ...) local_unnamed_addr #2

define dso_local i32 @main() local_unnamed_addr #3 {
entry:
  %ref.tmp = alloca %struct.Gradients, align 8
  %0 = bitcast %struct.Gradients* %ref.tmp to i8*
  call void @llvm.lifetime.start.p0i8(i64 72, i8* nonnull %0) #7
  call void @_Z7dsquared(%struct.Gradients* nonnull sret(%struct.Gradients) align 8 %ref.tmp, double 3.000000e+00)
  %dx1 = getelementptr inbounds %struct.Gradients, %struct.Gradients* %ref.tmp, i64 0, i32 0
  %call = call nonnull align 8 dereferenceable(8) double* @_ZNSt5arrayIdLm3EEixEm(%"struct.std::array"* nonnull dereferenceable(24) %dx1, i64 0) #7
  %1 = load double, double* %call, align 8
  %call1 = call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([5 x i8], [5 x i8]* @.str, i64 0, i64 0), double %1)
  call void @llvm.lifetime.end.p0i8(i64 72, i8* nonnull %0) #7
  ret i32 0
}

declare dso_local noundef i32 @printf(i8* nocapture noundef readonly, ...) local_unnamed_addr #4

declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #5

define linkonce_odr dso_local nonnull align 8 dereferenceable(8) double* @_ZNSt5arrayIdLm3EEixEm(%"struct.std::array"* nonnull dereferenceable(24) %this, i64 %__n) local_unnamed_addr #6 comdat align 2 {
entry:
  %_M_elems = getelementptr inbounds %"struct.std::array", %"struct.std::array"* %this, i64 0, i32 0
  %call = call nonnull align 8 dereferenceable(8) double* @_ZNSt14__array_traitsIdLm3EE6_S_refERA3_Kdm([3 x double]* nonnull align 8 dereferenceable(24) %_M_elems, i64 %__n) #7
  ret double* %call
}

declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #5

define linkonce_odr dso_local nonnull align 8 dereferenceable(8) double* @_ZNSt14__array_traitsIdLm3EE6_S_refERA3_Kdm([3 x double]* nonnull align 8 dereferenceable(24) %__t, i64 %__n) local_unnamed_addr #6 comdat align 2 {
entry:
  %arrayidx = getelementptr inbounds [3 x double], [3 x double]* %__t, i64 0, i64 %__n
  ret double* %arrayidx
}

attributes #0 = { nofree norecurse nounwind uwtable willreturn writeonly mustprogress "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { uwtable mustprogress "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { norecurse uwtable mustprogress "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nofree nounwind "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { argmemonly nofree nosync nounwind willreturn }
attributes #6 = { nounwind uwtable willreturn mustprogress "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #7 = { nounwind }



; CHECK: define internal void @fwddiffe3_Z6squared(%"struct.std::array"* noalias nocapture align 8 %agg.result, [3 x %"struct.std::array"*] %"agg.result'", double %x, [3 x double] %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = extractvalue [3 x %"struct.std::array"*] %"agg.result'", 0
; CHECK-NEXT:   %"arrayinit.begin'ipg" = getelementptr inbounds %"struct.std::array", %"struct.std::array"* %0, i64 0, i32 0, i64 0
; CHECK-NEXT:   %1 = insertvalue [3 x double*] undef, double* %"arrayinit.begin'ipg", 0
; CHECK-NEXT:   %2 = extractvalue [3 x %"struct.std::array"*] %"agg.result'", 1
; CHECK-NEXT:   %"arrayinit.begin'ipg1" = getelementptr inbounds %"struct.std::array", %"struct.std::array"* %2, i64 0, i32 0, i64 0
; CHECK-NEXT:   %3 = insertvalue [3 x double*] %1, double* %"arrayinit.begin'ipg1", 1
; CHECK-NEXT:   %4 = extractvalue [3 x %"struct.std::array"*] %"agg.result'", 2
; CHECK-NEXT:   %"arrayinit.begin'ipg2" = getelementptr inbounds %"struct.std::array", %"struct.std::array"* %4, i64 0, i32 0, i64 0
; CHECK-NEXT:   %5 = insertvalue [3 x double*] %3, double* %"arrayinit.begin'ipg2", 2
; CHECK-NEXT:   %arrayinit.begin = getelementptr inbounds %"struct.std::array", %"struct.std::array"* %agg.result, i64 0, i32 0, i64 0
; CHECK-NEXT:   %mul = fmul double %x, %x
; CHECK-NEXT:   %6 = extractvalue [3 x double] %"x'", 0
; CHECK-NEXT:   %7 = fmul fast double %6, %x
; CHECK-NEXT:   %8 = fadd fast double %7, %7
; CHECK-NEXT:   %9 = insertvalue [3 x double] undef, double %8, 0
; CHECK-NEXT:   %10 = extractvalue [3 x double] %"x'", 1
; CHECK-NEXT:   %11 = fmul fast double %10, %x
; CHECK-NEXT:   %12 = fadd fast double %11, %11
; CHECK-NEXT:   %13 = insertvalue [3 x double] %9, double %12, 1
; CHECK-NEXT:   %14 = extractvalue [3 x double] %"x'", 2
; CHECK-NEXT:   %15 = fmul fast double %14, %x
; CHECK-NEXT:   %16 = fadd fast double %15, %15
; CHECK-NEXT:   %17 = insertvalue [3 x double] %13, double %16, 2
; CHECK-NEXT:   store double %mul, double* %arrayinit.begin, align 8
; CHECK-NEXT:   store double %8, double* %"arrayinit.begin'ipg", align 8
; CHECK-NEXT:   store double %12, double* %"arrayinit.begin'ipg1", align 8
; CHECK-NEXT:   store double %16, double* %"arrayinit.begin'ipg2", align 8
; CHECK-NEXT:   %"arrayinit.element'ipg" = getelementptr inbounds %"struct.std::array", %"struct.std::array"* %0, i64 0, i32 0, i64 1
; CHECK-NEXT:   %18 = insertvalue [3 x double*] undef, double* %"arrayinit.element'ipg", 0
; CHECK-NEXT:   %"arrayinit.element'ipg3" = getelementptr inbounds %"struct.std::array", %"struct.std::array"* %2, i64 0, i32 0, i64 1
; CHECK-NEXT:   %19 = insertvalue [3 x double*] %18, double* %"arrayinit.element'ipg3", 1
; CHECK-NEXT:   %"arrayinit.element'ipg4" = getelementptr inbounds %"struct.std::array", %"struct.std::array"* %4, i64 0, i32 0, i64 1
; CHECK-NEXT:   %20 = insertvalue [3 x double*] %19, double* %"arrayinit.element'ipg4", 2
; CHECK-NEXT:   %arrayinit.element = getelementptr inbounds %"struct.std::array", %"struct.std::array"* %agg.result, i64 0, i32 0, i64 1
; CHECK-NEXT:   %mul2 = fmul double %mul, %x
; CHECK-NEXT:   %21 = fmul fast double %8, %x
; CHECK-NEXT:   %22 = fmul fast double %6, %mul
; CHECK-NEXT:   %23 = fadd fast double %21, %22
; CHECK-NEXT:   %24 = insertvalue [3 x double] undef, double %23, 0
; CHECK-NEXT:   %25 = fmul fast double %12, %x
; CHECK-NEXT:   %26 = fmul fast double %10, %mul
; CHECK-NEXT:   %27 = fadd fast double %25, %26
; CHECK-NEXT:   %28 = insertvalue [3 x double] %24, double %27, 1
; CHECK-NEXT:   %29 = fmul fast double %16, %x
; CHECK-NEXT:   %30 = fmul fast double %14, %mul
; CHECK-NEXT:   %31 = fadd fast double %29, %30
; CHECK-NEXT:   %32 = insertvalue [3 x double] %28, double %31, 2
; CHECK-NEXT:   store double %mul2, double* %arrayinit.element, align 8
; CHECK-NEXT:   store double %23, double* %"arrayinit.element'ipg", align 8
; CHECK-NEXT:   store double %27, double* %"arrayinit.element'ipg3", align 8
; CHECK-NEXT:   store double %31, double* %"arrayinit.element'ipg4", align 8
; CHECK-NEXT:   %"arrayinit.element3'ipg" = getelementptr inbounds %"struct.std::array", %"struct.std::array"* %0, i64 0, i32 0, i64 2
; CHECK-NEXT:   %33 = insertvalue [3 x double*] undef, double* %"arrayinit.element3'ipg", 0
; CHECK-NEXT:   %"arrayinit.element3'ipg5" = getelementptr inbounds %"struct.std::array", %"struct.std::array"* %2, i64 0, i32 0, i64 2
; CHECK-NEXT:   %34 = insertvalue [3 x double*] %33, double* %"arrayinit.element3'ipg5", 1
; CHECK-NEXT:   %"arrayinit.element3'ipg6" = getelementptr inbounds %"struct.std::array", %"struct.std::array"* %4, i64 0, i32 0, i64 2
; CHECK-NEXT:   %35 = insertvalue [3 x double*] %34, double* %"arrayinit.element3'ipg6", 2
; CHECK-NEXT:   %arrayinit.element3 = getelementptr inbounds %"struct.std::array", %"struct.std::array"* %agg.result, i64 0, i32 0, i64 2
; CHECK-NEXT:   store double %x, double* %arrayinit.element3, align 8
; CHECK-NEXT:   store double %6, double* %"arrayinit.element3'ipg", align 8
; CHECK-NEXT:   store double %10, double* %"arrayinit.element3'ipg5", align 8
; CHECK-NEXT:   store double %14, double* %"arrayinit.element3'ipg6", align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }