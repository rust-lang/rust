; RUN: if [ %llvmver -ge 12 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -simplifycfg -early-cse -S | FileCheck %s ; fi


; #include <array>

; double __enzyme_autodiff(...);

; std::array<double,3> square(double x) {
;     return {x,x*x,x*x*x};
; }

; double dsquare(double x) {
;     std::array<double,3> dx = {1,1,1};
;     return __enzyme_autodiff((void*)square, &dx, x);
; }


%"struct.std::array" = type { [3 x double] }

@__const._Z7dsquared.dx = private unnamed_addr constant %"struct.std::array" { [3 x double] [double 1.000000e+00, double 1.000000e+00, double 1.000000e+00] }, align 8

define dso_local void @_Z6squared(%"struct.std::array"* noalias nocapture sret(%"struct.std::array") align 8 %agg.result, double %x) #0 {
entry:
  %arrayinit.begin = getelementptr inbounds %"struct.std::array", %"struct.std::array"* %agg.result, i64 0, i32 0, i64 0
  store double %x, double* %arrayinit.begin, align 8
  %arrayinit.element = getelementptr inbounds %"struct.std::array", %"struct.std::array"* %agg.result, i64 0, i32 0, i64 1
  %mul = fmul double %x, %x
  store double %mul, double* %arrayinit.element, align 8
  %arrayinit.element1 = getelementptr inbounds %"struct.std::array", %"struct.std::array"* %agg.result, i64 0, i32 0, i64 2
  %mul3 = fmul double %mul, %x
  store double %mul3, double* %arrayinit.element1, align 8
  ret void
}

define dso_local double @_Z7dsquared(double %x) local_unnamed_addr #1 {
entry:
  %dx = alloca %"struct.std::array", align 8
  %0 = bitcast %"struct.std::array"* %dx to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %0) #4
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(24) %0, i8* nonnull align 8 dereferenceable(24) bitcast (%"struct.std::array"* @__const._Z7dsquared.dx to i8*), i64 24, i1 false)
  %call = call double (...) @_Z17__enzyme_autodiffz(i8* bitcast (void (%"struct.std::array"*, double)* @_Z6squared to i8*), %"struct.std::array"* nonnull %dx, double %x)
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %0) #4
  ret double %call
}

declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #2

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg) #2

declare dso_local double @_Z17__enzyme_autodiffz(...) local_unnamed_addr #3

declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #2

attributes #0 = { nofree norecurse nounwind uwtable willreturn writeonly mustprogress "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { uwtable mustprogress "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { argmemonly nofree nosync nounwind willreturn }
attributes #3 = { "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind }


; CHECK: define {{(dso_local )?}}double @_Z7dsquared(double %x)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %dx = alloca %"struct.std::array", align 8
; CHECK-NEXT:   %0 = bitcast %"struct.std::array"* %dx to i8*
; CHECK-NEXT:   call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %0)
; CHECK-NEXT:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(24) %0, i8* nonnull align 8 dereferenceable(24) bitcast (%"struct.std::array"* @__const._Z7dsquared.dx to i8*), i64 24, i1 false)
; CHECK-NEXT:   %1 = alloca %"struct.std::array", align 8
; CHECK-NEXT:   %2 = call { double } @diffe_Z6squared(%"struct.std::array"* %1, %"struct.std::array"* %dx, double %x)
; CHECK-NEXT:   %3 = extractvalue { double } %2, 0
; CHECK-NEXT:   call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %0)
; CHECK-NEXT:   ret double %3
; CHECK-NEXT: }


; CHECK: define internal { double } @diffe_Z6squared(%"struct.std::array"* noalias nocapture align 8 "enzyme_sret" %agg.result, %"struct.std::array"* nocapture "enzyme_sret" %"agg.result'", double %x)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"arrayinit.begin'ipg" = getelementptr inbounds %"struct.std::array", %"struct.std::array"* %"agg.result'", i64 0, i32 0, i64 0
; CHECK-NEXT:   %arrayinit.begin = getelementptr inbounds %"struct.std::array", %"struct.std::array"* %agg.result, i64 0, i32 0, i64 0
; CHECK-NEXT:   store double %x, double* %arrayinit.begin, align 8
; CHECK-NEXT:   %"arrayinit.element'ipg" = getelementptr inbounds %"struct.std::array", %"struct.std::array"* %"agg.result'", i64 0, i32 0, i64 1
; CHECK-NEXT:   %arrayinit.element = getelementptr inbounds %"struct.std::array", %"struct.std::array"* %agg.result, i64 0, i32 0, i64 1
; CHECK-NEXT:   %mul = fmul double %x, %x
; CHECK-NEXT:   store double %mul, double* %arrayinit.element, align 8
; CHECK-NEXT:   %"arrayinit.element1'ipg" = getelementptr inbounds %"struct.std::array", %"struct.std::array"* %"agg.result'", i64 0, i32 0, i64 2
; CHECK-NEXT:   %arrayinit.element1 = getelementptr inbounds %"struct.std::array", %"struct.std::array"* %agg.result, i64 0, i32 0, i64 2
; CHECK-NEXT:   %mul3 = fmul double %mul, %x
; CHECK-NEXT:   store double %mul3, double* %arrayinit.element1, align 8
; CHECK-NEXT:   %0 = load double, double* %"arrayinit.element1'ipg", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"arrayinit.element1'ipg", align 8
; CHECK-NEXT:   %m0diffemul = fmul fast double %0, %x
; CHECK-NEXT:   %m1diffex = fmul fast double %0, %mul
; CHECK-NEXT:   %1 = load double, double* %"arrayinit.element'ipg", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"arrayinit.element'ipg", align 8
; CHECK-NEXT:   %2 = fadd fast double %m0diffemul, %1
; CHECK-NEXT:   %m0diffex = fmul fast double %2, %x
; CHECK-NEXT:   %3 = fadd fast double %m1diffex, %m0diffex
; CHECK-NEXT:   %4 = fadd fast double %3, %m0diffex
; CHECK-NEXT:   %5 = load double, double* %"arrayinit.begin'ipg", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"arrayinit.begin'ipg", align 8
; CHECK-NEXT:   %6 = fadd fast double %4, %5
; CHECK-NEXT:   %7 = insertvalue { double } undef, double %6, 0
; CHECK-NEXT:   ret { double } %7
; CHECK-NEXT: }
