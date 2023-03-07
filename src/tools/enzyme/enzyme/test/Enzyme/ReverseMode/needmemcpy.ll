; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -adce -loop-deletion -correlated-propagation -simplifycfg -S | FileCheck %s

source_filename = "/mnt/Data/git/Enzyme/enzyme/test/Integration/integrateconst.cpp"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%"class.std::ios_base::Init" = type { i8 }
%"class.boost::detail::multi_array::extent_gen" = type { %"class.boost::array" }
%"class.boost::array" = type { [1 x %"class.boost::detail::multi_array::extent_range"] }
%"class.boost::detail::multi_array::extent_range" = type { %"struct.std::pair" }
%"struct.std::pair" = type { i64, i64 }
%"struct.boost::detail::multi_array::index_gen" = type { %"class.boost::array.0" }
%"class.boost::array.0" = type { [1 x %"class.boost::detail::multi_array::index_range"] }
%"class.boost::detail::multi_array::index_range" = type <{ i64, i64, i64, i8, [7 x i8] }>
%"class.boost::array.1" = type { [1 x double] }
%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }
%"struct.std::_Placeholder" = type { i8 }
%"class.std::exception" = type { i32 (...)** }
%"class.boost::numeric::odeint::explicit_stepper_base" = type { %"class.boost::numeric::odeint::algebra_stepper_base", %"struct.boost::numeric::odeint::initially_resizer", %"struct.boost::numeric::odeint::state_wrapper" }
%"class.boost::numeric::odeint::algebra_stepper_base" = type { %"struct.boost::numeric::odeint::array_algebra" }
%"struct.boost::numeric::odeint::array_algebra" = type { i8 }
%"struct.boost::numeric::odeint::initially_resizer" = type { i8 }
%"struct.boost::numeric::odeint::state_wrapper" = type { %"class.boost::array.1" }

$_ZN5boost7numeric6odeint21explicit_stepper_baseINS1_5eulerINS_5arrayIdLm1EEEdS5_dNS1_13array_algebraENS1_18default_operationsENS1_17initially_resizerEEELt1ES5_dS5_dS6_S7_S8_E11resize_implIS5_EEbRKT_ = comdat any

@_ZStL8__ioinit = internal global %"class.std::ios_base::Init" zeroinitializer, align 1
@_ZN5boost12_GLOBAL__N_17extentsE = internal global %"class.boost::detail::multi_array::extent_gen" zeroinitializer, align 8
@_ZN5boost12_GLOBAL__N_17indicesE = internal global %"struct.boost::detail::multi_array::index_gen" zeroinitializer, align 8
@_ZZ6foobardE1x = private unnamed_addr constant double 1.000000e+00, align 8
@.str = private unnamed_addr constant [47 x i8] c"final result t=%f x(t)=%f, -0.2=%f, steps=%zu\0A\00", align 1
@.str.3 = private unnamed_addr constant [48 x i8] c"t=%f d/dt(exp(-1.2*t))=%f, -1.2*exp(-1.2*t)=%f\0A\00", align 1
@stderr = external dso_local global %struct._IO_FILE*, align 8
@.str.4 = private unnamed_addr constant [68 x i8] c"Assertion Failed: fabs( [%s = %g] - [%s = %g] ) > %g at %s:%d (%s)\0A\00", align 1
@.str.5 = private unnamed_addr constant [4 x i8] c"res\00", align 1
@.str.6 = private unnamed_addr constant [6 x i8] c"mreal\00", align 1
@.str.7 = private unnamed_addr constant [64 x i8] c"/mnt/Data/git/Enzyme/enzyme/test/Integration/integrateconst.cpp\00", align 1
@__PRETTY_FUNCTION__.main = private unnamed_addr constant [23 x i8] c"int main(int, char **)\00", align 1
@.str.8 = private unnamed_addr constant [13 x i8] c"out of range\00", align 1
@.str.9 = private unnamed_addr constant [26 x i8] c"(i < N)&&(\22out of range\22)\00", align 1
@.str.10 = private unnamed_addr constant [29 x i8] c"/usr/include/boost/array.hpp\00", align 1
@__PRETTY_FUNCTION__._ZN5boost5arrayIdLm1EEixEm = private unnamed_addr constant [105 x i8] c"boost::array::reference boost::array<double, 1>::operator[](boost::array::size_type) [T = double, N = 1]\00", align 1
@__PRETTY_FUNCTION__._ZNK5boost5arrayIdLm1EEixEm = private unnamed_addr constant [117 x i8] c"boost::array::const_reference boost::array<double, 1>::operator[](boost::array::size_type) const [T = double, N = 1]\00", align 1
@_ZNSt12placeholders2_1E = external dso_local global %"struct.std::_Placeholder", align 1
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_integrateconst.cpp, i8* null }]

declare dso_local void @_ZNSt8ios_base4InitC1Ev(%"class.std::ios_base::Init"*) unnamed_addr #0

; Function Attrs: nounwind
declare dso_local void @_ZNSt8ios_base4InitD1Ev(%"class.std::ios_base::Init"*) unnamed_addr #1

; Function Attrs: alwaysinline nounwind uwtable
define internal void @__dtor__ZStL8__ioinit() #2 section ".text.startup" {
entry:
  call void @_ZNSt8ios_base4InitD1Ev(%"class.std::ios_base::Init"* @_ZStL8__ioinit)
  ret void
}

; Function Attrs: nounwind
declare dso_local i32 @atexit(void ()*) #3

; Function Attrs: alwaysinline nounwind uwtable
define dso_local void @_ZN5boost15throw_exceptionERKSt9exception(%"class.std::exception"* dereferenceable(8) %e) #2 {
entry:
  ret void
}

; Function Attrs: alwaysinline nounwind uwtable
define dso_local void @_Z6lorenzRKN5boost5arrayIdLm1EEERS1_d(%"class.boost::array.1"* dereferenceable(8) %x, %"class.boost::array.1"* dereferenceable(8) %dxdt, double %t) #2 {
entry:
  %elems.i = getelementptr inbounds %"class.boost::array.1", %"class.boost::array.1"* %x, i32 0, i32 0
  %arrayidx.i = getelementptr inbounds [1 x double], [1 x double]* %elems.i, i64 0, i64 0
  %0 = load double, double* %arrayidx.i, align 8, !tbaa !2
  %mul = fmul fast double -1.200000e+00, %0
  %elems.i1 = getelementptr inbounds %"class.boost::array.1", %"class.boost::array.1"* %dxdt, i32 0, i32 0
  %arrayidx.i2 = getelementptr inbounds [1 x double], [1 x double]* %elems.i1, i64 0, i64 0
  store double %mul, double* %arrayidx.i2, align 8, !tbaa !2
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #4

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #4

; Function Attrs: alwaysinline nounwind uwtable
define dso_local double @_Z6foobard(double %t) #2 {
entry:
  %x = alloca double, align 8
  %bc = bitcast double* %x to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %bc, i8* align 8 bitcast (double* @_ZZ6foobardE1x to i8*), i64 8, i1 false)
  %div = fdiv fast double %t, 3.000000e+00
  br label %while.body.i.i.i

while.body.i.i.i:                                 ; preds = %entry, %_ZN5boost7numeric6odeint21explicit_stepper_baseINS1_5eulerINS_5arrayIdLm1EEEdS5_dNS1_13array_algebraENS1_18default_operationsENS1_17initially_resizerEEELt1ES5_dS5_dS6_S7_S8_E7do_stepIPFvRKS5_RS5_dES5_EEvT_RT0_dd.exit.i.i.i
  %i = phi i64 [ %inc, %while.body.i.i.i ], [ 0, %entry ]
  %a1 = load double, double* %x, align 8, !tbaa !2
  %mul.i = fmul fast double -1.200000e+00, %a1
  %mul2 = fmul fast double %div, %mul.i
  %add = fadd fast double %a1, %mul2
  store double %add, double* %x, align 8, !tbaa !2
  %inc = add nsw i64 %i, 1
  %cmp2 = icmp ne i64 %inc, 3
  br i1 %cmp2, label %while.body.i.i.i, label %exit

exit:
  %a6 = load double, double* %x, align 8, !tbaa !2
  ret double %a6
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1) #4

declare dso_local i32 @printf(i8*, ...) #0

; Function Attrs: norecurse nounwind uwtable
define dso_local i32 @main(i32 %argc, i8** %argv) #5 {
entry:
  br label %for.body

for.cond:                                         ; preds = %for.end
  %cmp = icmp sle i32 %inc23, 100
  br i1 %cmp, label %for.body, label %for.end24

for.body:                                         ; preds = %for.cond, %entry
  %i.03 = phi i32 [ 1, %entry ], [ %inc23, %for.cond ]
  %conv = sitofp i32 %i.03 to double
  %0 = fmul fast double %conv, 1.000000e-01
  %mul = fmul fast double 1.200000e+00, %0
  %1 = fmul fast double %mul, 0x3FD5555555555555
  %sub9 = fsub fast double 1.000000e+00, %1
  br label %for.body6

for.body6:                                        ; preds = %for.body6, %for.body
  %i1.02 = phi i32 [ 0, %for.body ], [ %inc, %for.body6 ]
  %mreal.01 = phi double [ -1.200000e+00, %for.body ], [ %mul10, %for.body6 ]
  %mul10 = fmul fast double %mreal.01, %sub9
  %inc = add nsw i32 %i1.02, 1
  %cmp4 = icmp slt i32 %inc, 2
  br i1 %cmp4, label %for.body6, label %for.end

for.end:                                          ; preds = %for.body6
  %call = call fast double @__enzyme_autodiff(i8* bitcast (double (double)* @_Z6foobard to i8*), double %0)
  %call11 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([48 x i8], [48 x i8]* @.str.3, i32 0, i32 0), double %0, double %call, double %mul10)
  %sub12 = fsub fast double %call, %mul10
  %2 = call fast double @llvm.fabs.f64(double %sub12)
  %3 = call fast double @llvm.fabs.f64(double %mul10)
  %4 = fmul fast double %3, 1.000000e-01
  %cmp.i = fcmp fast olt double %4, 2.000000e-05
  %.div13 = select i1 %cmp.i, double 2.000000e-05, double %4
  %cmp16 = fcmp fast ogt double %2, %.div13
  %inc23 = add nsw i32 %i.03, 1
  br i1 %cmp16, label %if.then, label %for.cond

if.then:                                          ; preds = %for.end
  %5 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !6
  %call21 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %5, i8* getelementptr inbounds ([68 x i8], [68 x i8]* @.str.4, i32 0, i32 0), i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.5, i32 0, i32 0), double %call, i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.6, i32 0, i32 0), double %mul10, double %.div13, i8* getelementptr inbounds ([64 x i8], [64 x i8]* @.str.7, i32 0, i32 0), i32 75, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @__PRETTY_FUNCTION__.main, i32 0, i32 0))
  call void @abort() #8
  unreachable

for.end24:                                        ; preds = %for.cond
  ret i32 0
}

declare dso_local double @__enzyme_autodiff(i8*, double) #0

; Function Attrs: nounwind readnone speculatable
declare double @llvm.fabs.f64(double) #6

declare dso_local i32 @fprintf(%struct._IO_FILE*, i8*, ...) #0

; Function Attrs: noreturn nounwind
declare dso_local void @abort() #7

; Function Attrs: noreturn nounwind
declare dso_local void @__assert_fail(i8*, i8*, i32, i8*) #7

; Function Attrs: alwaysinline nounwind uwtable
define linkonce_odr dso_local zeroext i1 @_ZN5boost7numeric6odeint21explicit_stepper_baseINS1_5eulerINS_5arrayIdLm1EEEdS5_dNS1_13array_algebraENS1_18default_operationsENS1_17initially_resizerEEELt1ES5_dS5_dS6_S7_S8_E11resize_implIS5_EEbRKT_(%"class.boost::numeric::odeint::explicit_stepper_base"* %this, %"class.boost::array.1"* dereferenceable(8) %x) #2 comdat align 2 {
entry:
  ret i1 false
}

; Function Attrs: alwaysinline nounwind uwtable
define internal void @_GLOBAL__sub_I_integrateconst.cpp() #2 section ".text.startup" {
entry:
  call void @_ZNSt8ios_base4InitC1Ev(%"class.std::ios_base::Init"* @_ZStL8__ioinit) #3
  %0 = call i32 @atexit(void ()* @__dtor__ZStL8__ioinit) #3
  br label %arrayctor.loop.i.i.i

arrayctor.loop.i.i.i:                             ; preds = %arrayctor.loop.i.i.i, %entry
  %arrayctor.cur.i.i.i = phi %"class.boost::detail::multi_array::extent_range"* [ getelementptr inbounds (%"class.boost::detail::multi_array::extent_gen", %"class.boost::detail::multi_array::extent_gen"* @_ZN5boost12_GLOBAL__N_17extentsE, i64 0, i32 0, i32 0, i64 0), %entry ], [ %arrayctor.next.i.i.i, %arrayctor.loop.i.i.i ]
  %1 = bitcast %"class.boost::detail::multi_array::extent_range"* %arrayctor.cur.i.i.i to %"struct.std::pair"*
  %first.i.i.i.i = getelementptr inbounds %"struct.std::pair", %"struct.std::pair"* %1, i32 0, i32 0
  store i64 0, i64* %first.i.i.i.i, align 8, !tbaa !8
  %second.i.i.i.i = getelementptr inbounds %"struct.std::pair", %"struct.std::pair"* %1, i32 0, i32 1
  store i64 0, i64* %second.i.i.i.i, align 8, !tbaa !11
  %arrayctor.next.i.i.i = getelementptr inbounds %"class.boost::detail::multi_array::extent_range", %"class.boost::detail::multi_array::extent_range"* %arrayctor.cur.i.i.i, i64 1
  %arrayctor.done.i.i.i = icmp eq %"class.boost::detail::multi_array::extent_range"* %arrayctor.next.i.i.i, getelementptr inbounds (%"class.boost::detail::multi_array::extent_gen", %"class.boost::detail::multi_array::extent_gen"* @_ZN5boost12_GLOBAL__N_17extentsE, i64 1, i32 0, i32 0, i64 0)
  br i1 %arrayctor.done.i.i.i, label %arrayctor.loop.i.i.i4, label %arrayctor.loop.i.i.i

arrayctor.loop.i.i.i4:                            ; preds = %arrayctor.loop.i.i.i, %arrayctor.loop.i.i.i4
  %arrayctor.cur.i.i.i1 = phi %"class.boost::detail::multi_array::index_range"* [ %arrayctor.next.i.i.i2, %arrayctor.loop.i.i.i4 ], [ getelementptr inbounds (%"struct.boost::detail::multi_array::index_gen", %"struct.boost::detail::multi_array::index_gen"* @_ZN5boost12_GLOBAL__N_17indicesE, i64 0, i32 0, i32 0, i64 0), %arrayctor.loop.i.i.i ]
  %start_.i.i.i = getelementptr inbounds %"class.boost::detail::multi_array::index_range", %"class.boost::detail::multi_array::index_range"* %arrayctor.cur.i.i.i1, i32 0, i32 0
  store i64 -9223372036854775808, i64* %start_.i.i.i, align 8, !tbaa !12
  %finish_.i.i.i = getelementptr inbounds %"class.boost::detail::multi_array::index_range", %"class.boost::detail::multi_array::index_range"* %arrayctor.cur.i.i.i1, i32 0, i32 1
  store i64 9223372036854775807, i64* %finish_.i.i.i, align 8, !tbaa !15
  %stride_.i.i.i = getelementptr inbounds %"class.boost::detail::multi_array::index_range", %"class.boost::detail::multi_array::index_range"* %arrayctor.cur.i.i.i1, i32 0, i32 2
  store i64 1, i64* %stride_.i.i.i, align 8, !tbaa !16
  %degenerate_.i.i.i = getelementptr inbounds %"class.boost::detail::multi_array::index_range", %"class.boost::detail::multi_array::index_range"* %arrayctor.cur.i.i.i1, i32 0, i32 3
  store i8 0, i8* %degenerate_.i.i.i, align 8, !tbaa !17
  %arrayctor.next.i.i.i2 = getelementptr inbounds %"class.boost::detail::multi_array::index_range", %"class.boost::detail::multi_array::index_range"* %arrayctor.cur.i.i.i1, i64 1
  %arrayctor.done.i.i.i3 = icmp eq %"class.boost::detail::multi_array::index_range"* %arrayctor.next.i.i.i2, getelementptr inbounds (%"struct.boost::detail::multi_array::index_gen", %"struct.boost::detail::multi_array::index_gen"* @_ZN5boost12_GLOBAL__N_17indicesE, i64 1, i32 0, i32 0, i64 0)
  br i1 %arrayctor.done.i.i.i3, label %__cxx_global_var_init.2.exit, label %arrayctor.loop.i.i.i4

__cxx_global_var_init.2.exit:                     ; preds = %arrayctor.loop.i.i.i4
  ret void
}

attributes #0 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #2 = { alwaysinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #3 = { nounwind }
attributes #4 = { argmemonly nounwind }
attributes #5 = { norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #6 = { nounwind readnone speculatable }
attributes #7 = { noreturn nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #8 = { noreturn nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0}
!3 = !{!"double", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C++ TBAA"}
!6 = !{!7, !7, i64 0}
!7 = !{!"any pointer", !4, i64 0}
!8 = !{!9, !10, i64 0}
!9 = !{!"_ZTSSt4pairIllE", !10, i64 0, !10, i64 8}
!10 = !{!"long", !4, i64 0}
!11 = !{!9, !10, i64 8}
!12 = !{!13, !10, i64 0}
!13 = !{!"_ZTSN5boost6detail11multi_array11index_rangeIlmEE", !10, i64 0, !10, i64 8, !10, i64 16, !14, i64 24}
!14 = !{!"bool", !4, i64 0}
!15 = !{!13, !10, i64 8}
!16 = !{!13, !10, i64 16}
!17 = !{!13, !14, i64 24}

; CHECK: define internal { double } @diffe_Z6foobard(double %t, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"x'ipa" = alloca double, align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"x'ipa", align 8
; CHECK-NEXT:   %x = alloca double, align 8
; CHECK-NEXT:   %"bc'ipc" = bitcast double* %"x'ipa" to i8*
; CHECK-NEXT:   %bc = bitcast double* %x to i8*
; CHECK-NEXT:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 %bc, i8* align 8 bitcast (double* @_ZZ6foobardE1x to i8*), i64 8, i1 false)
; CHECK-NEXT:   %div = fdiv fast double %t, 3.000000e+00
; CHECK-NEXT:   %malloccall = tail call noalias nonnull dereferenceable(24) dereferenceable_or_null(24) i8* @malloc(i64 24)
; CHECK-NEXT:   %mul.i_malloccache = bitcast i8* %malloccall to double*
; CHECK-NEXT:   br label %while.body.i.i.i

; CHECK: while.body.i.i.i:                                 ; preds = %while.body.i.i.i, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %while.body.i.i.i ], [ 0, %entry ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %a1 = load double, double* %x, align 8, !tbaa !
; CHECK-NEXT:   %mul.i = fmul fast double -1.200000e+00, %a1
; CHECK-NEXT:   %mul2 = fmul fast double %div, %mul.i
; CHECK-NEXT:   %add = fadd fast double %a1, %mul2
; CHECK-NEXT:   store double %add, double* %x, align 8, !tbaa !
; CHECK-NEXT:   %0 = getelementptr inbounds double, double* %mul.i_malloccache, i64 %iv
; CHECK-NEXT:   store double %mul.i, double* %0, align 8, !invariant.group ![[ig:[0-9]+]]
; CHECK-NEXT:   %cmp2 = icmp ne i64 %iv.next, 3
; CHECK-NEXT:   br i1 %cmp2, label %while.body.i.i.i, label %invertexit

; CHECK: invertentry:                                      ; preds = %invertwhile.body.i.i.i
; CHECK-NEXT:   %d0diffet = fdiv fast double %5, 3.000000e+00
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull align 8 %"bc'ipc", i8 0, i64 8, i1 false)
; CHECK-NEXT:   %1 = insertvalue { double } undef, double %d0diffet, 0
; CHECK-NEXT:   tail call void @free(i8* nonnull %malloccall)
; CHECK-NEXT:   ret { double } %1

; CHECK: invertwhile.body.i.i.i:                           ; preds = %invertexit, %incinvertwhile.body.i.i.i
; CHECK-NEXT:   %"div'de.0" = phi double [ 0.000000e+00, %invertexit ], [ %5, %incinvertwhile.body.i.i.i ]
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ 2, %invertexit ], [ %10, %incinvertwhile.body.i.i.i ]
; CHECK-NEXT:   %2 = load double, double* %"x'ipa", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"x'ipa", align 8
; CHECK-NEXT:   %3 = getelementptr inbounds double, double* %mul.i_malloccache, i64 %"iv'ac.0"
; CHECK-NEXT:   %4 = load double, double* %3, align 8, !invariant.group ![[ig]]
; CHECK-NEXT:   %m0diffediv = fmul fast double %2, %4
; CHECK-NEXT:   %m1diffemul.i = fmul fast double %2, %div
; CHECK-NEXT:   %5 = fadd fast double %"div'de.0", %m0diffediv
; CHECK-NEXT:   %m1diffea1 = fmul fast double %m1diffemul.i, -1.200000e+00
; CHECK-NEXT:   %6 = fadd fast double %2, %m1diffea1
; CHECK-NEXT:   %7 = load double, double* %"x'ipa", align 8
; CHECK-NEXT:   %8 = fadd fast double %7, %6
; CHECK-NEXT:   store double %8, double* %"x'ipa", align 8
; CHECK-NEXT:   %9 = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %9, label %invertentry, label %incinvertwhile.body.i.i.i

; CHECK: incinvertwhile.body.i.i.i:                        ; preds = %invertwhile.body.i.i.i
; CHECK-NEXT:   %10 = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertwhile.body.i.i.i

; CHECK: invertexit:                                       ; preds = %while.body.i.i.i
; CHECK-NEXT:   %11 = load double, double* %"x'ipa", align 8
; CHECK-NEXT:   %12 = fadd fast double %11, %differeturn
; CHECK-NEXT:   store double %12, double* %"x'ipa", align 8
; CHECK-NEXT:   br label %invertwhile.body.i.i.i
; CHECK-NEXT: }
