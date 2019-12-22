; ModuleID = 'pre.ll'
source_filename = "./integrateexp.cpp"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%"class.std::ios_base::Init" = type { i8 }
%"class.std::exception" = type { i32 (...)** }
%"class.boost::array.1" = type { [1 x double] }
%"class.boost::numeric::odeint::step_adjustment_error" = type { %"class.boost::numeric::odeint::odeint_error" }
%"class.boost::numeric::odeint::odeint_error" = type { %"class.std::runtime_error" }
%"class.std::runtime_error" = type { %"class.std::exception", %"struct.std::__cow_string" }
%"struct.std::__cow_string" = type { %union.anon }
%union.anon = type { i8* }
%"struct.boost::exception_detail::error_info_injector" = type <{ %"class.boost::numeric::odeint::step_adjustment_error", %"class.boost::exception.base", [4 x i8] }>
%"class.boost::exception.base" = type <{ i32 (...)**, %"class.boost::exception_detail::refcount_ptr", i8*, i8*, i32 }>
%"class.boost::exception_detail::refcount_ptr" = type { %"struct.boost::exception_detail::error_info_container"* }
%"struct.boost::exception_detail::error_info_container" = type { i32 (...)** }
%"class.std::__cxx11::basic_string" = type { %"struct.std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Alloc_hider", i64, %union.anon.24 }
%"struct.std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Alloc_hider" = type { i8* }
%union.anon.24 = type { i64, [8 x i8] }
%"class.std::allocator" = type { i8 }

$_ZN5boost16exception_detail16throw_exception_INS_7numeric6odeint21step_adjustment_errorEEEvRKT_PKcS9_i = comdat any

$_ZN5boost7numeric6odeint21step_adjustment_errorC2ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE = comdat any

$_ZN5boost17enable_error_infoINS_7numeric6odeint21step_adjustment_errorEEENS_16exception_detail29enable_error_info_return_typeIT_E4typeERKS6_ = comdat any

$_ZN5boost16exception_detail19error_info_injectorINS_7numeric6odeint21step_adjustment_errorEED2Ev = comdat any

$_ZN5boost16exception_detail19error_info_injectorINS_7numeric6odeint21step_adjustment_errorEED0Ev = comdat any

$_ZThn16_N5boost16exception_detail19error_info_injectorINS_7numeric6odeint21step_adjustment_errorEED1Ev = comdat any

$_ZThn16_N5boost16exception_detail19error_info_injectorINS_7numeric6odeint21step_adjustment_errorEED0Ev = comdat any

$_ZN5boost7numeric6odeint21step_adjustment_errorD0Ev = comdat any

$_ZTVN5boost16exception_detail19error_info_injectorINS_7numeric6odeint21step_adjustment_errorEEE = comdat any

$_ZTSN5boost16exception_detail19error_info_injectorINS_7numeric6odeint21step_adjustment_errorEEE = comdat any

$_ZTSN5boost7numeric6odeint21step_adjustment_errorE = comdat any

$_ZTSN5boost7numeric6odeint12odeint_errorE = comdat any

$_ZTIN5boost7numeric6odeint12odeint_errorE = comdat any

$_ZTIN5boost7numeric6odeint21step_adjustment_errorE = comdat any

$_ZTSN5boost9exceptionE = comdat any

$_ZTIN5boost9exceptionE = comdat any

$_ZTIN5boost16exception_detail19error_info_injectorINS_7numeric6odeint21step_adjustment_errorEEE = comdat any

$_ZTVN5boost7numeric6odeint21step_adjustment_errorE = comdat any

$_ZTVN5boost9exceptionE = comdat any

@_ZStL8__ioinit = internal global %"class.std::ios_base::Init" zeroinitializer, align 1
@.str = private unnamed_addr constant [46 x i8] c"final result t=%f x(t)=%f, -0.2=%f, steps=%d\0A\00", align 1
@.str.3 = private unnamed_addr constant [48 x i8] c"t=%f d/dt(exp(-1.2*t))=%f, -1.2*exp(-1.2*t)=%f\0A\00", align 1
@.str.7 = private unnamed_addr constant [71 x i8] c"Max number of iterations exceeded (%d). A new step size was not found.\00", align 1
@__PRETTY_FUNCTION__._ZN5boost7numeric6odeint19failed_step_checkerclEv = private unnamed_addr constant [63 x i8] c"void boost::numeric::odeint::failed_step_checker::operator()()\00", align 1
@.str.8 = private unnamed_addr constant [65 x i8] c"/usr/include/boost/numeric/odeint/integrate/max_step_checker.hpp\00", align 1
@_ZTVN5boost16exception_detail19error_info_injectorINS_7numeric6odeint21step_adjustment_errorEEE = linkonce_odr dso_local unnamed_addr constant { [5 x i8*], [4 x i8*] } { [5 x i8*] [i8* null, i8* bitcast ({ i8*, i8*, i32, i32, i8*, i64, i8*, i64 }* @_ZTIN5boost16exception_detail19error_info_injectorINS_7numeric6odeint21step_adjustment_errorEEE to i8*), i8* bitcast (void (%"struct.boost::exception_detail::error_info_injector"*)* @_ZN5boost16exception_detail19error_info_injectorINS_7numeric6odeint21step_adjustment_errorEED2Ev to i8*), i8* bitcast (void (%"struct.boost::exception_detail::error_info_injector"*)* @_ZN5boost16exception_detail19error_info_injectorINS_7numeric6odeint21step_adjustment_errorEED0Ev to i8*), i8* bitcast (i8* (%"class.std::runtime_error"*)* @_ZNKSt13runtime_error4whatEv to i8*)], [4 x i8*] [i8* inttoptr (i64 -16 to i8*), i8* bitcast ({ i8*, i8*, i32, i32, i8*, i64, i8*, i64 }* @_ZTIN5boost16exception_detail19error_info_injectorINS_7numeric6odeint21step_adjustment_errorEEE to i8*), i8* bitcast (void (%"struct.boost::exception_detail::error_info_injector"*)* @_ZThn16_N5boost16exception_detail19error_info_injectorINS_7numeric6odeint21step_adjustment_errorEED1Ev to i8*), i8* bitcast (void (%"struct.boost::exception_detail::error_info_injector"*)* @_ZThn16_N5boost16exception_detail19error_info_injectorINS_7numeric6odeint21step_adjustment_errorEED0Ev to i8*)] }, comdat, align 8
@_ZTVN10__cxxabiv121__vmi_class_type_infoE = external dso_local global i8*
@_ZTSN5boost16exception_detail19error_info_injectorINS_7numeric6odeint21step_adjustment_errorEEE = linkonce_odr dso_local constant [92 x i8] c"N5boost16exception_detail19error_info_injectorINS_7numeric6odeint21step_adjustment_errorEEE\00", comdat
@_ZTVN10__cxxabiv120__si_class_type_infoE = external dso_local global i8*
@_ZTSN5boost7numeric6odeint21step_adjustment_errorE = linkonce_odr dso_local constant [47 x i8] c"N5boost7numeric6odeint21step_adjustment_errorE\00", comdat
@_ZTSN5boost7numeric6odeint12odeint_errorE = linkonce_odr dso_local constant [38 x i8] c"N5boost7numeric6odeint12odeint_errorE\00", comdat
@_ZTISt13runtime_error = external dso_local constant i8*
@_ZTIN5boost7numeric6odeint12odeint_errorE = linkonce_odr dso_local constant { i8*, i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([38 x i8], [38 x i8]* @_ZTSN5boost7numeric6odeint12odeint_errorE, i32 0, i32 0), i8* bitcast (i8** @_ZTISt13runtime_error to i8*) }, comdat
@_ZTIN5boost7numeric6odeint21step_adjustment_errorE = linkonce_odr dso_local constant { i8*, i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([47 x i8], [47 x i8]* @_ZTSN5boost7numeric6odeint21step_adjustment_errorE, i32 0, i32 0), i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN5boost7numeric6odeint12odeint_errorE to i8*) }, comdat
@_ZTVN10__cxxabiv117__class_type_infoE = external dso_local global i8*
@_ZTSN5boost9exceptionE = linkonce_odr dso_local constant [19 x i8] c"N5boost9exceptionE\00", comdat
@_ZTIN5boost9exceptionE = linkonce_odr dso_local constant { i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv117__class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([19 x i8], [19 x i8]* @_ZTSN5boost9exceptionE, i32 0, i32 0) }, comdat
@_ZTIN5boost16exception_detail19error_info_injectorINS_7numeric6odeint21step_adjustment_errorEEE = linkonce_odr dso_local constant { i8*, i8*, i32, i32, i8*, i64, i8*, i64 } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv121__vmi_class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([92 x i8], [92 x i8]* @_ZTSN5boost16exception_detail19error_info_injectorINS_7numeric6odeint21step_adjustment_errorEEE, i32 0, i32 0), i32 0, i32 2, i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN5boost7numeric6odeint21step_adjustment_errorE to i8*), i64 2, i8* bitcast ({ i8*, i8* }* @_ZTIN5boost9exceptionE to i8*), i64 4098 }, comdat
@_ZTVN5boost7numeric6odeint21step_adjustment_errorE = linkonce_odr dso_local unnamed_addr constant { [5 x i8*] } { [5 x i8*] [i8* null, i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN5boost7numeric6odeint21step_adjustment_errorE to i8*), i8* bitcast (void (%"class.std::runtime_error"*)* @_ZNSt13runtime_errorD2Ev to i8*), i8* bitcast (void (%"class.boost::numeric::odeint::step_adjustment_error"*)* @_ZN5boost7numeric6odeint21step_adjustment_errorD0Ev to i8*), i8* bitcast (i8* (%"class.std::runtime_error"*)* @_ZNKSt13runtime_error4whatEv to i8*)] }, comdat, align 8
@_ZTVN5boost9exceptionE = linkonce_odr dso_local unnamed_addr constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* bitcast ({ i8*, i8* }* @_ZTIN5boost9exceptionE to i8*), i8* bitcast (void ()* @__cxa_pure_virtual to i8*), i8* bitcast (void ()* @__cxa_pure_virtual to i8*)] }, comdat, align 8
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_integrateexp.cpp, i8* null }]

; Function Attrs: nounwind readnone uwtable
define dso_local zeroext i1 @_Z24approx_fp_equality_floatffd(float %f1, float %f2, double %threshold) local_unnamed_addr #0 {
entry:
  %sub = fsub fast float %f1, %f2
  %0 = tail call fast float @llvm.fabs.f32(float %sub) #4
  %conv = fpext float %0 to double
  %cmp = fcmp fast ule double %conv, %threshold
  ret i1 %cmp
}

declare dso_local void @_ZNSt8ios_base4InitC1Ev(%"class.std::ios_base::Init"*) unnamed_addr #1

; Function Attrs: nounwind
declare dso_local void @_ZNSt8ios_base4InitD1Ev(%"class.std::ios_base::Init"*) unnamed_addr #2

; Function Attrs: nounwind uwtable
define internal void @__dtor__ZStL8__ioinit() #3 section ".text.startup" {
entry:
  tail call void @_ZNSt8ios_base4InitD1Ev(%"class.std::ios_base::Init"* nonnull @_ZStL8__ioinit)
  ret void
}

; Function Attrs: nounwind
declare dso_local i32 @atexit(void ()*) local_unnamed_addr #4

; Function Attrs: norecurse nounwind readnone uwtable
define dso_local void @_ZN5boost15throw_exceptionERKSt9exception(%"class.std::exception"* nocapture dereferenceable(8) %e) local_unnamed_addr #5 {
entry:
  ret void
}

; Function Attrs: alwaysinline norecurse nounwind uwtable
define dso_local double @step(double* %x0) #6 {
entry:
  %load = load double, double* %x0, align 8, !tbaa !2
  ret double %load
}

; Function Attrs: alwaysinline norecurse nounwind uwtable
define dso_local void @_Z6lorenzRKN5boost5arrayIdLm1EEERS1_d(%"class.boost::array.1"* nocapture readonly dereferenceable(8) %x, %"class.boost::array.1"* nocapture dereferenceable(8) %dxdt, double %t) #6 {
entry:
  %arrayidx.i = getelementptr inbounds %"class.boost::array.1", %"class.boost::array.1"* %x, i64 0, i32 0, i64 0
  %0 = load double, double* %arrayidx.i, align 8, !tbaa !2
  %mul = fmul fast double %0, -1.200000e+00
  %arrayidx.i3 = getelementptr inbounds %"class.boost::array.1", %"class.boost::array.1"* %dxdt, i64 0, i32 0, i64 0
  store double %mul, double* %arrayidx.i3, align 8, !tbaa !2
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #7

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #7

declare noalias i8* @malloc(i64)

; Function Attrs: nounwind uwtable
define dso_local double @_Z6foobard(double %t) #3 {
entry:
  %malloccall = tail call i8* @malloc(i64 8) #4
  %x = bitcast i8* %malloccall to double*
  %0 = bitcast i8* %malloccall to i64*
  store i64 4607182418800017408, i64* %0, align 8
  %div = fmul fast double %t, 1.000000e-02
  %x.promoted = load double, double* %x, align 8
  br label %while.body.i.i.i

while.body.i.i.i:                                 ; preds = %while.body.i.i.i, %entry
  %load.i1 = phi double [ %x.promoted, %entry ], [ %add10.i.i.i, %while.body.i.i.i ]
  %step.029.i.i.i = phi i32 [ 0, %entry ], [ %inc.i.i.i, %while.body.i.i.i ]
  %1 = fmul fast double %load.i1, 0xBFF3333333333332
  %reass.mul325.i = fmul fast double %1, %div
  %add10.i.i.i = fadd fast double %reass.mul325.i, %load.i1
  %inc.i.i.i = add nuw nsw i32 %step.029.i.i.i, 1
  %conv8.i.i.i = sitofp i32 %inc.i.i.i to double
  %mul.i.i.i = fmul fast double %div, %conv8.i.i.i
  %add.i.i.i = fadd fast double %mul.i.i.i, %div
  %sub.i.i.i.i = fsub fast double %add.i.i.i, %t
  %cmp2.i.i.i.i = fcmp fast ugt double %sub.i.i.i.i, 0x3CB0000000000000
  br i1 %cmp2.i.i.i.i, label %loopexit, label %while.body.i.i.i

loopexit:                                         ; preds = %while.body.i.i.i
  store double %add10.i.i.i, double* %x, align 8
  ret double %add10.i.i.i
}

; Function Attrs: nounwind
declare dso_local i32 @printf(i8* nocapture readonly, ...) local_unnamed_addr #2

; Function Attrs: norecurse nounwind uwtable
define dso_local i32 @main(i32 %argc, i8** nocapture readnone %argv) local_unnamed_addr #8 {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret i32 0

for.body:                                         ; preds = %for.body, %entry
  %i.033 = phi i32 [ 1, %entry ], [ %inc15, %for.body ]
  %conv = sitofp i32 %i.033 to double
  %div = fmul fast double %conv, 1.000000e-01
  %call = tail call fast double @__enzyme_autodiff(i8* bitcast (double (double)* @_Z6foobard to i8*), double %div) #4
  %mul11 = fmul fast double %conv, -1.200000e-01
  %0 = tail call fast double @llvm.exp.f64(double %mul11)
  %mul12 = fmul fast double %0, -1.200000e+00
  %call13 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([48 x i8], [48 x i8]* @.str.3, i64 0, i64 0), double %div, double %call, double %mul12)
  %inc15 = add nuw nsw i32 %i.033, 1
  %exitcond = icmp eq i32 %inc15, 101
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

declare dso_local double @__enzyme_autodiff(i8*, double) local_unnamed_addr #1

; Function Attrs: nounwind readnone speculatable
declare double @llvm.exp.f64(double) #9

; Function Attrs: nounwind readnone speculatable
declare float @llvm.fabs.f32(float) #9

; Function Attrs: nounwind readnone speculatable
declare double @llvm.fabs.f64(double) #9

; Function Attrs: nounwind readnone speculatable
declare double @llvm.pow.f64(double, double) #9

; Function Attrs: nounwind
declare dso_local i32 @sprintf(i8* nocapture, i8* nocapture readonly, ...) local_unnamed_addr #2

; Function Attrs: noreturn nounwind uwtable
define linkonce_odr dso_local void @_ZN5boost16exception_detail16throw_exception_INS_7numeric6odeint21step_adjustment_errorEEEvRKT_PKcS9_i(%"class.boost::numeric::odeint::step_adjustment_error"* dereferenceable(16) %x, i8* %current_function, i8* %file, i32 %line) local_unnamed_addr #10 comdat {
entry:
  %ref.tmp = alloca %"struct.boost::exception_detail::error_info_injector", align 8
  %0 = bitcast %"struct.boost::exception_detail::error_info_injector"* %ref.tmp to i8*
  call void @llvm.lifetime.start.p0i8(i64 56, i8* nonnull %0) #4
  call void @_ZN5boost17enable_error_infoINS_7numeric6odeint21step_adjustment_errorEEENS_16exception_detail29enable_error_info_return_typeIT_E4typeERKS6_(%"struct.boost::exception_detail::error_info_injector"* nonnull sret %ref.tmp, %"class.boost::numeric::odeint::step_adjustment_error"* nonnull dereferenceable(16) %x)
  %1 = ptrtoint i8* %current_function to i64
  %2 = getelementptr inbounds %"struct.boost::exception_detail::error_info_injector", %"struct.boost::exception_detail::error_info_injector"* %ref.tmp, i64 0, i32 1, i32 2
  %3 = bitcast i8** %2 to i64*
  store i64 %1, i64* %3, align 8, !tbaa !6
  %4 = ptrtoint i8* %file to i64
  %5 = getelementptr inbounds %"struct.boost::exception_detail::error_info_injector", %"struct.boost::exception_detail::error_info_injector"* %ref.tmp, i64 0, i32 1, i32 3
  %6 = bitcast i8** %5 to i64*
  store i64 %4, i64* %6, align 8, !tbaa !11
  %7 = getelementptr inbounds %"struct.boost::exception_detail::error_info_injector", %"struct.boost::exception_detail::error_info_injector"* %ref.tmp, i64 0, i32 1, i32 4
  store i32 %line, i32* %7, align 8, !tbaa !12
  call void @_ZN5boost16exception_detail19error_info_injectorINS_7numeric6odeint21step_adjustment_errorEED2Ev(%"struct.boost::exception_detail::error_info_injector"* nonnull %ref.tmp) #4
  call void @llvm.lifetime.end.p0i8(i64 56, i8* nonnull %0) #4
  unreachable
}

; Function Attrs: nounwind uwtable
declare dso_local void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2EPKcRKS3_(%"class.std::__cxx11::basic_string"*, i8*, %"class.std::allocator"* dereferenceable(1)) unnamed_addr #3 align 2

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZN5boost7numeric6odeint21step_adjustment_errorC2ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(%"class.boost::numeric::odeint::step_adjustment_error"* %this, %"class.std::__cxx11::basic_string"* dereferenceable(32) %s) unnamed_addr #3 comdat align 2 {
entry:
  %0 = getelementptr inbounds %"class.boost::numeric::odeint::step_adjustment_error", %"class.boost::numeric::odeint::step_adjustment_error"* %this, i64 0, i32 0, i32 0
  tail call void @_ZNSt13runtime_errorC2ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(%"class.std::runtime_error"* %0, %"class.std::__cxx11::basic_string"* nonnull dereferenceable(32) %s) #4
  %1 = getelementptr inbounds %"class.boost::numeric::odeint::step_adjustment_error", %"class.boost::numeric::odeint::step_adjustment_error"* %this, i64 0, i32 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [5 x i8*] }, { [5 x i8*] }* @_ZTVN5boost7numeric6odeint21step_adjustment_errorE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %1, align 8, !tbaa !13
  ret void
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local void @_ZN5boost17enable_error_infoINS_7numeric6odeint21step_adjustment_errorEEENS_16exception_detail29enable_error_info_return_typeIT_E4typeERKS6_(%"struct.boost::exception_detail::error_info_injector"* noalias sret %agg.result, %"class.boost::numeric::odeint::step_adjustment_error"* dereferenceable(16) %x) local_unnamed_addr #11 comdat {
entry:
  %0 = getelementptr inbounds %"struct.boost::exception_detail::error_info_injector", %"struct.boost::exception_detail::error_info_injector"* %agg.result, i64 0, i32 0, i32 0, i32 0
  %1 = getelementptr inbounds %"class.boost::numeric::odeint::step_adjustment_error", %"class.boost::numeric::odeint::step_adjustment_error"* %x, i64 0, i32 0, i32 0
  tail call void @_ZNSt13runtime_errorC2ERKS_(%"class.std::runtime_error"* %0, %"class.std::runtime_error"* nonnull dereferenceable(16) %1) #4
  %2 = getelementptr inbounds %"struct.boost::exception_detail::error_info_injector", %"struct.boost::exception_detail::error_info_injector"* %agg.result, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %3 = getelementptr inbounds %"struct.boost::exception_detail::error_info_injector", %"struct.boost::exception_detail::error_info_injector"* %agg.result, i64 0, i32 1, i32 0
  %4 = getelementptr inbounds %"struct.boost::exception_detail::error_info_injector", %"struct.boost::exception_detail::error_info_injector"* %agg.result, i64 0, i32 1, i32 1, i32 0
  %5 = getelementptr inbounds %"struct.boost::exception_detail::error_info_injector", %"struct.boost::exception_detail::error_info_injector"* %agg.result, i64 0, i32 1, i32 4
  %6 = bitcast %"struct.boost::exception_detail::error_info_container"** %4 to i8*
  tail call void @llvm.memset.p0i8.i64(i8* nonnull align 8 %6, i8 0, i64 24, i1 false) #4
  store i32 -1, i32* %5, align 8, !tbaa !12
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [5 x i8*], [4 x i8*] }, { [5 x i8*], [4 x i8*] }* @_ZTVN5boost16exception_detail19error_info_injectorINS_7numeric6odeint21step_adjustment_errorEEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %2, align 8, !tbaa !13
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [5 x i8*], [4 x i8*] }, { [5 x i8*], [4 x i8*] }* @_ZTVN5boost16exception_detail19error_info_injectorINS_7numeric6odeint21step_adjustment_errorEEE, i64 0, inrange i32 1, i64 2) to i32 (...)**), i32 (...)*** %3, align 8, !tbaa !13
  ret void
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZN5boost16exception_detail19error_info_injectorINS_7numeric6odeint21step_adjustment_errorEED2Ev(%"struct.boost::exception_detail::error_info_injector"* %this) unnamed_addr #3 comdat align 2 {
entry:
  %0 = getelementptr inbounds %"struct.boost::exception_detail::error_info_injector", %"struct.boost::exception_detail::error_info_injector"* %this, i64 0, i32 1, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN5boost9exceptionE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %0, align 8, !tbaa !13
  %1 = getelementptr inbounds %"struct.boost::exception_detail::error_info_injector", %"struct.boost::exception_detail::error_info_injector"* %this, i64 0, i32 1, i32 1, i32 0
  %2 = load %"struct.boost::exception_detail::error_info_container"*, %"struct.boost::exception_detail::error_info_container"** %1, align 8, !tbaa !15
  %tobool.i.i.i = icmp eq %"struct.boost::exception_detail::error_info_container"* %2, null
  br i1 %tobool.i.i.i, label %_ZN5boost9exceptionD2Ev.exit, label %land.lhs.true.i.i.i

land.lhs.true.i.i.i:                              ; preds = %entry
  %3 = bitcast %"struct.boost::exception_detail::error_info_container"* %2 to i1 (%"struct.boost::exception_detail::error_info_container"*)***
  %vtable.i.i.i = load i1 (%"struct.boost::exception_detail::error_info_container"*)**, i1 (%"struct.boost::exception_detail::error_info_container"*)*** %3, align 8, !tbaa !13
  %vfn.i.i.i = getelementptr inbounds i1 (%"struct.boost::exception_detail::error_info_container"*)*, i1 (%"struct.boost::exception_detail::error_info_container"*)** %vtable.i.i.i, i64 4
  %4 = load i1 (%"struct.boost::exception_detail::error_info_container"*)*, i1 (%"struct.boost::exception_detail::error_info_container"*)** %vfn.i.i.i, align 8
  %call.i.i.i = tail call zeroext i1 %4(%"struct.boost::exception_detail::error_info_container"* nonnull %2) #4
  br i1 %call.i.i.i, label %if.then.i.i.i, label %_ZN5boost9exceptionD2Ev.exit

if.then.i.i.i:                                    ; preds = %land.lhs.true.i.i.i
  store %"struct.boost::exception_detail::error_info_container"* null, %"struct.boost::exception_detail::error_info_container"** %1, align 8, !tbaa !15
  br label %_ZN5boost9exceptionD2Ev.exit

_ZN5boost9exceptionD2Ev.exit:                     ; preds = %if.then.i.i.i, %land.lhs.true.i.i.i, %entry
  %5 = getelementptr inbounds %"struct.boost::exception_detail::error_info_injector", %"struct.boost::exception_detail::error_info_injector"* %this, i64 0, i32 0, i32 0, i32 0
  tail call void @_ZNSt13runtime_errorD2Ev(%"class.std::runtime_error"* %5) #4
  ret void
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZN5boost16exception_detail19error_info_injectorINS_7numeric6odeint21step_adjustment_errorEED0Ev(%"struct.boost::exception_detail::error_info_injector"* %this) unnamed_addr #3 comdat align 2 {
entry:
  %0 = getelementptr inbounds %"struct.boost::exception_detail::error_info_injector", %"struct.boost::exception_detail::error_info_injector"* %this, i64 0, i32 1, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN5boost9exceptionE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %0, align 8, !tbaa !13
  %1 = getelementptr inbounds %"struct.boost::exception_detail::error_info_injector", %"struct.boost::exception_detail::error_info_injector"* %this, i64 0, i32 1, i32 1, i32 0
  %2 = load %"struct.boost::exception_detail::error_info_container"*, %"struct.boost::exception_detail::error_info_container"** %1, align 8, !tbaa !15
  %tobool.i.i.i.i = icmp eq %"struct.boost::exception_detail::error_info_container"* %2, null
  br i1 %tobool.i.i.i.i, label %_ZN5boost16exception_detail19error_info_injectorINS_7numeric6odeint21step_adjustment_errorEED2Ev.exit, label %land.lhs.true.i.i.i.i

land.lhs.true.i.i.i.i:                            ; preds = %entry
  %3 = bitcast %"struct.boost::exception_detail::error_info_container"* %2 to i1 (%"struct.boost::exception_detail::error_info_container"*)***
  %vtable.i.i.i.i = load i1 (%"struct.boost::exception_detail::error_info_container"*)**, i1 (%"struct.boost::exception_detail::error_info_container"*)*** %3, align 8, !tbaa !13
  %vfn.i.i.i.i = getelementptr inbounds i1 (%"struct.boost::exception_detail::error_info_container"*)*, i1 (%"struct.boost::exception_detail::error_info_container"*)** %vtable.i.i.i.i, i64 4
  %4 = load i1 (%"struct.boost::exception_detail::error_info_container"*)*, i1 (%"struct.boost::exception_detail::error_info_container"*)** %vfn.i.i.i.i, align 8
  %call.i.i.i.i = tail call zeroext i1 %4(%"struct.boost::exception_detail::error_info_container"* nonnull %2) #4
  br i1 %call.i.i.i.i, label %if.then.i.i.i.i, label %_ZN5boost16exception_detail19error_info_injectorINS_7numeric6odeint21step_adjustment_errorEED2Ev.exit

if.then.i.i.i.i:                                  ; preds = %land.lhs.true.i.i.i.i
  store %"struct.boost::exception_detail::error_info_container"* null, %"struct.boost::exception_detail::error_info_container"** %1, align 8, !tbaa !15
  br label %_ZN5boost16exception_detail19error_info_injectorINS_7numeric6odeint21step_adjustment_errorEED2Ev.exit

_ZN5boost16exception_detail19error_info_injectorINS_7numeric6odeint21step_adjustment_errorEED2Ev.exit: ; preds = %if.then.i.i.i.i, %land.lhs.true.i.i.i.i, %entry
  %5 = getelementptr inbounds %"struct.boost::exception_detail::error_info_injector", %"struct.boost::exception_detail::error_info_injector"* %this, i64 0, i32 0, i32 0, i32 0
  tail call void @_ZNSt13runtime_errorD2Ev(%"class.std::runtime_error"* %5) #4
  %6 = bitcast %"struct.boost::exception_detail::error_info_injector"* %this to i8*
  tail call void @_ZdlPv(i8* %6) #13
  ret void
}

; Function Attrs: nounwind
declare dso_local i8* @_ZNKSt13runtime_error4whatEv(%"class.std::runtime_error"*) unnamed_addr #2

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZThn16_N5boost16exception_detail19error_info_injectorINS_7numeric6odeint21step_adjustment_errorEED1Ev(%"struct.boost::exception_detail::error_info_injector"* %this) unnamed_addr #3 comdat align 2 {
entry:
  %0 = getelementptr inbounds %"struct.boost::exception_detail::error_info_injector", %"struct.boost::exception_detail::error_info_injector"* %this, i64 -1, i32 1, i32 3
  %1 = getelementptr inbounds i8*, i8** %0, i64 2
  %2 = bitcast i8** %1 to i32 (...)***
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN5boost9exceptionE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %2, align 8, !tbaa !13
  %3 = getelementptr inbounds i8*, i8** %0, i64 3
  %4 = bitcast i8** %3 to %"struct.boost::exception_detail::error_info_container"**
  %5 = load %"struct.boost::exception_detail::error_info_container"*, %"struct.boost::exception_detail::error_info_container"** %4, align 8, !tbaa !15
  %tobool.i.i.i.i = icmp eq %"struct.boost::exception_detail::error_info_container"* %5, null
  br i1 %tobool.i.i.i.i, label %_ZN5boost16exception_detail19error_info_injectorINS_7numeric6odeint21step_adjustment_errorEED2Ev.exit, label %land.lhs.true.i.i.i.i

land.lhs.true.i.i.i.i:                            ; preds = %entry
  %6 = bitcast %"struct.boost::exception_detail::error_info_container"* %5 to i1 (%"struct.boost::exception_detail::error_info_container"*)***
  %vtable.i.i.i.i = load i1 (%"struct.boost::exception_detail::error_info_container"*)**, i1 (%"struct.boost::exception_detail::error_info_container"*)*** %6, align 8, !tbaa !13
  %vfn.i.i.i.i = getelementptr inbounds i1 (%"struct.boost::exception_detail::error_info_container"*)*, i1 (%"struct.boost::exception_detail::error_info_container"*)** %vtable.i.i.i.i, i64 4
  %7 = load i1 (%"struct.boost::exception_detail::error_info_container"*)*, i1 (%"struct.boost::exception_detail::error_info_container"*)** %vfn.i.i.i.i, align 8
  %call.i.i.i.i = tail call zeroext i1 %7(%"struct.boost::exception_detail::error_info_container"* nonnull %5) #4
  br i1 %call.i.i.i.i, label %if.then.i.i.i.i, label %_ZN5boost16exception_detail19error_info_injectorINS_7numeric6odeint21step_adjustment_errorEED2Ev.exit

if.then.i.i.i.i:                                  ; preds = %land.lhs.true.i.i.i.i
  store %"struct.boost::exception_detail::error_info_container"* null, %"struct.boost::exception_detail::error_info_container"** %4, align 8, !tbaa !15
  br label %_ZN5boost16exception_detail19error_info_injectorINS_7numeric6odeint21step_adjustment_errorEED2Ev.exit

_ZN5boost16exception_detail19error_info_injectorINS_7numeric6odeint21step_adjustment_errorEED2Ev.exit: ; preds = %if.then.i.i.i.i, %land.lhs.true.i.i.i.i, %entry
  %8 = bitcast i8** %0 to %"class.std::runtime_error"*
  tail call void @_ZNSt13runtime_errorD2Ev(%"class.std::runtime_error"* nonnull %8) #4
  ret void
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZThn16_N5boost16exception_detail19error_info_injectorINS_7numeric6odeint21step_adjustment_errorEED0Ev(%"struct.boost::exception_detail::error_info_injector"* %this) unnamed_addr #3 comdat align 2 {
entry:
  %0 = getelementptr inbounds %"struct.boost::exception_detail::error_info_injector", %"struct.boost::exception_detail::error_info_injector"* %this, i64 -1, i32 1, i32 3
  %1 = getelementptr inbounds i8*, i8** %0, i64 2
  %2 = bitcast i8** %1 to i32 (...)***
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN5boost9exceptionE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %2, align 8, !tbaa !13
  %3 = getelementptr inbounds i8*, i8** %0, i64 3
  %4 = bitcast i8** %3 to %"struct.boost::exception_detail::error_info_container"**
  %5 = load %"struct.boost::exception_detail::error_info_container"*, %"struct.boost::exception_detail::error_info_container"** %4, align 8, !tbaa !15
  %tobool.i.i.i.i.i = icmp eq %"struct.boost::exception_detail::error_info_container"* %5, null
  br i1 %tobool.i.i.i.i.i, label %_ZN5boost16exception_detail19error_info_injectorINS_7numeric6odeint21step_adjustment_errorEED0Ev.exit, label %land.lhs.true.i.i.i.i.i

land.lhs.true.i.i.i.i.i:                          ; preds = %entry
  %6 = bitcast %"struct.boost::exception_detail::error_info_container"* %5 to i1 (%"struct.boost::exception_detail::error_info_container"*)***
  %vtable.i.i.i.i.i = load i1 (%"struct.boost::exception_detail::error_info_container"*)**, i1 (%"struct.boost::exception_detail::error_info_container"*)*** %6, align 8, !tbaa !13
  %vfn.i.i.i.i.i = getelementptr inbounds i1 (%"struct.boost::exception_detail::error_info_container"*)*, i1 (%"struct.boost::exception_detail::error_info_container"*)** %vtable.i.i.i.i.i, i64 4
  %7 = load i1 (%"struct.boost::exception_detail::error_info_container"*)*, i1 (%"struct.boost::exception_detail::error_info_container"*)** %vfn.i.i.i.i.i, align 8
  %call.i.i.i.i.i = tail call zeroext i1 %7(%"struct.boost::exception_detail::error_info_container"* nonnull %5) #4
  br i1 %call.i.i.i.i.i, label %if.then.i.i.i.i.i, label %_ZN5boost16exception_detail19error_info_injectorINS_7numeric6odeint21step_adjustment_errorEED0Ev.exit

if.then.i.i.i.i.i:                                ; preds = %land.lhs.true.i.i.i.i.i
  store %"struct.boost::exception_detail::error_info_container"* null, %"struct.boost::exception_detail::error_info_container"** %4, align 8, !tbaa !15
  br label %_ZN5boost16exception_detail19error_info_injectorINS_7numeric6odeint21step_adjustment_errorEED0Ev.exit

_ZN5boost16exception_detail19error_info_injectorINS_7numeric6odeint21step_adjustment_errorEED0Ev.exit: ; preds = %if.then.i.i.i.i.i, %land.lhs.true.i.i.i.i.i, %entry
  %8 = bitcast i8** %0 to %"class.std::runtime_error"*
  tail call void @_ZNSt13runtime_errorD2Ev(%"class.std::runtime_error"* nonnull %8) #4
  %9 = bitcast i8** %0 to i8*
  tail call void @_ZdlPv(i8* nonnull %9) #13
  ret void
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local void @_ZN5boost7numeric6odeint21step_adjustment_errorD0Ev(%"class.boost::numeric::odeint::step_adjustment_error"* %this) unnamed_addr #11 comdat align 2 {
entry:
  %0 = getelementptr inbounds %"class.boost::numeric::odeint::step_adjustment_error", %"class.boost::numeric::odeint::step_adjustment_error"* %this, i64 0, i32 0, i32 0
  tail call void @_ZNSt13runtime_errorD2Ev(%"class.std::runtime_error"* %0) #4
  %1 = bitcast %"class.boost::numeric::odeint::step_adjustment_error"* %this to i8*
  tail call void @_ZdlPv(i8* %1) #13
  ret void
}

; Function Attrs: nounwind
declare dso_local void @_ZNSt13runtime_errorC2ERKS_(%"class.std::runtime_error"*, %"class.std::runtime_error"* dereferenceable(16)) unnamed_addr #2

; Function Attrs: nounwind
declare dso_local void @_ZNSt13runtime_errorD2Ev(%"class.std::runtime_error"*) unnamed_addr #2

; Function Attrs: nobuiltin nounwind
declare dso_local void @_ZdlPv(i8*) local_unnamed_addr #12

declare dso_local void @__cxa_pure_virtual() unnamed_addr

declare dso_local void @_ZNSt13runtime_errorC2ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(%"class.std::runtime_error"*, %"class.std::__cxx11::basic_string"* dereferenceable(32)) unnamed_addr #1

; Function Attrs: nounwind uwtable
define internal void @_GLOBAL__sub_I_integrateexp.cpp() #3 section ".text.startup" {
entry:
  tail call void @_ZNSt8ios_base4InitC1Ev(%"class.std::ios_base::Init"* nonnull @_ZStL8__ioinit) #4
  %0 = tail call i32 @atexit(void ()* nonnull @__dtor__ZStL8__ioinit) #4
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1) #7

; Function Attrs: nounwind readnone speculatable
declare <2 x double> @llvm.fabs.v2f64(<2 x double>) #9

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1) #7

attributes #0 = { nounwind readnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #2 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #3 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #4 = { nounwind }
attributes #5 = { norecurse nounwind readnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #6 = { alwaysinline norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #7 = { argmemonly nounwind }
attributes #8 = { norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #9 = { nounwind readnone speculatable }
attributes #10 = { noreturn nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #11 = { inlinehint nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #12 = { nobuiltin nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #13 = { builtin nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0}
!3 = !{!"double", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C++ TBAA"}
!6 = !{!7, !9, i64 16}
!7 = !{!"_ZTSN5boost9exceptionE", !8, i64 8, !9, i64 16, !9, i64 24, !10, i64 32}
!8 = !{!"_ZTSN5boost16exception_detail12refcount_ptrINS0_20error_info_containerEEE", !9, i64 0}
!9 = !{!"any pointer", !4, i64 0}
!10 = !{!"int", !4, i64 0}
!11 = !{!7, !9, i64 24}
!12 = !{!7, !10, i64 32}
!13 = !{!14, !14, i64 0}
!14 = !{!"vtable pointer", !5, i64 0}
!15 = !{!8, !9, i64 0}
