; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -adce -loop-deletion -correlated-propagation -simplifycfg -S | FileCheck %s

; ModuleID = 'ode-unopt.ll'
source_filename = "ode.cpp"
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
%"struct.std::_Placeholder" = type { i8 }
%"class.boost::array.1" = type { [1 x double] }
%struct.timeval = type { i64, i64 }
%struct.timezone = type { i32, i32 }

@_ZStL8__ioinit = internal global %"class.std::ios_base::Init" zeroinitializer, align 1
@_ZN5boost12_GLOBAL__N_17extentsE = internal global %"class.boost::detail::multi_array::extent_gen" zeroinitializer, align 8
@_ZN5boost12_GLOBAL__N_17indicesE = internal global %"struct.boost::detail::multi_array::index_gen" zeroinitializer, align 8
@.str = private unnamed_addr constant [22 x i8] c"calling lorenz at %f\0A\00", align 1
@.str.3 = private unnamed_addr constant [10 x i8] c"iters=%d\0A\00", align 1
@.str.4 = private unnamed_addr constant [26 x i8] c"Enzyme real %0.6f res=%f\0A\00", align 1
@.str.5 = private unnamed_addr constant [29 x i8] c"Enzyme forward %0.6f res=%f\0A\00", align 1
@.str.6 = private unnamed_addr constant [31 x i8] c"Enzyme combined %0.6f res'=%f\0A\00", align 1
@.str.8 = private unnamed_addr constant [26 x i8] c"(i < N)&&(\22out of range\22)\00", align 1
@.str.9 = private unnamed_addr constant [29 x i8] c"/usr/include/boost/array.hpp\00", align 1
@__PRETTY_FUNCTION__._ZN5boost5arrayIdLm1EEixEm = private unnamed_addr constant [105 x i8] c"boost::array::reference boost::array<double, 1>::operator[](boost::array::size_type) [T = double, N = 1]\00", align 1
@__PRETTY_FUNCTION__._ZNK5boost5arrayIdLm1EEixEm = private unnamed_addr constant [117 x i8] c"boost::array::const_reference boost::array<double, 1>::operator[](boost::array::size_type) const [T = double, N = 1]\00", align 1
@_ZNSt12placeholders2_1E = external dso_local global %"struct.std::_Placeholder", align 1
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_ode.cpp, i8* null }]

; Function Attrs: nounwind uwtable
define internal fastcc void @__cxx_global_var_init() unnamed_addr #0 section ".text.startup" {
entry:
  tail call void @_ZNSt8ios_base4InitC1Ev(%"class.std::ios_base::Init"* nonnull @_ZStL8__ioinit) #3
  %0 = tail call i32 @atexit(void ()* nonnull @__dtor__ZStL8__ioinit) #3
  ret void
}

declare dso_local void @_ZNSt8ios_base4InitC1Ev(%"class.std::ios_base::Init"*) unnamed_addr #1

; Function Attrs: nounwind
declare dso_local void @_ZNSt8ios_base4InitD1Ev(%"class.std::ios_base::Init"*) unnamed_addr #2

; Function Attrs: nounwind uwtable
define internal void @__dtor__ZStL8__ioinit() #0 section ".text.startup" {
entry:
  tail call void @_ZNSt8ios_base4InitD1Ev(%"class.std::ios_base::Init"* nonnull @_ZStL8__ioinit)
  ret void
}

; Function Attrs: nounwind
declare dso_local i32 @atexit(void ()*) local_unnamed_addr #3

; Function Attrs: nounwind uwtable
define internal fastcc void @__cxx_global_var_init.1() unnamed_addr #4 section ".text.startup" {
entry:
  store i64 0, i64* getelementptr inbounds (%"class.boost::detail::multi_array::extent_gen", %"class.boost::detail::multi_array::extent_gen"* @_ZN5boost12_GLOBAL__N_17extentsE, i64 0, i32 0, i32 0, i64 0, i32 0, i32 0), align 8, !tbaa !2
  store i64 0, i64* getelementptr inbounds (%"class.boost::detail::multi_array::extent_gen", %"class.boost::detail::multi_array::extent_gen"* @_ZN5boost12_GLOBAL__N_17extentsE, i64 0, i32 0, i32 0, i64 0, i32 0, i32 1), align 8, !tbaa !7
  ret void
}

; Function Attrs: alwaysinline
define dso_local void @_Z6lorenzRKN5boost5arrayIdLm1EEERS1_d(%"class.boost::array.1"* dereferenceable(8) %x, %"class.boost::array.1"* dereferenceable(8) %dxdt, double %t) #5 {
entry:
  %call = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([22 x i8], [22 x i8]* @.str, i64 0, i64 0), double %t)
  %arrayidx.i1 = getelementptr inbounds %"class.boost::array.1", %"class.boost::array.1"* %x, i64 0, i32 0, i64 0
  %0 = load double, double* %arrayidx.i1, align 8, !tbaa !8
  %mul = fmul fast double %0, -1.200000e+00
  %arrayidx.i = getelementptr inbounds %"class.boost::array.1", %"class.boost::array.1"* %dxdt, i64 0, i32 0, i64 0
  store double %mul, double* %arrayidx.i, align 8, !tbaa !8
  ret void
}

; Function Attrs: nounwind
declare dso_local i32 @printf(i8* nocapture readonly, ...) local_unnamed_addr #2

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #6

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #6

; Function Attrs: nounwind uwtable
define dso_local double @_Z6foobardm(double %t, i64 %iters) #4 {
entry:
  %x = alloca %"class.boost::array.1", align 8
  %0 = bitcast %"class.boost::array.1"* %x to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %0) #3
  %1 = bitcast %"class.boost::array.1"* %x to i64*
  store i64 4607182418800017408, i64* %1, align 8
  %conv = uitofp i64 %iters to double
  %div = fdiv fast double %t, %conv
  %arrayidx.i1.i = getelementptr inbounds %"class.boost::array.1", %"class.boost::array.1"* %x, i64 0, i32 0, i64 0
  call void @sub(i64 ptrtoint (void (%"class.boost::array.1"*)* @indir to i64), %"class.boost::array.1"* %x) #3
  %call.i = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([22 x i8], [22 x i8]* @.str, i64 0, i64 0), double 0.000000e+00) #3
  %2 = load double, double* %arrayidx.i1.i, align 8, !tbaa !8
  %mul.i1 = fmul fast double %2, -1.200000e+00
  %mul2.i.i.i = fmul fast double %mul.i1, %div
  %add.i.i.i = fadd fast double %mul2.i.i.i, %2
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %0) #3
  ret double %add.i.i.i
}

; Function Attrs: alwaysinline norecurse nounwind uwtable
define dso_local i32 @main(i32 %argc, i8** nocapture readonly %argv) local_unnamed_addr #7 {
entry:
  %start.i = alloca %struct.timeval, align 8
  %end.i = alloca %struct.timeval, align 8
  %start5.i = alloca %struct.timeval, align 8
  %end6.i = alloca %struct.timeval, align 8
  %start14.i = alloca %struct.timeval, align 8
  %end15.i = alloca %struct.timeval, align 8
  %arrayidx = getelementptr inbounds i8*, i8** %argv, i64 1
  %0 = load i8*, i8** %arrayidx, align 8, !tbaa !10
  %call.i = tail call i64 @strtol(i8* nocapture nonnull %0, i8** null, i32 10) #3
  %conv.i = trunc i64 %call.i to i32
  %div = sdiv i32 %conv.i, 20
  %cmp14 = icmp sgt i32 %div, %conv.i
  br i1 %cmp14, label %for.cond.cleanup, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  %1 = sext i32 %div to i64
  %2 = sext i32 %conv.i to i64
  %3 = bitcast %struct.timeval* %start.i to i8*
  %4 = bitcast %struct.timeval* %end.i to i8*
  %tv_sec.i.i = getelementptr inbounds %struct.timeval, %struct.timeval* %end.i, i64 0, i32 0
  %tv_sec1.i.i = getelementptr inbounds %struct.timeval, %struct.timeval* %start.i, i64 0, i32 0
  %tv_usec.i.i = getelementptr inbounds %struct.timeval, %struct.timeval* %end.i, i64 0, i32 1
  %tv_usec2.i.i = getelementptr inbounds %struct.timeval, %struct.timeval* %start.i, i64 0, i32 1
  %5 = bitcast %struct.timeval* %start5.i to i8*
  %6 = bitcast %struct.timeval* %end6.i to i8*
  %tv_sec.i12.i = getelementptr inbounds %struct.timeval, %struct.timeval* %end6.i, i64 0, i32 0
  %tv_sec1.i13.i = getelementptr inbounds %struct.timeval, %struct.timeval* %start5.i, i64 0, i32 0
  %tv_usec.i16.i = getelementptr inbounds %struct.timeval, %struct.timeval* %end6.i, i64 0, i32 1
  %tv_usec2.i17.i = getelementptr inbounds %struct.timeval, %struct.timeval* %start5.i, i64 0, i32 1
  %7 = bitcast %struct.timeval* %start14.i to i8*
  %8 = bitcast %struct.timeval* %end15.i to i8*
  %tv_sec.i1.i = getelementptr inbounds %struct.timeval, %struct.timeval* %end15.i, i64 0, i32 0
  %tv_sec1.i2.i = getelementptr inbounds %struct.timeval, %struct.timeval* %start14.i, i64 0, i32 0
  %tv_usec.i5.i = getelementptr inbounds %struct.timeval, %struct.timeval* %end15.i, i64 0, i32 1
  %tv_usec2.i6.i = getelementptr inbounds %struct.timeval, %struct.timeval* %start14.i, i64 0, i32 1
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret i32 0

for.body:                                         ; preds = %for.body, %for.body.preheader
  %indvars.iv = phi i64 [ %1, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %9 = trunc i64 %indvars.iv to i32
  %call1 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str.3, i64 0, i64 0), i32 %9)
  tail call void @_Z12adept_sincosdm(double 2.100000e+00, i64 %indvars.iv) #3
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %3) #3
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %4) #3
  %call.i1 = call i32 @gettimeofday(%struct.timeval* nonnull %start.i, %struct.timezone* null) #3
  %call1.i = tail call fast double @_Z6foobardm(double 2.100000e+00, i64 %indvars.iv) #3
  %call2.i = call i32 @gettimeofday(%struct.timeval* nonnull %end.i, %struct.timezone* null) #3
  %10 = load i64, i64* %tv_sec.i.i, align 8, !tbaa !12
  %11 = load i64, i64* %tv_sec1.i.i, align 8, !tbaa !12
  %sub.i.i = sub nsw i64 %10, %11
  %conv.i.i = sitofp i64 %sub.i.i to double
  %12 = load i64, i64* %tv_usec.i.i, align 8, !tbaa !14
  %13 = load i64, i64* %tv_usec2.i.i, align 8, !tbaa !14
  %sub3.i.i = sub nsw i64 %12, %13
  %conv4.i.i = sitofp i64 %sub3.i.i to double
  %mul.i.i = fmul fast double %conv4.i.i, 0x3EB0C6F7A0B5ED8D
  %add.i.i = fadd fast double %mul.i.i, %conv.i.i
  %conv5.i.i = fptrunc double %add.i.i to float
  %conv.i2 = fpext float %conv5.i.i to double
  %call4.i = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([26 x i8], [26 x i8]* @.str.4, i64 0, i64 0), double %conv.i2, double %call1.i) #3
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %4) #3
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %3) #3
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %5) #3
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %6) #3
  %call7.i = call i32 @gettimeofday(%struct.timeval* nonnull %start5.i, %struct.timezone* null) #3
  %call9.i = tail call fast double @_Z6foobardm(double 2.100000e+00, i64 %indvars.iv) #3
  %call10.i = call i32 @gettimeofday(%struct.timeval* nonnull %end6.i, %struct.timezone* null) #3
  %14 = load i64, i64* %tv_sec.i12.i, align 8, !tbaa !12
  %15 = load i64, i64* %tv_sec1.i13.i, align 8, !tbaa !12
  %sub.i14.i = sub nsw i64 %14, %15
  %conv.i15.i = sitofp i64 %sub.i14.i to double
  %16 = load i64, i64* %tv_usec.i16.i, align 8, !tbaa !14
  %17 = load i64, i64* %tv_usec2.i17.i, align 8, !tbaa !14
  %sub3.i18.i = sub nsw i64 %16, %17
  %conv4.i19.i = sitofp i64 %sub3.i18.i to double
  %mul.i20.i = fmul fast double %conv4.i19.i, 0x3EB0C6F7A0B5ED8D
  %add.i21.i = fadd fast double %mul.i20.i, %conv.i15.i
  %conv5.i22.i = fptrunc double %add.i21.i to float
  %conv12.i = fpext float %conv5.i22.i to double
  %call13.i = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([29 x i8], [29 x i8]* @.str.5, i64 0, i64 0), double %conv12.i, double %call9.i) #3
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %6) #3
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %5) #3
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %7) #3
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %8) #3
  %call16.i = call i32 @gettimeofday(%struct.timeval* nonnull %start14.i, %struct.timezone* null) #3
  %call17.i = tail call fast double @_Z17__enzyme_autodiffIdJPFddmEdmEET_DpT0_(double (double, i64)* nonnull @_Z6foobardm, double 2.100000e+00, i64 %indvars.iv) #3
  %call18.i = call i32 @gettimeofday(%struct.timeval* nonnull %end15.i, %struct.timezone* null) #3
  %18 = load i64, i64* %tv_sec.i1.i, align 8, !tbaa !12
  %19 = load i64, i64* %tv_sec1.i2.i, align 8, !tbaa !12
  %sub.i3.i = sub nsw i64 %18, %19
  %conv.i4.i = sitofp i64 %sub.i3.i to double
  %20 = load i64, i64* %tv_usec.i5.i, align 8, !tbaa !14
  %21 = load i64, i64* %tv_usec2.i6.i, align 8, !tbaa !14
  %sub3.i7.i = sub nsw i64 %20, %21
  %conv4.i8.i = sitofp i64 %sub3.i7.i to double
  %mul.i9.i = fmul fast double %conv4.i8.i, 0x3EB0C6F7A0B5ED8D
  %add.i10.i = fadd fast double %mul.i9.i, %conv.i4.i
  %conv5.i11.i = fptrunc double %add.i10.i to float
  %conv20.i = fpext float %conv5.i11.i to double
  %call21.i = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([31 x i8], [31 x i8]* @.str.6, i64 0, i64 0), double %conv20.i, double %call17.i) #3
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %8) #3
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %7) #3
  %indvars.iv.next = add i64 %indvars.iv, %1
  %cmp = icmp sgt i64 %indvars.iv.next, %2
  br i1 %cmp, label %for.cond.cleanup, label %for.body
}

declare dso_local void @_Z12adept_sincosdm(double, i64) local_unnamed_addr #1

; Function Attrs: nounwind
declare dso_local i64 @strtol(i8* readonly, i8** nocapture, i32) local_unnamed_addr #2

; Function Attrs: nounwind
declare dso_local i32 @gettimeofday(%struct.timeval* nocapture, %struct.timezone* nocapture) local_unnamed_addr #2

declare dso_local double @_Z17__enzyme_autodiffIdJPFddmEdmEET_DpT0_(double (double, i64)*, double, i64) local_unnamed_addr #1

; Function Attrs: noreturn nounwind
declare dso_local void @__assert_fail(i8*, i8*, i32, i8*) local_unnamed_addr #8

define void @indir(%"class.boost::array.1"* dereferenceable(8) %x) {
entry:
  ret void
}

define linkonce_odr dso_local void @sub(i64 %.unpack.i, %"class.boost::array.1"* %Arg) {
entry:
  %fn = inttoptr i64 %.unpack.i to void (%"class.boost::array.1"*)*
  call void %fn(%"class.boost::array.1"* nonnull dereferenceable(8) %Arg) #3
  ret void
}

; Function Attrs: nounwind uwtable
define internal void @_GLOBAL__sub_I_ode.cpp() #0 section ".text.startup" {
entry:
  tail call fastcc void @__cxx_global_var_init()
  tail call fastcc void @__cxx_global_var_init.1()
  store i64 -9223372036854775808, i64* getelementptr inbounds (%"struct.boost::detail::multi_array::index_gen", %"struct.boost::detail::multi_array::index_gen"* @_ZN5boost12_GLOBAL__N_17indicesE, i64 0, i32 0, i32 0, i64 0, i32 0), align 8, !tbaa !15
  store i64 9223372036854775807, i64* getelementptr inbounds (%"struct.boost::detail::multi_array::index_gen", %"struct.boost::detail::multi_array::index_gen"* @_ZN5boost12_GLOBAL__N_17indicesE, i64 0, i32 0, i32 0, i64 0, i32 1), align 8, !tbaa !18
  store i64 1, i64* getelementptr inbounds (%"struct.boost::detail::multi_array::index_gen", %"struct.boost::detail::multi_array::index_gen"* @_ZN5boost12_GLOBAL__N_17indicesE, i64 0, i32 0, i32 0, i64 0, i32 2), align 8, !tbaa !19
  store i8 0, i8* getelementptr inbounds (%"struct.boost::detail::multi_array::index_gen", %"struct.boost::detail::multi_array::index_gen"* @_ZN5boost12_GLOBAL__N_17indicesE, i64 0, i32 0, i32 0, i64 0, i32 3), align 8, !tbaa !20
  ret void
}

attributes #0 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #2 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #3 = { nounwind }
attributes #4 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { alwaysinline }
attributes #6 = { argmemonly nounwind }
attributes #7 = { alwaysinline norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #8 = { noreturn nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 8.0.1 "}
!2 = !{!3, !4, i64 0}
!3 = !{!"_ZTSSt4pairIllE", !4, i64 0, !4, i64 8}
!4 = !{!"long", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}
!7 = !{!3, !4, i64 8}
!8 = !{!9, !9, i64 0}
!9 = !{!"double", !5, i64 0}
!10 = !{!11, !11, i64 0}
!11 = !{!"any pointer", !5, i64 0}
!12 = !{!13, !4, i64 0}
!13 = !{!"_ZTS7timeval", !4, i64 0, !4, i64 8}
!14 = !{!13, !4, i64 8}
!15 = !{!16, !4, i64 0}
!16 = !{!"_ZTSN5boost6detail11multi_array11index_rangeIlmEE", !4, i64 0, !4, i64 8, !4, i64 16, !17, i64 24}
!17 = !{!"bool", !5, i64 0}
!18 = !{!16, !4, i64 8}
!19 = !{!16, !4, i64 16}
!20 = !{!16, !17, i64 24}

; CHECK: define internal { double } @diffe_Z6foobardm(double %t, i64 %iters, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"x'ipa" = alloca %"class.boost::array.1", align 8
; CHECK-NEXT:   store %"class.boost::array.1" zeroinitializer, %"class.boost::array.1"* %"x'ipa", align 8
; CHECK-NEXT:   %x = alloca %"class.boost::array.1", align 8
; CHECK-NEXT:   %"'ipc" = bitcast %"class.boost::array.1"* %"x'ipa" to i64*
; CHECK-NEXT:   %0 = bitcast %"class.boost::array.1"* %x to i64*
; CHECK-NEXT:   store i64 4607182418800017408, i64* %0, align 8
; CHECK-NEXT:   %conv = uitofp i64 %iters to double
; CHECK-NEXT:   %div = fdiv fast double %t, %conv
; CHECK-NEXT:   %"arrayidx.i1.i'ipg" = getelementptr inbounds %"class.boost::array.1", %"class.boost::array.1"* %"x'ipa", i64 0, i32 0, i64 0
; CHECK-NEXT:   %arrayidx.i1.i = getelementptr inbounds %"class.boost::array.1", %"class.boost::array.1"* %x, i64 0, i32 0, i64 0
; CHECK-NEXT:   %_augmented = call i8* @augmented_sub(i64 ptrtoint (void (%"class.boost::array.1"*)* @indir to i64), i64 ptrtoint ({ i8* (%"class.boost::array.1"*, %"class.boost::array.1"*)*, void (%"class.boost::array.1"*, %"class.boost::array.1"*, i8*)* }* @"_enzyme_reverse_indir'" to i64), %"class.boost::array.1"* nonnull %x, %"class.boost::array.1"* nonnull %"x'ipa")
; CHECK-NEXT:   %call.i = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([22 x i8], [22 x i8]* @.str, i64 0, i64 0), double 0.000000e+00)
; CHECK-NEXT:   %1 = load double, double* %arrayidx.i1.i, align 8, !tbaa !8
; CHECK-NEXT:   %mul.i1 = fmul fast double %1, -1.200000e+00
; CHECK-NEXT:   %m0diffemul.i1 = fmul fast double %differeturn, %div
; CHECK-NEXT:   %m1diffediv = fmul fast double %differeturn, %mul.i1
; CHECK-NEXT:   %m0diffe = fmul fast double %m0diffemul.i1, -1.200000e+00
; CHECK-NEXT:   %2 = fadd fast double %differeturn, %m0diffe
; CHECK-NEXT:   %3 = load double, double* %"arrayidx.i1.i'ipg", align 8
; CHECK-NEXT:   %4 = fadd fast double %3, %2
; CHECK-NEXT:   store double %4, double* %"arrayidx.i1.i'ipg", align 8
; CHECK-NEXT:   call void @diffesub(i64 ptrtoint (void (%"class.boost::array.1"*)* @indir to i64), i64 ptrtoint ({ i8* (%"class.boost::array.1"*, %"class.boost::array.1"*)*, void (%"class.boost::array.1"*, %"class.boost::array.1"*, i8*)* }* @"_enzyme_reverse_indir'" to i64), %"class.boost::array.1"* nonnull %x, %"class.boost::array.1"* nonnull %"x'ipa", i8* %_augmented)
; CHECK-NEXT:   %d0diffet = fdiv fast double %m1diffediv, %conv
; CHECK-NEXT:   store i64 0, i64* %"'ipc", align 8
; CHECK-NEXT:   %5 = insertvalue { double } undef, double %d0diffet, 0
; CHECK-NEXT:   ret { double } %5
; CHECK-NEXT: }

; TODO no need for malloc/free
; CHECK: define internal i8* @augmented_indir(%"class.boost::array.1"* dereferenceable(8) %x, %"class.boost::array.1"* %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret i8* null
; CHECK-NEXT: }

; CHECK: define internal void @diffeindir(%"class.boost::array.1"* dereferenceable(8) %x, %"class.boost::array.1"* %"x'", i8* %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   tail call void @free(i8* nonnull %tapeArg)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal i8* @augmented_sub(i64 %.unpack.i, i64 %".unpack.i'", %"class.boost::array.1"* %Arg, %"class.boost::array.1"* %"Arg'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"fn'ipc" = inttoptr i64 %".unpack.i'" to void (%"class.boost::array.1"*)*
; CHECK-NEXT:   %fn = inttoptr i64 %.unpack.i to void (%"class.boost::array.1"*)*
; CHECK-NEXT:   %0 = bitcast void (%"class.boost::array.1"*)* %fn to i8*
; CHECK-NEXT:   %1 = bitcast void (%"class.boost::array.1"*)* %"fn'ipc" to i8*
; CHECK-NEXT:   %2 = icmp eq i8* %0, %1
; CHECK-NEXT:   br i1 %2, label %error.i, label %__enzyme_runtimeinactiveerr.exit

; CHECK: error.i:                                          ; preds = %entry
; CHECK-NEXT:   %3 = call i32 @puts(i8* getelementptr inbounds ([79 x i8], [79 x i8]* @.str.1, i32 0, i32 0))
; CHECK-NEXT:   call void @exit(i32 1)
; CHECK-NEXT:   unreachable

; CHECK: __enzyme_runtimeinactiveerr.exit:                 ; preds = %entry
; CHECK-NEXT:   %4 = bitcast void (%"class.boost::array.1"*)* %"fn'ipc" to { i8* } (%"class.boost::array.1"*, %"class.boost::array.1"*)**
; CHECK-NEXT:   %5 = load { i8* } (%"class.boost::array.1"*, %"class.boost::array.1"*)*, { i8* } (%"class.boost::array.1"*, %"class.boost::array.1"*)** %4
; CHECK-NEXT:   %_augmented = call { i8* } %5(%"class.boost::array.1"* %Arg, %"class.boost::array.1"* %"Arg'")
; CHECK-NEXT:   %subcache = extractvalue { i8* } %_augmented, 0
; CHECK-NEXT:   ret i8* %subcache
; CHECK-NEXT: }

; CHECK: define internal void @diffesub(i64 %.unpack.i, i64 %".unpack.i'", %"class.boost::array.1"* %Arg, %"class.boost::array.1"* %"Arg'", i8* %tapeArg1)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"fn'ipc" = inttoptr i64 %".unpack.i'" to void (%"class.boost::array.1"*)*
; CHECK-NEXT:   %0 = bitcast void (%"class.boost::array.1"*)* %"fn'ipc" to {} (%"class.boost::array.1"*, %"class.boost::array.1"*, i8*)**
; CHECK-NEXT:   %1 = getelementptr {} (%"class.boost::array.1"*, %"class.boost::array.1"*, i8*)*, {} (%"class.boost::array.1"*, %"class.boost::array.1"*, i8*)** %0, i64 1
; CHECK-NEXT:   %2 = load {} (%"class.boost::array.1"*, %"class.boost::array.1"*, i8*)*, {} (%"class.boost::array.1"*, %"class.boost::array.1"*, i8*)** %1
; CHECK-NEXT:   %3 = call {} %2(%"class.boost::array.1"* %Arg, %"class.boost::array.1"* %"Arg'", i8* %tapeArg1)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
