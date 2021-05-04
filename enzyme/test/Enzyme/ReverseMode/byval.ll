; RUN: if [ %llvmver -ge 9 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -adce -loop-deletion -correlated-propagation -simplifycfg -S | FileCheck %s; fi

; ModuleID = 'linked.ll'
source_filename = "llvm-link"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%"class.std::ios_base::Init" = type { i8 }
%"class.std::basic_ostream" = type { i32 (...)**, %"class.std::basic_ios" }
%"class.std::basic_ios" = type { %"class.std::ios_base", %"class.std::basic_ostream"*, i8, i8, %"class.std::basic_streambuf"*, %"class.std::ctype"*, %"class.std::num_put"*, %"class.std::num_put"* }
%"class.std::ios_base" = type { i32 (...)**, i64, i64, i32, i32, i32, %"struct.std::ios_base::_Callback_list"*, %"struct.std::ios_base::_Words", [8 x %"struct.std::ios_base::_Words"], i32, %"struct.std::ios_base::_Words"*, %"class.std::locale" }
%"struct.std::ios_base::_Callback_list" = type { %"struct.std::ios_base::_Callback_list"*, void (i32, %"class.std::ios_base"*, i32)*, i32, i32 }
%"struct.std::ios_base::_Words" = type { i8*, i64 }
%"class.std::locale" = type { %"class.std::locale::_Impl"* }
%"class.std::locale::_Impl" = type { i32, %"class.std::locale::facet"**, i64, %"class.std::locale::facet"**, i8** }
%"class.std::locale::facet" = type <{ i32 (...)**, i32, [4 x i8] }>
%"class.std::basic_streambuf" = type { i32 (...)**, i8*, i8*, i8*, i8*, i8*, i8*, %"class.std::locale" }
%"class.std::ctype" = type <{ %"class.std::locale::facet.base", [4 x i8], %struct.__locale_struct*, i8, [7 x i8], i32*, i32*, i16*, i8, [256 x i8], [256 x i8], i8, [6 x i8] }>
%"class.std::locale::facet.base" = type <{ i32 (...)**, i32 }>
%struct.__locale_struct = type { [13 x %struct.__locale_data*], i16*, i32*, i32*, [13 x i8*] }
%struct.__locale_data = type opaque
%"class.std::num_put" = type { %"class.std::locale::facet.base", [4 x i8] }
%class.twonum = type { double, double }
%class.car = type { double, double, double }

@llvm.global_ctors = appending global [2 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_liba.cpp, i8* null }, { i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_main.cpp, i8* null }]
@_ZStL8__ioinit = internal global %"class.std::ios_base::Init" zeroinitializer, align 1
@__dso_handle = external hidden global i8
@_ZSt4cout = external dso_local global %"class.std::basic_ostream", align 8
@.str = private unnamed_addr constant [2 x i8] c" \00", align 1
@enzyme_dup = dso_local local_unnamed_addr global i32 0, align 4
@enzyme_out = dso_local local_unnamed_addr global i32 0, align 4
@enzyme_const = dso_local local_unnamed_addr global i32 0, align 4
@_ZStL8__ioinit.2 = internal global %"class.std::ios_base::Init" zeroinitializer, align 1
@.str.3 = private unnamed_addr constant [7 x i8] c"speed \00", align 1
@.str.1 = private unnamed_addr constant [5 x i8] c"erg \00", align 1
@.str.2 = private unnamed_addr constant [15 x i8] c"..............\00", align 1
@.str.3.4 = private unnamed_addr constant [22 x i8] c"d/dpos erg at pos=3: \00", align 1

@_ZN6twonumC1Edd = dso_local unnamed_addr alias void (%class.twonum*, double, double), void (%class.twonum*, double, double)* @_ZN6twonumC2Edd
@_ZN3carC1Eddd = dso_local unnamed_addr alias void (%class.car*, double, double, double), void (%class.car*, double, double, double)* @_ZN3carC2Eddd

; Function Attrs: uwtable
define internal void @_GLOBAL__sub_I_liba.cpp() #0 section ".text.startup" {
entry:
  tail call void @_ZNSt8ios_base4InitC1Ev(%"class.std::ios_base::Init"* nonnull @_ZStL8__ioinit)
  %0 = tail call i32 @__cxa_atexit(void (i8*)* bitcast (void (%"class.std::ios_base::Init"*)* @_ZNSt8ios_base4InitD1Ev to void (i8*)*), i8* getelementptr inbounds (%"class.std::ios_base::Init", %"class.std::ios_base::Init"* @_ZStL8__ioinit, i64 0, i32 0), i8* nonnull @__dso_handle) #10
  ret void
}

declare dso_local void @_ZNSt8ios_base4InitC1Ev(%"class.std::ios_base::Init"*) unnamed_addr #1

; Function Attrs: nounwind
declare dso_local void @_ZNSt8ios_base4InitD1Ev(%"class.std::ios_base::Init"*) unnamed_addr #2

; Function Attrs: nofree nounwind
declare dso_local i32 @__cxa_atexit(void (i8*)*, i8*, i8*) local_unnamed_addr #3

; Function Attrs: nofree norecurse nounwind uwtable writeonly
define dso_local void @_ZN6twonumC2Edd(%class.twonum* nocapture %this, double %n1, double %n2) unnamed_addr #4 align 2 {
entry:
  %num1 = getelementptr inbounds %class.twonum, %class.twonum* %this, i64 0, i32 0
  store double %n1, double* %num1, align 8, !tbaa !2
  %num2 = getelementptr inbounds %class.twonum, %class.twonum* %this, i64 0, i32 1
  store double %n2, double* %num2, align 8, !tbaa !7
  ret void
}

; Function Attrs: norecurse nounwind readonly uwtable
define dso_local double @_ZN6twonum7sumwithEd(%class.twonum* nocapture readonly %this, double %num3) local_unnamed_addr #5 align 2 {
entry:
  %num1 = getelementptr inbounds %class.twonum, %class.twonum* %this, i64 0, i32 0
  %0 = load double, double* %num1, align 8, !tbaa !2
  %num2 = getelementptr inbounds %class.twonum, %class.twonum* %this, i64 0, i32 1
  %1 = load double, double* %num2, align 8, !tbaa !7
  %add = fadd fast double %0, %num3
  %add2 = fadd fast double %add, %1
  ret double %add2
}

; Function Attrs: norecurse nounwind readonly uwtable
define dso_local double @_ZN6twonum11sum_sqr_allEv(%class.twonum* nocapture readonly %this) local_unnamed_addr #5 align 2 {
entry:
  %num1 = getelementptr inbounds %class.twonum, %class.twonum* %this, i64 0, i32 0
  %0 = load double, double* %num1, align 8, !tbaa !2
  %mul = fmul fast double %0, %0
  %num2 = getelementptr inbounds %class.twonum, %class.twonum* %this, i64 0, i32 1
  %1 = load double, double* %num2, align 8, !tbaa !7
  %mul4 = fmul fast double %1, %1
  %add = fadd fast double %mul4, %mul
  ret double %add
}

; Function Attrs: nofree norecurse nounwind uwtable writeonly
define dso_local void @_ZN6twonum10changenum2Ed(%class.twonum* nocapture %this, double %n) local_unnamed_addr #4 align 2 {
entry:
  %num2 = getelementptr inbounds %class.twonum, %class.twonum* %this, i64 0, i32 1
  store double %n, double* %num2, align 8, !tbaa !7
  ret void
}

; Function Attrs: uwtable
define dso_local void @_ZN6twonum5printEv(%class.twonum* nocapture readonly %this) local_unnamed_addr #0 align 2 {
entry:
  %num1 = getelementptr inbounds %class.twonum, %class.twonum* %this, i64 0, i32 0
  %0 = load double, double* %num1, align 8, !tbaa !2
  %call.i = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo9_M_insertIdEERSoT_(%"class.std::basic_ostream"* nonnull @_ZSt4cout, double %0)
  %call1.i = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* nonnull dereferenceable(272) %call.i, i8* nonnull getelementptr inbounds ([2 x i8], [2 x i8]* @.str, i64 0, i64 0), i64 1)
  %num2 = getelementptr inbounds %class.twonum, %class.twonum* %this, i64 0, i32 1
  %1 = load double, double* %num2, align 8, !tbaa !7
  %call.i5 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo9_M_insertIdEERSoT_(%"class.std::basic_ostream"* nonnull %call.i, double %1)
  %2 = bitcast %"class.std::basic_ostream"* %call.i5 to i8**
  %vtable.i = load i8*, i8** %2, align 8, !tbaa !8
  %vbase.offset.ptr.i = getelementptr i8, i8* %vtable.i, i64 -24
  %3 = bitcast i8* %vbase.offset.ptr.i to i64*
  %vbase.offset.i = load i64, i64* %3, align 8
  %4 = bitcast %"class.std::basic_ostream"* %call.i5 to i8*
  %add.ptr.i = getelementptr inbounds i8, i8* %4, i64 %vbase.offset.i
  %_M_ctype.i = getelementptr inbounds i8, i8* %add.ptr.i, i64 240
  %5 = bitcast i8* %_M_ctype.i to %"class.std::ctype"**
  %6 = load %"class.std::ctype"*, %"class.std::ctype"** %5, align 8, !tbaa !10
  %tobool.i13 = icmp eq %"class.std::ctype"* %6, null
  br i1 %tobool.i13, label %if.then.i14, label %_ZSt13__check_facetISt5ctypeIcEERKT_PS3_.exit

if.then.i14:                                      ; preds = %entry
  tail call void @_ZSt16__throw_bad_castv() #11
  unreachable

_ZSt13__check_facetISt5ctypeIcEERKT_PS3_.exit:    ; preds = %entry
  %_M_widen_ok.i = getelementptr inbounds %"class.std::ctype", %"class.std::ctype"* %6, i64 0, i32 8
  %7 = load i8, i8* %_M_widen_ok.i, align 8, !tbaa !14
  %tobool.i = icmp eq i8 %7, 0
  br i1 %tobool.i, label %if.end.i, label %if.then.i

if.then.i:                                        ; preds = %_ZSt13__check_facetISt5ctypeIcEERKT_PS3_.exit
  %arrayidx.i = getelementptr inbounds %"class.std::ctype", %"class.std::ctype"* %6, i64 0, i32 9, i64 10
  %8 = load i8, i8* %arrayidx.i, align 1, !tbaa !16
  br label %_ZNKSt5ctypeIcE5widenEc.exit

if.end.i:                                         ; preds = %_ZSt13__check_facetISt5ctypeIcEERKT_PS3_.exit
  tail call void @_ZNKSt5ctypeIcE13_M_widen_initEv(%"class.std::ctype"* nonnull %6)
  %9 = bitcast %"class.std::ctype"* %6 to i8 (%"class.std::ctype"*, i8)***
  %vtable.i11 = load i8 (%"class.std::ctype"*, i8)**, i8 (%"class.std::ctype"*, i8)*** %9, align 8, !tbaa !8
  %vfn.i = getelementptr inbounds i8 (%"class.std::ctype"*, i8)*, i8 (%"class.std::ctype"*, i8)** %vtable.i11, i64 6
  %10 = load i8 (%"class.std::ctype"*, i8)*, i8 (%"class.std::ctype"*, i8)** %vfn.i, align 8
  %call.i12 = tail call signext i8 %10(%"class.std::ctype"* nonnull %6, i8 signext 10)
  br label %_ZNKSt5ctypeIcE5widenEc.exit

_ZNKSt5ctypeIcE5widenEc.exit:                     ; preds = %if.end.i, %if.then.i
  %retval.0.i = phi i8 [ %8, %if.then.i ], [ %call.i12, %if.end.i ]
  %call1.i8 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo3putEc(%"class.std::basic_ostream"* nonnull %call.i5, i8 signext %retval.0.i)
  %call.i9 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo5flushEv(%"class.std::basic_ostream"* nonnull %call1.i8)
  ret void
}

declare dso_local dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo9_M_insertIdEERSoT_(%"class.std::basic_ostream"*, double) local_unnamed_addr #1

declare dso_local dereferenceable(272) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* dereferenceable(272), i8*, i64) local_unnamed_addr #1

; Function Attrs: noreturn
declare dso_local void @_ZSt16__throw_bad_castv() local_unnamed_addr #6

declare dso_local void @_ZNKSt5ctypeIcE13_M_widen_initEv(%"class.std::ctype"*) local_unnamed_addr #1

declare dso_local dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo3putEc(%"class.std::basic_ostream"*, i8 signext) local_unnamed_addr #1

declare dso_local dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo5flushEv(%"class.std::basic_ostream"*) local_unnamed_addr #1

; Function Attrs: nofree norecurse nounwind uwtable writeonly
define dso_local void @_ZN3carC2Eddd(%class.car* nocapture %this, double %x, double %xdot, double %xddot) unnamed_addr #4 align 2 {
entry:
  %pos = getelementptr inbounds %class.car, %class.car* %this, i64 0, i32 0
  store double %x, double* %pos, align 8, !tbaa !17
  %speed = getelementptr inbounds %class.car, %class.car* %this, i64 0, i32 1
  store double %xdot, double* %speed, align 8, !tbaa !19
  %accel = getelementptr inbounds %class.car, %class.car* %this, i64 0, i32 2
  store double %xddot, double* %accel, align 8, !tbaa !20
  ret void
}

; Function Attrs: norecurse nounwind readonly uwtable
define dso_local double @_ZN3car9get_speedEv(%class.car* nocapture readonly %this) local_unnamed_addr #5 align 2 {
entry:
  %speed = getelementptr inbounds %class.car, %class.car* %this, i64 0, i32 1
  %0 = load double, double* %speed, align 8, !tbaa !19
  ret double %0
}

; Function Attrs: norecurse nounwind readonly uwtable
define dso_local double @_ZN3car7get_posEv(%class.car* nocapture readonly %this) local_unnamed_addr #5 align 2 {
entry:
  %pos = getelementptr inbounds %class.car, %class.car* %this, i64 0, i32 0
  %0 = load double, double* %pos, align 8, !tbaa !17
  ret double %0
}

; Function Attrs: norecurse nounwind readonly uwtable
define dso_local double @_ZN3car9get_accelEv(%class.car* nocapture readonly %this) local_unnamed_addr #5 align 2 {
entry:
  %accel = getelementptr inbounds %class.car, %class.car* %this, i64 0, i32 2
  %0 = load double, double* %accel, align 8, !tbaa !20
  ret double %0
}

; Function Attrs: nofree norecurse nounwind uwtable writeonly
define dso_local void @_ZN3car7set_posEd(%class.car* nocapture %this, double %pos1) local_unnamed_addr #4 align 2 {
entry:
  %pos = getelementptr inbounds %class.car, %class.car* %this, i64 0, i32 0
  store double %pos1, double* %pos, align 8, !tbaa !17
  ret void
}

; Function Attrs: uwtable
define internal void @_GLOBAL__sub_I_main.cpp() #0 section ".text.startup" {
entry:
  tail call void @_ZNSt8ios_base4InitC1Ev(%"class.std::ios_base::Init"* nonnull @_ZStL8__ioinit.2)
  %0 = tail call i32 @__cxa_atexit(void (i8*)* bitcast (void (%"class.std::ios_base::Init"*)* @_ZNSt8ios_base4InitD1Ev to void (i8*)*), i8* getelementptr inbounds (%"class.std::ios_base::Init", %"class.std::ios_base::Init"* @_ZStL8__ioinit.2, i64 0, i32 0), i8* nonnull @__dso_handle) #10
  ret void
}

; Function Attrs: uwtable
define dso_local double @_Z13car_erg_atpos3card(%class.car* byval(%class.car) align 8 %car1, double %pos) #0 {
entry:
  call void @_ZN3car7set_posEd(%class.car* nonnull %car1, double %pos)
  %call = call fast double @_ZN3car9get_speedEv(%class.car* nonnull %car1)
  %mul = fmul fast double %call, 5.000000e-01
  %call1 = call fast double @_ZN3car9get_speedEv(%class.car* nonnull %car1)
  %mul2 = fmul fast double %mul, %call1
  %call3 = call fast double @_ZN3car7get_posEv(%class.car* nonnull %car1)
  %mul4 = fmul fast double %call3, 1.000000e+01
  %add = fadd fast double %mul4, %mul2
  ret double %add
}

; Function Attrs: nounwind uwtable
define dso_local double @_Z14dcar_erg_atpos3card(%class.car* nocapture readonly byval(%class.car) align 8 %car1, double %pos) local_unnamed_addr #7 {
entry:
  %0 = load i32, i32* @enzyme_const, align 4, !tbaa !21
  %1 = load i32, i32* @enzyme_out, align 4, !tbaa !21
  %call = tail call fast double @_Z17__enzyme_autodiffIdJi3caridEET_PvDpT0_(i8* bitcast (double (%class.car*, double)* @_Z13car_erg_atpos3card to i8*), i32 %0, %class.car* nonnull byval(%class.car) align 8 %car1, i32 %1, double %pos) #10
  ret double %call
}

; Function Attrs: nounwind
declare dso_local double @_Z17__enzyme_autodiffIdJi3caridEET_PvDpT0_(i8*, i32, %class.car* byval(%class.car) align 8, i32, double) local_unnamed_addr #2

; Function Attrs: norecurse uwtable
define dso_local i32 @main() local_unnamed_addr #8 {
entry:
  %agg.tmp21 = alloca %class.car, align 8
  %mycar = alloca %class.car, align 8
  %0 = bitcast %class.car* %mycar to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %0) #10
  call void @_ZN3carC1Eddd(%class.car* nonnull %mycar, double 1.000000e+00, double 2.000000e+00, double 3.000000e+00)
  %call1.i = call dereferenceable(272) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* nonnull dereferenceable(272) @_ZSt4cout, i8* nonnull getelementptr inbounds ([7 x i8], [7 x i8]* @.str.3, i64 0, i64 0), i64 6)
  %call1 = call fast double @_ZN3car9get_speedEv(%class.car* nonnull %mycar)
  %call.i = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo9_M_insertIdEERSoT_(%"class.std::basic_ostream"* nonnull @_ZSt4cout, double %call1)
  %1 = bitcast %"class.std::basic_ostream"* %call.i to i8**
  %vtable.i = load i8*, i8** %1, align 8, !tbaa !8
  %vbase.offset.ptr.i = getelementptr i8, i8* %vtable.i, i64 -24
  %2 = bitcast i8* %vbase.offset.ptr.i to i64*
  %vbase.offset.i = load i64, i64* %2, align 8
  %3 = bitcast %"class.std::basic_ostream"* %call.i to i8*
  %add.ptr.i = getelementptr inbounds i8, i8* %3, i64 %vbase.offset.i
  %_M_ctype.i = getelementptr inbounds i8, i8* %add.ptr.i, i64 240
  %4 = bitcast i8* %_M_ctype.i to %"class.std::ctype"**
  %5 = load %"class.std::ctype"*, %"class.std::ctype"** %4, align 8, !tbaa !10
  %tobool.i95 = icmp eq %"class.std::ctype"* %5, null
  br i1 %tobool.i95, label %if.then.i96, label %_ZSt13__check_facetISt5ctypeIcEERKT_PS3_.exit

if.then.i96:                                      ; preds = %entry
  call void @_ZSt16__throw_bad_castv() #11
  unreachable

_ZSt13__check_facetISt5ctypeIcEERKT_PS3_.exit:    ; preds = %entry
  %_M_widen_ok.i = getelementptr inbounds %"class.std::ctype", %"class.std::ctype"* %5, i64 0, i32 8
  %6 = load i8, i8* %_M_widen_ok.i, align 8, !tbaa !14
  %tobool.i = icmp eq i8 %6, 0
  br i1 %tobool.i, label %if.end.i, label %if.then.i

if.then.i:                                        ; preds = %_ZSt13__check_facetISt5ctypeIcEERKT_PS3_.exit
  %arrayidx.i = getelementptr inbounds %"class.std::ctype", %"class.std::ctype"* %5, i64 0, i32 9, i64 10
  %7 = load i8, i8* %arrayidx.i, align 1, !tbaa !16
  br label %_ZNKSt5ctypeIcE5widenEc.exit

if.end.i:                                         ; preds = %_ZSt13__check_facetISt5ctypeIcEERKT_PS3_.exit
  call void @_ZNKSt5ctypeIcE13_M_widen_initEv(%"class.std::ctype"* nonnull %5)
  %8 = bitcast %"class.std::ctype"* %5 to i8 (%"class.std::ctype"*, i8)***
  %vtable.i57 = load i8 (%"class.std::ctype"*, i8)**, i8 (%"class.std::ctype"*, i8)*** %8, align 8, !tbaa !8
  %vfn.i = getelementptr inbounds i8 (%"class.std::ctype"*, i8)*, i8 (%"class.std::ctype"*, i8)** %vtable.i57, i64 6
  %9 = load i8 (%"class.std::ctype"*, i8)*, i8 (%"class.std::ctype"*, i8)** %vfn.i, align 8
  %call.i58 = call signext i8 %9(%"class.std::ctype"* nonnull %5, i8 signext 10)
  br label %_ZNKSt5ctypeIcE5widenEc.exit

_ZNKSt5ctypeIcE5widenEc.exit:                     ; preds = %if.end.i, %if.then.i
  %retval.0.i = phi i8 [ %7, %if.then.i ], [ %call.i58, %if.end.i ]
  %call1.i17 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo3putEc(%"class.std::basic_ostream"* nonnull %call.i, i8 signext %retval.0.i)
  %call.i18 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo5flushEv(%"class.std::basic_ostream"* nonnull %call1.i17)
  %call1.i20 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* nonnull dereferenceable(272) @_ZSt4cout, i8* nonnull getelementptr inbounds ([5 x i8], [5 x i8]* @.str.1, i64 0, i64 0), i64 4)
  %10 = bitcast %class.car* %agg.tmp21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %10)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 %10, i8* nonnull align 8 %0, i64 24, i1 false)
  call void @_ZN3car7set_posEd(%class.car* nonnull %agg.tmp21, double 3.000000e+00)
  %call.i22 = call fast double @_ZN3car9get_speedEv(%class.car* nonnull %agg.tmp21)
  %mul.i = fmul fast double %call.i22, 5.000000e-01
  %call1.i23 = call fast double @_ZN3car9get_speedEv(%class.car* nonnull %agg.tmp21)
  %mul2.i = fmul fast double %mul.i, %call1.i23
  %call3.i = call fast double @_ZN3car7get_posEv(%class.car* nonnull %agg.tmp21)
  %mul4.i = fmul fast double %call3.i, 1.000000e+01
  %add.i = fadd fast double %mul4.i, %mul2.i
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %10)
  %call.i24 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo9_M_insertIdEERSoT_(%"class.std::basic_ostream"* nonnull @_ZSt4cout, double %add.i)
  %11 = bitcast %"class.std::basic_ostream"* %call.i24 to i8**
  %vtable.i26 = load i8*, i8** %11, align 8, !tbaa !8
  %vbase.offset.ptr.i27 = getelementptr i8, i8* %vtable.i26, i64 -24
  %12 = bitcast i8* %vbase.offset.ptr.i27 to i64*
  %vbase.offset.i28 = load i64, i64* %12, align 8
  %13 = bitcast %"class.std::basic_ostream"* %call.i24 to i8*
  %add.ptr.i29 = getelementptr inbounds i8, i8* %13, i64 %vbase.offset.i28
  %_M_ctype.i59 = getelementptr inbounds i8, i8* %add.ptr.i29, i64 240
  %14 = bitcast i8* %_M_ctype.i59 to %"class.std::ctype"**
  %15 = load %"class.std::ctype"*, %"class.std::ctype"** %14, align 8, !tbaa !10
  %tobool.i98 = icmp eq %"class.std::ctype"* %15, null
  br i1 %tobool.i98, label %if.then.i99, label %_ZSt13__check_facetISt5ctypeIcEERKT_PS3_.exit101

if.then.i99:                                      ; preds = %_ZNKSt5ctypeIcE5widenEc.exit
  call void @_ZSt16__throw_bad_castv() #11
  unreachable

_ZSt13__check_facetISt5ctypeIcEERKT_PS3_.exit101: ; preds = %_ZNKSt5ctypeIcE5widenEc.exit
  %_M_widen_ok.i61 = getelementptr inbounds %"class.std::ctype", %"class.std::ctype"* %15, i64 0, i32 8
  %16 = load i8, i8* %_M_widen_ok.i61, align 8, !tbaa !14
  %tobool.i62 = icmp eq i8 %16, 0
  br i1 %tobool.i62, label %if.end.i68, label %if.then.i64

if.then.i64:                                      ; preds = %_ZSt13__check_facetISt5ctypeIcEERKT_PS3_.exit101
  %arrayidx.i63 = getelementptr inbounds %"class.std::ctype", %"class.std::ctype"* %15, i64 0, i32 9, i64 10
  %17 = load i8, i8* %arrayidx.i63, align 1, !tbaa !16
  br label %_ZNKSt5ctypeIcE5widenEc.exit70

if.end.i68:                                       ; preds = %_ZSt13__check_facetISt5ctypeIcEERKT_PS3_.exit101
  call void @_ZNKSt5ctypeIcE13_M_widen_initEv(%"class.std::ctype"* nonnull %15)
  %18 = bitcast %"class.std::ctype"* %15 to i8 (%"class.std::ctype"*, i8)***
  %vtable.i65 = load i8 (%"class.std::ctype"*, i8)**, i8 (%"class.std::ctype"*, i8)*** %18, align 8, !tbaa !8
  %vfn.i66 = getelementptr inbounds i8 (%"class.std::ctype"*, i8)*, i8 (%"class.std::ctype"*, i8)** %vtable.i65, i64 6
  %19 = load i8 (%"class.std::ctype"*, i8)*, i8 (%"class.std::ctype"*, i8)** %vfn.i66, align 8
  %call.i67 = call signext i8 %19(%"class.std::ctype"* nonnull %15, i8 signext 10)
  br label %_ZNKSt5ctypeIcE5widenEc.exit70

_ZNKSt5ctypeIcE5widenEc.exit70:                   ; preds = %if.end.i68, %if.then.i64
  %retval.0.i69 = phi i8 [ %17, %if.then.i64 ], [ %call.i67, %if.end.i68 ]
  %call1.i31 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo3putEc(%"class.std::basic_ostream"* nonnull %call.i24, i8 signext %retval.0.i69)
  %call.i32 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo5flushEv(%"class.std::basic_ostream"* nonnull %call1.i31)
  %call1.i34 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* nonnull dereferenceable(272) @_ZSt4cout, i8* nonnull getelementptr inbounds ([15 x i8], [15 x i8]* @.str.2, i64 0, i64 0), i64 14)
  %vtable.i36 = load i8*, i8** bitcast (%"class.std::basic_ostream"* @_ZSt4cout to i8**), align 8, !tbaa !8
  %vbase.offset.ptr.i37 = getelementptr i8, i8* %vtable.i36, i64 -24
  %20 = bitcast i8* %vbase.offset.ptr.i37 to i64*
  %vbase.offset.i38 = load i64, i64* %20, align 8
  %add.ptr.i39 = getelementptr inbounds i8, i8* bitcast (%"class.std::basic_ostream"* @_ZSt4cout to i8*), i64 %vbase.offset.i38
  %_M_ctype.i71 = getelementptr inbounds i8, i8* %add.ptr.i39, i64 240
  %21 = bitcast i8* %_M_ctype.i71 to %"class.std::ctype"**
  %22 = load %"class.std::ctype"*, %"class.std::ctype"** %21, align 8, !tbaa !10
  %tobool.i102 = icmp eq %"class.std::ctype"* %22, null
  br i1 %tobool.i102, label %if.then.i103, label %_ZSt13__check_facetISt5ctypeIcEERKT_PS3_.exit105

if.then.i103:                                     ; preds = %_ZNKSt5ctypeIcE5widenEc.exit70
  call void @_ZSt16__throw_bad_castv() #11
  unreachable

_ZSt13__check_facetISt5ctypeIcEERKT_PS3_.exit105: ; preds = %_ZNKSt5ctypeIcE5widenEc.exit70
  %_M_widen_ok.i73 = getelementptr inbounds %"class.std::ctype", %"class.std::ctype"* %22, i64 0, i32 8
  %23 = load i8, i8* %_M_widen_ok.i73, align 8, !tbaa !14
  %tobool.i74 = icmp eq i8 %23, 0
  br i1 %tobool.i74, label %if.end.i80, label %if.then.i76

if.then.i76:                                      ; preds = %_ZSt13__check_facetISt5ctypeIcEERKT_PS3_.exit105
  %arrayidx.i75 = getelementptr inbounds %"class.std::ctype", %"class.std::ctype"* %22, i64 0, i32 9, i64 10
  %24 = load i8, i8* %arrayidx.i75, align 1, !tbaa !16
  br label %_ZNKSt5ctypeIcE5widenEc.exit82

if.end.i80:                                       ; preds = %_ZSt13__check_facetISt5ctypeIcEERKT_PS3_.exit105
  call void @_ZNKSt5ctypeIcE13_M_widen_initEv(%"class.std::ctype"* nonnull %22)
  %25 = bitcast %"class.std::ctype"* %22 to i8 (%"class.std::ctype"*, i8)***
  %vtable.i77 = load i8 (%"class.std::ctype"*, i8)**, i8 (%"class.std::ctype"*, i8)*** %25, align 8, !tbaa !8
  %vfn.i78 = getelementptr inbounds i8 (%"class.std::ctype"*, i8)*, i8 (%"class.std::ctype"*, i8)** %vtable.i77, i64 6
  %26 = load i8 (%"class.std::ctype"*, i8)*, i8 (%"class.std::ctype"*, i8)** %vfn.i78, align 8
  %call.i79 = call signext i8 %26(%"class.std::ctype"* nonnull %22, i8 signext 10)
  br label %_ZNKSt5ctypeIcE5widenEc.exit82

_ZNKSt5ctypeIcE5widenEc.exit82:                   ; preds = %if.end.i80, %if.then.i76
  %retval.0.i81 = phi i8 [ %24, %if.then.i76 ], [ %call.i79, %if.end.i80 ]
  %call1.i41 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo3putEc(%"class.std::basic_ostream"* nonnull @_ZSt4cout, i8 signext %retval.0.i81)
  %call.i42 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo5flushEv(%"class.std::basic_ostream"* nonnull %call1.i41)
  %call1.i44 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* nonnull dereferenceable(272) @_ZSt4cout, i8* nonnull getelementptr inbounds ([22 x i8], [22 x i8]* @.str.3.4, i64 0, i64 0), i64 21)
  %27 = load i32, i32* @enzyme_const, align 4, !tbaa !21
  %28 = load i32, i32* @enzyme_out, align 4, !tbaa !21
  %call.i46 = call fast double @_Z17__enzyme_autodiffIdJi3caridEET_PvDpT0_(i8* bitcast (double (%class.car*, double)* @_Z13car_erg_atpos3card to i8*), i32 %27, %class.car* nonnull byval(%class.car) align 8 %mycar, i32 %28, double 3.000000e+00) #10
  %call.i47 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo9_M_insertIdEERSoT_(%"class.std::basic_ostream"* nonnull @_ZSt4cout, double %call.i46)
  %29 = bitcast %"class.std::basic_ostream"* %call.i47 to i8**
  %vtable.i49 = load i8*, i8** %29, align 8, !tbaa !8
  %vbase.offset.ptr.i50 = getelementptr i8, i8* %vtable.i49, i64 -24
  %30 = bitcast i8* %vbase.offset.ptr.i50 to i64*
  %vbase.offset.i51 = load i64, i64* %30, align 8
  %31 = bitcast %"class.std::basic_ostream"* %call.i47 to i8*
  %add.ptr.i52 = getelementptr inbounds i8, i8* %31, i64 %vbase.offset.i51
  %_M_ctype.i83 = getelementptr inbounds i8, i8* %add.ptr.i52, i64 240
  %32 = bitcast i8* %_M_ctype.i83 to %"class.std::ctype"**
  %33 = load %"class.std::ctype"*, %"class.std::ctype"** %32, align 8, !tbaa !10
  %tobool.i106 = icmp eq %"class.std::ctype"* %33, null
  br i1 %tobool.i106, label %if.then.i107, label %_ZSt13__check_facetISt5ctypeIcEERKT_PS3_.exit109

if.then.i107:                                     ; preds = %_ZNKSt5ctypeIcE5widenEc.exit82
  call void @_ZSt16__throw_bad_castv() #11
  unreachable

_ZSt13__check_facetISt5ctypeIcEERKT_PS3_.exit109: ; preds = %_ZNKSt5ctypeIcE5widenEc.exit82
  %_M_widen_ok.i85 = getelementptr inbounds %"class.std::ctype", %"class.std::ctype"* %33, i64 0, i32 8
  %34 = load i8, i8* %_M_widen_ok.i85, align 8, !tbaa !14
  %tobool.i86 = icmp eq i8 %34, 0
  br i1 %tobool.i86, label %if.end.i92, label %if.then.i88

if.then.i88:                                      ; preds = %_ZSt13__check_facetISt5ctypeIcEERKT_PS3_.exit109
  %arrayidx.i87 = getelementptr inbounds %"class.std::ctype", %"class.std::ctype"* %33, i64 0, i32 9, i64 10
  %35 = load i8, i8* %arrayidx.i87, align 1, !tbaa !16
  br label %_ZNKSt5ctypeIcE5widenEc.exit94

if.end.i92:                                       ; preds = %_ZSt13__check_facetISt5ctypeIcEERKT_PS3_.exit109
  call void @_ZNKSt5ctypeIcE13_M_widen_initEv(%"class.std::ctype"* nonnull %33)
  %36 = bitcast %"class.std::ctype"* %33 to i8 (%"class.std::ctype"*, i8)***
  %vtable.i89 = load i8 (%"class.std::ctype"*, i8)**, i8 (%"class.std::ctype"*, i8)*** %36, align 8, !tbaa !8
  %vfn.i90 = getelementptr inbounds i8 (%"class.std::ctype"*, i8)*, i8 (%"class.std::ctype"*, i8)** %vtable.i89, i64 6
  %37 = load i8 (%"class.std::ctype"*, i8)*, i8 (%"class.std::ctype"*, i8)** %vfn.i90, align 8
  %call.i91 = call signext i8 %37(%"class.std::ctype"* nonnull %33, i8 signext 10)
  br label %_ZNKSt5ctypeIcE5widenEc.exit94

_ZNKSt5ctypeIcE5widenEc.exit94:                   ; preds = %if.end.i92, %if.then.i88
  %retval.0.i93 = phi i8 [ %35, %if.then.i88 ], [ %call.i91, %if.end.i92 ]
  %call1.i54 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo3putEc(%"class.std::basic_ostream"* nonnull %call.i47, i8 signext %retval.0.i93)
  %call.i55 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo5flushEv(%"class.std::basic_ostream"* nonnull %call1.i54)
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %0) #10
  ret i32 0
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #9

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg) #9

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #9

attributes #0 = { uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #2 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #3 = { nofree nounwind }
attributes #4 = { nofree norecurse nounwind uwtable writeonly "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #5 = { norecurse nounwind readonly uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #6 = { noreturn "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #7 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #8 = { norecurse uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #9 = { argmemonly nounwind willreturn }
attributes #10 = { nounwind }
attributes #11 = { noreturn }

!llvm.ident = !{!0, !0}
!llvm.module.flags = !{!1}

!0 = !{!"clang version 9.0.0 (https://github.com/llvm/llvm-project.git 0399d5a9682b3cef71c653373e38890c63c4c365)"}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{!3, !4, i64 0}
!3 = !{!"_ZTS6twonum", !4, i64 0, !4, i64 8}
!4 = !{!"double", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}
!7 = !{!3, !4, i64 8}
!8 = !{!9, !9, i64 0}
!9 = !{!"vtable pointer", !6, i64 0}
!10 = !{!11, !12, i64 240}
!11 = !{!"_ZTSSt9basic_iosIcSt11char_traitsIcEE", !12, i64 216, !5, i64 224, !13, i64 225, !12, i64 232, !12, i64 240, !12, i64 248, !12, i64 256}
!12 = !{!"any pointer", !5, i64 0}
!13 = !{!"bool", !5, i64 0}
!14 = !{!15, !5, i64 56}
!15 = !{!"_ZTSSt5ctypeIcE", !12, i64 16, !13, i64 24, !12, i64 32, !12, i64 40, !12, i64 48, !5, i64 56, !5, i64 57, !5, i64 313, !5, i64 569}
!16 = !{!5, !5, i64 0}
!17 = !{!18, !4, i64 0}
!18 = !{!"_ZTS3car", !4, i64 0, !4, i64 8, !4, i64 16}
!19 = !{!18, !4, i64 8}
!20 = !{!18, !4, i64 16}
!21 = !{!22, !22, i64 0}
!22 = !{!"int", !5, i64 0}

; CHECK: define internal { double } @diffe_Z13car_erg_atpos3card(%class.car* byval(%class.car) align 8 %car1, double %pos, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"car1'ipa" = alloca %class.car
; CHECK-NEXT:   %0 = bitcast %class.car* %"car1'ipa" to i8*
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull %0, i8 0, i64 24, i1 false)