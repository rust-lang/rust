; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

; ensure no cache is created

; ModuleID = 'qmu2.cpp'
source_filename = "qmu2.cpp"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%"class.std::ios_base::Init" = type { i8 }
%"class.std::mersenne_twister_engine" = type { [624 x i64], i64 }

$_ZSt18generate_canonicalIdLm53ESt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEET_RT1_ = comdat any

@_ZStL8__ioinit = internal global %"class.std::ios_base::Init" zeroinitializer, align 1
@__dso_handle = external hidden global i8
@enzyme_dup = dso_local local_unnamed_addr global i32 0, align 4
@enzyme_out = dso_local local_unnamed_addr global i32 0, align 4
@enzyme_const = dso_local local_unnamed_addr global i32 0, align 4
@.str.1 = private unnamed_addr constant [13 x i8] c"dx[%d] = %f\0A\00", align 1
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_qmu2.cpp, i8* null }]
@str = private unnamed_addr constant [16 x i8] c"before autodiff\00"

declare dso_local void @_ZNSt8ios_base4InitC1Ev(%"class.std::ios_base::Init"*) unnamed_addr #0

; Function Attrs: nounwind
declare dso_local void @_ZNSt8ios_base4InitD1Ev(%"class.std::ios_base::Init"*) unnamed_addr #1

; Function Attrs: nounwind
declare dso_local i32 @__cxa_atexit(void (i8*)*, i8*, i8*) local_unnamed_addr #2

; Function Attrs: uwtable
define dso_local double @_Z3fooPdi(double* noalias nocapture readonly %parts, i32 %n) #3 {
entry:
  %conv = sext i32 %n to i64
  %0 = tail call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %conv, i64 4)
  %1 = extractvalue { i64, i1 } %0, 1
  %2 = extractvalue { i64, i1 } %0, 0
  %3 = select i1 %1, i64 -1, i64 %2
  %call = tail call i8* @_Znam(i64 %3) #9
  %4 = bitcast i8* %call to i32*
  %cmp8135 = icmp sgt i32 %n, 0
  br i1 %cmp8135, label %for.body10.preheader, label %delete.notnull

for.body10.preheader:                             ; preds = %entry
  %wide.trip.count150 = zext i32 %n to i64
  %min.iters.check = icmp eq i32 %n, 1
  br i1 %min.iters.check, label %for.body10.preheader159, label %vector.scevcheck

vector.scevcheck:                                 ; preds = %for.body10.preheader
  %5 = add nsw i64 %wide.trip.count150, -1
  %6 = trunc i64 %5 to i32
  %mul = call { i32, i1 } @llvm.umul.with.overflow.i32(i32 %6, i32 3)
  %mul.overflow = extractvalue { i32, i1 } %mul, 1
  %7 = icmp ugt i64 %5, 4294967295
  %8 = or i1 %7, %mul.overflow
  br i1 %8, label %for.body10.preheader159, label %vector.ph

vector.ph:                                        ; preds = %vector.scevcheck
  %n.vec = and i64 %wide.trip.count150, 4294967294
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %9 = getelementptr inbounds i32, i32* %4, i64 %index
  %10 = mul i64 %index, 3
  %11 = and i64 %10, 4294967294
  %12 = getelementptr inbounds double, double* %parts, i64 %11
  %13 = bitcast double* %12 to <6 x double>*
  %wide.vec = load <6 x double>, <6 x double>* %13, align 8, !tbaa !2
  %strided.vec = shufflevector <6 x double> %wide.vec, <6 x double> undef, <2 x i32> <i32 0, i32 3>
  %strided.vec157 = shufflevector <6 x double> %wide.vec, <6 x double> undef, <2 x i32> <i32 1, i32 4>
  %strided.vec158 = shufflevector <6 x double> %wide.vec, <6 x double> undef, <2 x i32> <i32 2, i32 5>
  %14 = fdiv <2 x double> %strided.vec, <double 3.100000e+00, double 3.100000e+00>
  %15 = fptosi <2 x double> %14 to <2 x i32>
  %16 = fdiv <2 x double> %strided.vec157, <double 3.100000e+00, double 3.100000e+00>
  %17 = fmul <2 x double> %16, <double 4.000000e+00, double 4.000000e+00>
  %18 = fptosi <2 x double> %17 to <2 x i32>
  %19 = add nsw <2 x i32> %15, %18
  %20 = fdiv <2 x double> %strided.vec158, <double 3.100000e+00, double 3.100000e+00>
  %21 = fmul <2 x double> %20, <double 1.600000e+01, double 1.600000e+01>
  %22 = fptosi <2 x double> %21 to <2 x i32>
  %23 = add nsw <2 x i32> %19, %22
  %24 = bitcast i32* %9 to <2 x i32>*
  store <2 x i32> %23, <2 x i32>* %24, align 4, !tbaa !6
  %index.next = add i64 %index, 2
  %25 = icmp eq i64 %index.next, %n.vec
  br i1 %25, label %middle.block, label %vector.body, !llvm.loop !8

middle.block:                                     ; preds = %vector.body
  %cmp.n = icmp eq i64 %n.vec, %wide.trip.count150
  br i1 %cmp.n, label %for.cond46.preheader.lr.ph.preheader, label %for.body10.preheader159

for.body10.preheader159:                          ; preds = %middle.block, %vector.scevcheck, %for.body10.preheader
  %indvars.iv148.ph = phi i64 [ 0, %vector.scevcheck ], [ 0, %for.body10.preheader ], [ %n.vec, %middle.block ]
  br label %for.body10

for.body10:                                       ; preds = %for.body10.preheader159, %for.body10
  %indvars.iv148 = phi i64 [ %indvars.iv.next149, %for.body10 ], [ %indvars.iv148.ph, %for.body10.preheader159 ]
  %arrayidx12 = getelementptr inbounds i32, i32* %4, i64 %indvars.iv148
  %mul17 = mul i64 %indvars.iv148, 3
  %26 = and i64 %mul17, 4294967295
  %arrayidx19 = getelementptr inbounds double, double* %parts, i64 %26
  %27 = load double, double* %arrayidx19, align 8, !tbaa !2
  %div21 = fdiv double %27, 3.100000e+00
  %conv26 = fptosi double %div21 to i32
  %28 = add nuw nsw i64 %26, 1
  %arrayidx19.1 = getelementptr inbounds double, double* %parts, i64 %28
  %29 = load double, double* %arrayidx19.1, align 8, !tbaa !2
  %div21.1 = fdiv double %29, 3.100000e+00
  %mul25.1 = fmul double %div21.1, 4.000000e+00
  %conv26.1 = fptosi double %mul25.1 to i32
  %add29.1 = add nsw i32 %conv26, %conv26.1
  %30 = add nuw nsw i64 %26, 2
  %arrayidx19.2 = getelementptr inbounds double, double* %parts, i64 %30
  %31 = load double, double* %arrayidx19.2, align 8, !tbaa !2
  %div21.2 = fdiv double %31, 3.100000e+00
  %mul25.2 = fmul double %div21.2, 1.600000e+01
  %conv26.2 = fptosi double %mul25.2 to i32
  %add29.2 = add nsw i32 %add29.1, %conv26.2
  store i32 %add29.2, i32* %arrayidx12, align 4, !tbaa !6
  %indvars.iv.next149 = add nuw nsw i64 %indvars.iv148, 1
  %exitcond151 = icmp eq i64 %indvars.iv.next149, %wide.trip.count150
  br i1 %exitcond151, label %for.cond46.preheader.lr.ph.preheader, label %for.body10, !llvm.loop !10

for.cond46.preheader.lr.ph.preheader:             ; preds = %for.body10, %middle.block
  br label %for.cond46.preheader.lr.ph

for.cond46.preheader.lr.ph:                       ; preds = %for.cond46.preheader.lr.ph.preheader, %for.cond.cleanup43
  %indvars.iv140 = phi i64 [ %indvars.iv.next141, %for.cond.cleanup43 ], [ 0, %for.cond46.preheader.lr.ph.preheader ]
  %out.0132 = phi double [ %spec.select, %for.cond.cleanup43 ], [ 0.000000e+00, %for.cond46.preheader.lr.ph.preheader ]
  %mul50 = mul i64 %indvars.iv140, 3
  %arrayidx65 = getelementptr inbounds i32, i32* %4, i64 %indvars.iv140
  %32 = load i32, i32* %arrayidx65, align 4, !tbaa !6
  %33 = and i64 %mul50, 4294967295
  %arrayidx53 = getelementptr inbounds double, double* %parts, i64 %33
  %.pre = load double, double* %arrayidx53, align 8, !tbaa !2
  %34 = add nuw nsw i64 %33, 1
  %arrayidx53.1 = getelementptr inbounds double, double* %parts, i64 %34
  %35 = bitcast double* %arrayidx53.1 to <2 x double>*
  %36 = load <2 x double>, <2 x double>* %35, align 8, !tbaa !2
  br label %for.cond46.preheader

for.cond46.preheader:                             ; preds = %for.cond46.preheader, %for.cond46.preheader.lr.ph
  %indvars.iv = phi i64 [ 0, %for.cond46.preheader.lr.ph ], [ %indvars.iv.next, %for.cond46.preheader ]
  %out.1128 = phi double [ %out.0132, %for.cond46.preheader.lr.ph ], [ %spec.select, %for.cond46.preheader ]
  %mul54 = mul i64 %indvars.iv, 3
  %37 = and i64 %mul54, 4294967295
  %arrayidx57 = getelementptr inbounds double, double* %parts, i64 %37
  %38 = load double, double* %arrayidx57, align 8, !tbaa !2
  %sub58 = fsub double %.pre, %38
  %mul59 = fmul double %sub58, %sub58
  %add60 = fadd double %mul59, 0.000000e+00
  %39 = add nuw nsw i64 %37, 1
  %arrayidx57.1 = getelementptr inbounds double, double* %parts, i64 %39
  %40 = bitcast double* %arrayidx57.1 to <2 x double>*
  %41 = load <2 x double>, <2 x double>* %40, align 8, !tbaa !2
  %42 = fsub <2 x double> %36, %41
  %43 = fmul <2 x double> %42, %42
  %44 = extractelement <2 x double> %43, i32 0
  %add60.1 = fadd double %add60, %44
  %45 = extractelement <2 x double> %43, i32 1
  %add60.2 = fadd double %add60.1, %45
  %arrayidx67 = getelementptr inbounds i32, i32* %4, i64 %indvars.iv
  %46 = load i32, i32* %arrayidx67, align 4, !tbaa !6
  %cmp68 = icmp eq i32 %32, %46
  %add69 = fadd double %out.1128, %add60.2
  %spec.select = select i1 %cmp68, double %add69, double %out.1128
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count150
  br i1 %exitcond, label %for.cond.cleanup43, label %for.cond46.preheader

for.cond.cleanup43:                               ; preds = %for.cond46.preheader
  %indvars.iv.next141 = add nuw nsw i64 %indvars.iv140, 1
  %exitcond143 = icmp eq i64 %indvars.iv.next141, %wide.trip.count150
  br i1 %exitcond143, label %delete.notnull, label %for.cond46.preheader.lr.ph

delete.notnull:                                   ; preds = %for.cond.cleanup43, %entry
  %out.0.lcssa = phi double [ 0.000000e+00, %entry ], [ %spec.select, %for.cond.cleanup43 ]
  tail call void @_ZdaPv(i8* nonnull %call) #10
  ret double %out.0.lcssa
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #4

; Function Attrs: nounwind readnone speculatable
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64) #5

; Function Attrs: nobuiltin
declare dso_local noalias nonnull i8* @_Znam(i64) local_unnamed_addr #6

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #4

; Function Attrs: nobuiltin nounwind
declare dso_local void @_ZdaPv(i8*) local_unnamed_addr #7

; Function Attrs: norecurse uwtable
define dso_local i32 @main() local_unnamed_addr #8 {
entry:
  %e2 = alloca %"class.std::mersenne_twister_engine", align 8
  tail call void @srand(i32 42) #2
  %0 = bitcast %"class.std::mersenne_twister_engine"* %e2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 5000, i8* nonnull %0) #2
  %arrayidx.i.i = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %e2, i64 0, i32 0, i64 0
  store i64 42, i64* %arrayidx.i.i, align 8, !tbaa !11
  br label %for.body.i.i

for.body.i.i:                                     ; preds = %for.body.i.i.1, %entry
  %1 = phi i64 [ 42, %entry ], [ %rem.i.i18.i.i.1, %for.body.i.i.1 ]
  %__i.021.i.i = phi i64 [ 1, %entry ], [ %inc.i.i.1, %for.body.i.i.1 ]
  %shr.i.i = lshr i64 %1, 30
  %xor.i.i = xor i64 %shr.i.i, %1
  %mul.i.i = mul nuw nsw i64 %xor.i.i, 1812433253
  %rem.i.i19.lhs.trunc.i.i = trunc i64 %__i.021.i.i to i16
  %rem.i.i1920.i.i = urem i16 %rem.i.i19.lhs.trunc.i.i, 624
  %rem.i.i19.zext.i.i = zext i16 %rem.i.i1920.i.i to i64
  %add.i.i = add nuw i64 %mul.i.i, %rem.i.i19.zext.i.i
  %rem.i.i18.i.i = and i64 %add.i.i, 4294967295
  %arrayidx7.i.i = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %e2, i64 0, i32 0, i64 %__i.021.i.i
  store i64 %rem.i.i18.i.i, i64* %arrayidx7.i.i, align 8, !tbaa !11
  %inc.i.i = add nuw nsw i64 %__i.021.i.i, 1
  %exitcond.i.i = icmp eq i64 %inc.i.i, 624
  br i1 %exitcond.i.i, label %_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEC2Em.exit, label %for.body.i.i.1

_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEC2Em.exit: ; preds = %for.body.i.i
  %_M_p.i.i = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %e2, i64 0, i32 1
  store i64 624, i64* %_M_p.i.i, align 8, !tbaa !13
  %call = tail call i8* @_Znam(i64 2400000) #9
  %2 = bitcast i8* %call to double*
  %call3 = tail call i8* @_Znam(i64 2400000) #9
  %3 = bitcast i8* %call3 to double*
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  %puts = call i32 @puts(i8* getelementptr inbounds ([16 x i8], [16 x i8]* @str, i64 0, i64 0))
  call void @diffe_Z3fooPdi(double* %2, double* %3, i32 100000, double 1.000000e+00)
  br label %for.body16

for.body:                                         ; preds = %for.body, %_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEC2Em.exit
  %indvars.iv51 = phi i64 [ 0, %_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEC2Em.exit ], [ %indvars.iv.next52, %for.body ]
  %call.i.i.i = call double @_ZSt18generate_canonicalIdLm53ESt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEET_RT1_(%"class.std::mersenne_twister_engine"* nonnull dereferenceable(5000) %e2)
  %mul.i.i45 = fmul double %call.i.i.i, 1.000000e+01
  %add.i.i46 = fadd double %mul.i.i45, 0.000000e+00
  %arrayidx = getelementptr inbounds double, double* %2, i64 %indvars.iv51
  store double %add.i.i46, double* %arrayidx, align 8, !tbaa !2
  %arrayidx7 = getelementptr inbounds double, double* %3, i64 %indvars.iv51
  store double 0.000000e+00, double* %arrayidx7, align 8, !tbaa !2
  %indvars.iv.next52 = add nuw nsw i64 %indvars.iv51, 1
  %exitcond53 = icmp eq i64 %indvars.iv.next52, 300000
  br i1 %exitcond53, label %for.cond.cleanup, label %for.body

for.cond.cleanup15:                               ; preds = %for.body16
  call void @llvm.lifetime.end.p0i8(i64 5000, i8* nonnull %0) #2
  ret i32 0

for.body16:                                       ; preds = %for.body16, %for.cond.cleanup
  %indvars.iv = phi i64 [ 0, %for.cond.cleanup ], [ %indvars.iv.next, %for.body16 ]
  %arrayidx18 = getelementptr inbounds double, double* %3, i64 %indvars.iv
  %4 = load double, double* %arrayidx18, align 8, !tbaa !2
  %5 = trunc i64 %indvars.iv to i32
  %call19 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.1, i64 0, i64 0), i32 %5, double %4)
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 100
  br i1 %exitcond, label %for.cond.cleanup15, label %for.body16

for.body.i.i.1:                                   ; preds = %for.body.i.i
  %shr.i.i.1 = lshr i64 %rem.i.i18.i.i, 30
  %xor.i.i.1 = xor i64 %shr.i.i.1, %rem.i.i18.i.i
  %mul.i.i.1 = mul nuw nsw i64 %xor.i.i.1, 1812433253
  %rem.i.i19.lhs.trunc.i.i.1 = trunc i64 %inc.i.i to i16
  %rem.i.i1920.i.i.1 = urem i16 %rem.i.i19.lhs.trunc.i.i.1, 624
  %rem.i.i19.zext.i.i.1 = zext i16 %rem.i.i1920.i.i.1 to i64
  %add.i.i.1 = add nuw i64 %mul.i.i.1, %rem.i.i19.zext.i.i.1
  %rem.i.i18.i.i.1 = and i64 %add.i.i.1, 4294967295
  %arrayidx7.i.i.1 = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %e2, i64 0, i32 0, i64 %inc.i.i
  store i64 %rem.i.i18.i.i.1, i64* %arrayidx7.i.i.1, align 8, !tbaa !11
  %inc.i.i.1 = add nuw nsw i64 %__i.021.i.i, 2
  br label %for.body.i.i
}

; Function Attrs: nounwind
declare dso_local void @srand(i32) local_unnamed_addr #1

; Function Attrs: nounwind
declare dso_local i32 @printf(i8* nocapture readonly, ...) local_unnamed_addr #1

; Function Attrs: uwtable
define linkonce_odr dso_local double @_ZSt18generate_canonicalIdLm53ESt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEET_RT1_(%"class.std::mersenne_twister_engine"* dereferenceable(5000) %__urng) local_unnamed_addr #3 comdat {
entry:
  %call.i55 = tail call x86_fp80 @logl(x86_fp80 0xK401F8000000000000000) #2
  %call.i = tail call x86_fp80 @logl(x86_fp80 0xK40008000000000000000) #2
  %div = fdiv x86_fp80 %call.i55, %call.i
  %conv7 = fptoui x86_fp80 %div to i64
  %sub11 = add i64 %conv7, 52
  %div12 = udiv i64 %sub11, %conv7
  %cmp.i53 = icmp ugt i64 %div12, 1
  %spec.select = select i1 %cmp.i53, i64 %div12, i64 1
  %cmp63 = icmp eq i64 %spec.select, 0
  br i1 %cmp63, label %for.cond.cleanup, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %entry
  %_M_p.i = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %__urng, i64 0, i32 1
  %arrayidx.phi.trans.insert.i = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %__urng, i64 0, i32 0, i64 0
  %arrayidx19.phi.trans.insert.i = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %__urng, i64 0, i32 0, i64 227
  %arrayidx42.i = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %__urng, i64 0, i32 0, i64 623
  %arrayidx49.i = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %__urng, i64 0, i32 0, i64 396
  %.pre = load i64, i64* %_M_p.i, align 8, !tbaa !13
  br label %for.body

for.cond.cleanup:                                 ; preds = %_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv.exit, %entry
  %__sum.0.lcssa = phi double [ 0.000000e+00, %entry ], [ %add18, %_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv.exit ]
  %__tmp.0.lcssa = phi double [ 1.000000e+00, %entry ], [ %conv21, %_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv.exit ]
  %div22 = fdiv double %__sum.0.lcssa, %__tmp.0.lcssa
  %cmp23 = fcmp ult double %div22, 1.000000e+00
  br i1 %cmp23, label %if.end, label %if.then, !prof !15

for.body:                                         ; preds = %for.body.lr.ph, %_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv.exit
  %0 = phi i64 [ %.pre, %for.body.lr.ph ], [ %inc.i, %_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv.exit ]
  %__k.066 = phi i64 [ %spec.select, %for.body.lr.ph ], [ %dec, %_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv.exit ]
  %__tmp.065 = phi double [ 1.000000e+00, %for.body.lr.ph ], [ %conv21, %_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv.exit ]
  %__sum.064 = phi double [ 0.000000e+00, %for.body.lr.ph ], [ %add18, %_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv.exit ]
  %cmp.i45 = icmp ugt i64 %0, 623
  br i1 %cmp.i45, label %if.then.i, label %for.body._ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv.exit_crit_edge

for.body._ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv.exit_crit_edge: ; preds = %for.body
  %.pre68 = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %__urng, i64 0, i32 0, i64 %0
  br label %_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv.exit

if.then.i:                                        ; preds = %for.body
  %.pre.i46 = load i64, i64* %arrayidx.phi.trans.insert.i, align 8, !tbaa !11
  br label %for.body.i

for.body.i:                                       ; preds = %for.body.i, %if.then.i
  %1 = phi i64 [ %.pre.i46, %if.then.i ], [ %2, %for.body.i ]
  %__k.079.i = phi i64 [ 0, %if.then.i ], [ %add.i, %for.body.i ]
  %arrayidx.i47 = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %__urng, i64 0, i32 0, i64 %__k.079.i
  %and.i48 = and i64 %1, -2147483648
  %add.i = add nuw nsw i64 %__k.079.i, 1
  %arrayidx3.i = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %__urng, i64 0, i32 0, i64 %add.i
  %2 = load i64, i64* %arrayidx3.i, align 8, !tbaa !11
  %and4.i = and i64 %2, 2147483646
  %or.i = or i64 %and4.i, %and.i48
  %add6.i = add nuw nsw i64 %__k.079.i, 397
  %arrayidx7.i = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %__urng, i64 0, i32 0, i64 %add6.i
  %3 = load i64, i64* %arrayidx7.i, align 8, !tbaa !11
  %shr.i49 = lshr exact i64 %or.i, 1
  %xor.i50 = xor i64 %shr.i49, %3
  %and8.i = and i64 %2, 1
  %tobool.i = icmp eq i64 %and8.i, 0
  %cond.i = select i1 %tobool.i, i64 0, i64 2567483615
  %xor9.i51 = xor i64 %xor.i50, %cond.i
  store i64 %xor9.i51, i64* %arrayidx.i47, align 8, !tbaa !11
  %exitcond80.i = icmp eq i64 %add.i, 227
  br i1 %exitcond80.i, label %for.body16.preheader.i, label %for.body.i

for.body16.preheader.i:                           ; preds = %for.body.i
  %.pre81.i = load i64, i64* %arrayidx19.phi.trans.insert.i, align 8, !tbaa !11
  br label %for.body16.i

for.body16.i:                                     ; preds = %for.body16.i, %for.body16.preheader.i
  %4 = phi i64 [ %5, %for.body16.i ], [ %.pre81.i, %for.body16.preheader.i ]
  %__k12.078.i = phi i64 [ %add22.i, %for.body16.i ], [ 227, %for.body16.preheader.i ]
  %arrayidx19.i = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %__urng, i64 0, i32 0, i64 %__k12.078.i
  %and20.i = and i64 %4, -2147483648
  %add22.i = add nuw nsw i64 %__k12.078.i, 1
  %arrayidx23.i = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %__urng, i64 0, i32 0, i64 %add22.i
  %5 = load i64, i64* %arrayidx23.i, align 8, !tbaa !11
  %and24.i = and i64 %5, 2147483646
  %or25.i = or i64 %and24.i, %and20.i
  %add27.i = add nsw i64 %__k12.078.i, -227
  %arrayidx28.i = getelementptr inbounds %"class.std::mersenne_twister_engine", %"class.std::mersenne_twister_engine"* %__urng, i64 0, i32 0, i64 %add27.i
  %6 = load i64, i64* %arrayidx28.i, align 8, !tbaa !11
  %shr29.i = lshr exact i64 %or25.i, 1
  %xor30.i = xor i64 %shr29.i, %6
  %and31.i = and i64 %5, 1
  %tobool32.i = icmp eq i64 %and31.i, 0
  %cond33.i = select i1 %tobool32.i, i64 0, i64 2567483615
  %xor34.i = xor i64 %xor30.i, %cond33.i
  store i64 %xor34.i, i64* %arrayidx19.i, align 8, !tbaa !11
  %exitcond.i = icmp eq i64 %add22.i, 623
  br i1 %exitcond.i, label %_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EE11_M_gen_randEv.exit, label %for.body16.i

_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EE11_M_gen_randEv.exit: ; preds = %for.body16.i
  %7 = load i64, i64* %arrayidx42.i, align 8, !tbaa !11
  %and43.i = and i64 %7, -2147483648
  %8 = load i64, i64* %arrayidx.phi.trans.insert.i, align 8, !tbaa !11
  %and46.i = and i64 %8, 2147483646
  %or47.i = or i64 %and46.i, %and43.i
  %9 = load i64, i64* %arrayidx49.i, align 8, !tbaa !11
  %shr50.i = lshr exact i64 %or47.i, 1
  %xor51.i = xor i64 %shr50.i, %9
  %and52.i = and i64 %8, 1
  %tobool53.i = icmp eq i64 %and52.i, 0
  %cond54.i = select i1 %tobool53.i, i64 0, i64 2567483615
  %xor55.i = xor i64 %xor51.i, %cond54.i
  store i64 %xor55.i, i64* %arrayidx42.i, align 8, !tbaa !11
  store i64 0, i64* %_M_p.i, align 8, !tbaa !13
  br label %_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv.exit

_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv.exit: ; preds = %for.body._ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv.exit_crit_edge, %_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EE11_M_gen_randEv.exit
  %arrayidx.i.pre-phi = phi i64* [ %.pre68, %for.body._ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv.exit_crit_edge ], [ %arrayidx.phi.trans.insert.i, %_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EE11_M_gen_randEv.exit ]
  %10 = phi i64 [ %0, %for.body._ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv.exit_crit_edge ], [ 0, %_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EE11_M_gen_randEv.exit ]
  %inc.i = add i64 %10, 1
  store i64 %inc.i, i64* %_M_p.i, align 8, !tbaa !13
  %11 = load i64, i64* %arrayidx.i.pre-phi, align 8, !tbaa !11
  %shr.i = lshr i64 %11, 11
  %and.i = and i64 %shr.i, 4294967295
  %xor.i = xor i64 %and.i, %11
  %shl.i = shl i64 %xor.i, 7
  %and3.i = and i64 %shl.i, 2636928640
  %xor4.i = xor i64 %and3.i, %xor.i
  %shl5.i = shl i64 %xor4.i, 15
  %and6.i = and i64 %shl5.i, 4022730752
  %xor7.i = xor i64 %and6.i, %xor4.i
  %shr8.i = lshr i64 %xor7.i, 18
  %xor9.i = xor i64 %shr8.i, %xor7.i
  %conv17 = uitofp i64 %xor9.i to double
  %mul = fmul double %__tmp.065, %conv17
  %add18 = fadd double %__sum.064, %mul
  %conv19 = fpext double %__tmp.065 to x86_fp80
  %mul20 = fmul x86_fp80 %conv19, 0xK401F8000000000000000
  %conv21 = fptrunc x86_fp80 %mul20 to double
  %dec = add i64 %__k.066, -1
  %cmp = icmp eq i64 %dec, 0
  br i1 %cmp, label %for.cond.cleanup, label %for.body

if.then:                                          ; preds = %for.cond.cleanup
  %call25 = tail call double @nextafter(double 1.000000e+00, double 0.000000e+00) #2
  br label %if.end

if.end:                                           ; preds = %for.cond.cleanup, %if.then
  %__ret.0 = phi double [ %call25, %if.then ], [ %div22, %for.cond.cleanup ]
  ret double %__ret.0
}

; Function Attrs: nounwind
declare dso_local double @nextafter(double, double) local_unnamed_addr #1

; Function Attrs: nounwind
declare dso_local x86_fp80 @logl(x86_fp80) local_unnamed_addr #1

; Function Attrs: uwtable
define internal void @_GLOBAL__sub_I_qmu2.cpp() #3 section ".text.startup" {
entry:
  tail call void @_ZNSt8ios_base4InitC1Ev(%"class.std::ios_base::Init"* nonnull @_ZStL8__ioinit)
  %0 = tail call i32 @__cxa_atexit(void (i8*)* bitcast (void (%"class.std::ios_base::Init"*)* @_ZNSt8ios_base4InitD1Ev to void (i8*)*), i8* getelementptr inbounds (%"class.std::ios_base::Init", %"class.std::ios_base::Init"* @_ZStL8__ioinit, i64 0, i32 0), i8* nonnull @__dso_handle) #2
  ret void
}

; Function Attrs: nounwind
declare i32 @puts(i8* nocapture readonly) local_unnamed_addr #2

; Function Attrs: uwtable
define dso_local double @preprocess__Z3fooPdi(double* noalias nocapture readonly %parts, i32 %n) #3 {
entry:
  %conv = sext i32 %n to i64
  %0 = tail call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %conv, i64 4)
  %1 = extractvalue { i64, i1 } %0, 1
  %2 = extractvalue { i64, i1 } %0, 0
  %3 = select i1 %1, i64 -1, i64 %2
  %call = tail call i8* @_Znam(i64 %3) #9
  %4 = bitcast i8* %call to i32*
  %cmp8135 = icmp sgt i32 %n, 0
  br i1 %cmp8135, label %for.body10.preheader, label %delete.notnull

for.body10.preheader:                             ; preds = %entry
  %wide.trip.count150 = zext i32 %n to i64
  %min.iters.check = icmp eq i32 %n, 1
  br i1 %min.iters.check, label %for.body10.preheader5, label %vector.scevcheck

vector.scevcheck:                                 ; preds = %for.body10.preheader
  %5 = add nsw i64 %wide.trip.count150, -1
  %6 = trunc i64 %5 to i32
  %mul = call { i32, i1 } @llvm.umul.with.overflow.i32(i32 %6, i32 3)
  %mul.overflow = extractvalue { i32, i1 } %mul, 1
  %7 = icmp ugt i64 %5, 4294967295
  %8 = or i1 %7, %mul.overflow
  br i1 %8, label %for.body10.preheader5, label %vector.ph

vector.ph:                                        ; preds = %vector.scevcheck
  %n.vec = and i64 %wide.trip.count150, 4294967294
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %9 = getelementptr inbounds i32, i32* %4, i64 %index
  %10 = mul i64 %index, 3
  %11 = and i64 %10, 4294967294
  %12 = getelementptr inbounds double, double* %parts, i64 %11
  %13 = bitcast double* %12 to <6 x double>*
  %wide.vec = load <6 x double>, <6 x double>* %13, align 8, !tbaa !2
  %strided.vec = shufflevector <6 x double> %wide.vec, <6 x double> undef, <2 x i32> <i32 0, i32 3>
  %strided.vec3 = shufflevector <6 x double> %wide.vec, <6 x double> undef, <2 x i32> <i32 1, i32 4>
  %strided.vec4 = shufflevector <6 x double> %wide.vec, <6 x double> undef, <2 x i32> <i32 2, i32 5>
  %14 = fdiv <2 x double> %strided.vec, <double 3.100000e+00, double 3.100000e+00>
  %15 = fptosi <2 x double> %14 to <2 x i32>
  %16 = fdiv <2 x double> %strided.vec3, <double 3.100000e+00, double 3.100000e+00>
  %17 = fmul <2 x double> %16, <double 4.000000e+00, double 4.000000e+00>
  %18 = fptosi <2 x double> %17 to <2 x i32>
  %19 = add nsw <2 x i32> %15, %18
  %20 = fdiv <2 x double> %strided.vec4, <double 3.100000e+00, double 3.100000e+00>
  %21 = fmul <2 x double> %20, <double 1.600000e+01, double 1.600000e+01>
  %22 = fptosi <2 x double> %21 to <2 x i32>
  %23 = add nsw <2 x i32> %19, %22
  %24 = bitcast i32* %9 to <2 x i32>*
  store <2 x i32> %23, <2 x i32>* %24, align 4, !tbaa !6
  %index.next = add i64 %index, 2
  %25 = icmp eq i64 %index.next, %n.vec
  br i1 %25, label %middle.block, label %vector.body, !llvm.loop !16

middle.block:                                     ; preds = %vector.body
  %cmp.n = icmp eq i64 %n.vec, %wide.trip.count150
  br i1 %cmp.n, label %for.cond46.preheader.lr.ph.preheader, label %for.body10.preheader5

for.body10.preheader5:                            ; preds = %middle.block, %vector.scevcheck, %for.body10.preheader
  %tiv1.ph = phi i64 [ 0, %vector.scevcheck ], [ 0, %for.body10.preheader ], [ %n.vec, %middle.block ]
  br label %for.body10

for.body10:                                       ; preds = %for.body10.preheader5, %for.body10
  %tiv1 = phi i64 [ %tiv.next2, %for.body10 ], [ %tiv1.ph, %for.body10.preheader5 ]
  %tiv.next2 = add nuw nsw i64 %tiv1, 1
  %arrayidx12 = getelementptr inbounds i32, i32* %4, i64 %tiv1
  %mul17 = mul i64 %tiv1, 3
  %26 = and i64 %mul17, 4294967295
  %arrayidx19 = getelementptr inbounds double, double* %parts, i64 %26
  %27 = load double, double* %arrayidx19, align 8, !tbaa !2
  %div21 = fdiv double %27, 3.100000e+00
  %conv26 = fptosi double %div21 to i32
  %28 = add nuw nsw i64 %26, 1
  %arrayidx19.1 = getelementptr inbounds double, double* %parts, i64 %28
  %29 = load double, double* %arrayidx19.1, align 8, !tbaa !2
  %div21.1 = fdiv double %29, 3.100000e+00
  %mul25.1 = fmul double %div21.1, 4.000000e+00
  %conv26.1 = fptosi double %mul25.1 to i32
  %add29.1 = add nsw i32 %conv26, %conv26.1
  %30 = add nuw nsw i64 %26, 2
  %arrayidx19.2 = getelementptr inbounds double, double* %parts, i64 %30
  %31 = load double, double* %arrayidx19.2, align 8, !tbaa !2
  %div21.2 = fdiv double %31, 3.100000e+00
  %mul25.2 = fmul double %div21.2, 1.600000e+01
  %conv26.2 = fptosi double %mul25.2 to i32
  %add29.2 = add nsw i32 %add29.1, %conv26.2
  store i32 %add29.2, i32* %arrayidx12, align 4, !tbaa !6
  %exitcond151 = icmp eq i64 %tiv.next2, %wide.trip.count150
  br i1 %exitcond151, label %for.cond46.preheader.lr.ph.preheader, label %for.body10, !llvm.loop !17

for.cond46.preheader.lr.ph.preheader:             ; preds = %for.body10, %middle.block
  br label %for.cond46.preheader.lr.ph

for.cond46.preheader.lr.ph:                       ; preds = %for.cond46.preheader.lr.ph.preheader, %for.cond.cleanup43
  %tiv = phi i64 [ %tiv.next, %for.cond.cleanup43 ], [ 0, %for.cond46.preheader.lr.ph.preheader ]
  %out.0132 = phi double [ %spec.select, %for.cond.cleanup43 ], [ 0.000000e+00, %for.cond46.preheader.lr.ph.preheader ]
  %tiv.next = add nuw nsw i64 %tiv, 1
  %mul50 = mul i64 %tiv, 3
  %arrayidx65 = getelementptr inbounds i32, i32* %4, i64 %tiv
  %32 = load i32, i32* %arrayidx65, align 4, !tbaa !6
  %33 = and i64 %mul50, 4294967295
  %arrayidx53 = getelementptr inbounds double, double* %parts, i64 %33
  %.pre = load double, double* %arrayidx53, align 8, !tbaa !2
  %34 = add nuw nsw i64 %33, 1
  %arrayidx53.1 = getelementptr inbounds double, double* %parts, i64 %34
  %35 = bitcast double* %arrayidx53.1 to <2 x double>*
  %36 = load <2 x double>, <2 x double>* %35, align 8, !tbaa !2
  br label %for.cond46.preheader

for.cond46.preheader:                             ; preds = %for.cond46.preheader, %for.cond46.preheader.lr.ph
  %indvars.iv = phi i64 [ 0, %for.cond46.preheader.lr.ph ], [ %indvars.iv.next, %for.cond46.preheader ]
  %out.1128 = phi double [ %out.0132, %for.cond46.preheader.lr.ph ], [ %spec.select, %for.cond46.preheader ]
  %mul54 = mul i64 %indvars.iv, 3
  %37 = and i64 %mul54, 4294967295
  %arrayidx57 = getelementptr inbounds double, double* %parts, i64 %37
  %38 = load double, double* %arrayidx57, align 8, !tbaa !2
  %sub58 = fsub double %.pre, %38
  %mul59 = fmul double %sub58, %sub58
  %add60 = fadd double %mul59, 0.000000e+00
  %39 = add nuw nsw i64 %37, 1
  %arrayidx57.1 = getelementptr inbounds double, double* %parts, i64 %39
  %40 = bitcast double* %arrayidx57.1 to <2 x double>*
  %41 = load <2 x double>, <2 x double>* %40, align 8, !tbaa !2
  %42 = fsub <2 x double> %36, %41
  %43 = fmul <2 x double> %42, %42
  %44 = extractelement <2 x double> %43, i32 0
  %add60.1 = fadd double %add60, %44
  %45 = extractelement <2 x double> %43, i32 1
  %add60.2 = fadd double %add60.1, %45
  %arrayidx67 = getelementptr inbounds i32, i32* %4, i64 %indvars.iv
  %46 = load i32, i32* %arrayidx67, align 4, !tbaa !6
  %cmp68 = icmp eq i32 %32, %46
  %add69 = fadd double %out.1128, %add60.2
  %spec.select = select i1 %cmp68, double %add69, double %out.1128
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count150
  br i1 %exitcond, label %for.cond.cleanup43, label %for.cond46.preheader

for.cond.cleanup43:                               ; preds = %for.cond46.preheader
  %exitcond143 = icmp eq i64 %tiv.next, %wide.trip.count150
  br i1 %exitcond143, label %delete.notnull, label %for.cond46.preheader.lr.ph

delete.notnull:                                   ; preds = %for.cond.cleanup43, %entry
  %out.0.lcssa = phi double [ 0.000000e+00, %entry ], [ %spec.select, %for.cond.cleanup43 ]
  ret double %out.0.lcssa
}

; Function Attrs: uwtable
define internal void @diffe_Z3fooPdi(double* noalias nocapture readonly %parts, double* nocapture %"parts'", i32 %n, double %differeturn) #3 {
entry:
  %conv = sext i32 %n to i64
  %0 = tail call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %conv, i64 4)
  %1 = extractvalue { i64, i1 } %0, 1
  %2 = extractvalue { i64, i1 } %0, 0
  %3 = select i1 %1, i64 -1, i64 %2
  %call = tail call i8* @_Znam(i64 %3) #9
  %4 = bitcast i8* %call to i32*
  %cmp8135 = icmp sgt i32 %n, 0
  br i1 %cmp8135, label %for.body10.preheader, label %invertentry

for.body10.preheader:                             ; preds = %entry
  %wide.trip.count150 = zext i32 %n to i64
  %min.iters.check = icmp eq i32 %n, 1
  br i1 %min.iters.check, label %for.body10.preheader37, label %vector.scevcheck

vector.scevcheck:                                 ; preds = %for.body10.preheader
  %5 = add nsw i64 %wide.trip.count150, -1
  %6 = trunc i64 %5 to i32
  %mul = call { i32, i1 } @llvm.umul.with.overflow.i32(i32 %6, i32 3)
  %mul.overflow = extractvalue { i32, i1 } %mul, 1
  %7 = icmp ugt i64 %5, 4294967295
  %8 = or i1 %7, %mul.overflow
  br i1 %8, label %for.body10.preheader37, label %vector.ph

vector.ph:                                        ; preds = %vector.scevcheck
  %n.vec = and i64 %wide.trip.count150, 4294967294
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %9 = getelementptr inbounds i32, i32* %4, i64 %index
  %10 = mul i64 %index, 3
  %11 = and i64 %10, 4294967294
  %12 = getelementptr inbounds double, double* %parts, i64 %11
  %13 = bitcast double* %12 to <6 x double>*
  %wide.vec = load <6 x double>, <6 x double>* %13, align 8, !tbaa !2
  %strided.vec = shufflevector <6 x double> %wide.vec, <6 x double> undef, <2 x i32> <i32 0, i32 3>
  %strided.vec35 = shufflevector <6 x double> %wide.vec, <6 x double> undef, <2 x i32> <i32 1, i32 4>
  %strided.vec36 = shufflevector <6 x double> %wide.vec, <6 x double> undef, <2 x i32> <i32 2, i32 5>
  %14 = fdiv <2 x double> %strided.vec, <double 3.100000e+00, double 3.100000e+00>
  %15 = fptosi <2 x double> %14 to <2 x i32>
  %16 = fdiv <2 x double> %strided.vec35, <double 3.100000e+00, double 3.100000e+00>
  %17 = fmul <2 x double> %16, <double 4.000000e+00, double 4.000000e+00>
  %18 = fptosi <2 x double> %17 to <2 x i32>
  %19 = add nsw <2 x i32> %15, %18
  %20 = fdiv <2 x double> %strided.vec36, <double 3.100000e+00, double 3.100000e+00>
  %21 = fmul <2 x double> %20, <double 1.600000e+01, double 1.600000e+01>
  %22 = fptosi <2 x double> %21 to <2 x i32>
  %23 = add nsw <2 x i32> %19, %22
  %24 = bitcast i32* %9 to <2 x i32>*
  store <2 x i32> %23, <2 x i32>* %24, align 4, !tbaa !6
  %index.next = add i64 %index, 2
  %25 = icmp eq i64 %index.next, %n.vec
  br i1 %25, label %middle.block, label %vector.body, !llvm.loop !18

middle.block:                                     ; preds = %vector.body
  %cmp.n = icmp eq i64 %n.vec, %wide.trip.count150
  br i1 %cmp.n, label %delete.notnull, label %for.body10.preheader37

for.body10.preheader37:                           ; preds = %middle.block, %vector.scevcheck, %for.body10.preheader
  %iv.ph = phi i64 [ 0, %vector.scevcheck ], [ 0, %for.body10.preheader ], [ %n.vec, %middle.block ]
  br label %for.body10

for.body10:                                       ; preds = %for.body10.preheader37, %for.body10
  %iv = phi i64 [ %iv.next, %for.body10 ], [ %iv.ph, %for.body10.preheader37 ]
  %iv.next = add nuw nsw i64 %iv, 1
  %arrayidx12 = getelementptr inbounds i32, i32* %4, i64 %iv
  %mul17 = mul i64 %iv, 3
  %26 = and i64 %mul17, 4294967295
  %arrayidx19 = getelementptr inbounds double, double* %parts, i64 %26
  %27 = load double, double* %arrayidx19, align 8, !tbaa !2
  %div21 = fdiv double %27, 3.100000e+00
  %conv26 = fptosi double %div21 to i32
  %28 = add nuw nsw i64 %26, 1
  %arrayidx19.1 = getelementptr inbounds double, double* %parts, i64 %28
  %29 = load double, double* %arrayidx19.1, align 8, !tbaa !2
  %div21.1 = fdiv double %29, 3.100000e+00
  %mul25.1 = fmul double %div21.1, 4.000000e+00
  %conv26.1 = fptosi double %mul25.1 to i32
  %add29.1 = add nsw i32 %conv26, %conv26.1
  %30 = add nuw nsw i64 %26, 2
  %arrayidx19.2 = getelementptr inbounds double, double* %parts, i64 %30
  %31 = load double, double* %arrayidx19.2, align 8, !tbaa !2
  %div21.2 = fdiv double %31, 3.100000e+00
  %mul25.2 = fmul double %div21.2, 1.600000e+01
  %conv26.2 = fptosi double %mul25.2 to i32
  %add29.2 = add nsw i32 %add29.1, %conv26.2
  store i32 %add29.2, i32* %arrayidx12, align 4, !tbaa !6
  %exitcond151 = icmp eq i64 %iv.next, %wide.trip.count150
  br i1 %exitcond151, label %delete.notnull, label %for.body10, !llvm.loop !19

delete.notnull:                                   ; preds = %for.body10, %middle.block
  br i1 %cmp8135, label %invertdelete.notnull.loopexit, label %invertentry

invertentry:                                      ; preds = %invertfor.cond46.preheader.lr.ph, %entry, %delete.notnull
  tail call void @_ZdaPv(i8* nonnull %call)
  ret void

invertfor.cond46.preheader.lr.ph:                 ; preds = %invertfor.cond46.preheader
  %"arrayidx53.2'ipg_unwrap" = getelementptr inbounds double, double* %"parts'", i64 %_unwrap13
  %32 = load double, double* %"arrayidx53.2'ipg_unwrap", align 8
  %33 = fadd fast double %32, %46
  store double %33, double* %"arrayidx53.2'ipg_unwrap", align 8
  %"arrayidx53'ipg_unwrap" = getelementptr inbounds double, double* %"parts'", i64 %_unwrap12
  %34 = bitcast double* %"arrayidx53'ipg_unwrap" to <2 x double>*
  %35 = load <2 x double>, <2 x double>* %34, align 8
  %36 = fadd fast <2 x double> %35, %56
  %37 = bitcast double* %"arrayidx53'ipg_unwrap" to <2 x double>*
  store <2 x double> %36, <2 x double>* %37, align 8
  %38 = icmp eq i64 %"iv1'ac.0", 0
  %39 = fadd fast double %62, %64
  %40 = select i1 %38, double %62, double %39
  %41 = add nsw i64 %"iv1'ac.0", -1
  br i1 %38, label %invertentry, label %invertfor.cond.cleanup43

invertfor.cond46.preheader:                       ; preds = %invertfor.cond46.preheader, %invertfor.cond.cleanup43
  %"spec.select'de.0" = phi double [ %"spec.select'de.1", %invertfor.cond.cleanup43 ], [ %62, %invertfor.cond46.preheader ]
  %"out.0132'de.0" = phi double [ 0.000000e+00, %invertfor.cond.cleanup43 ], [ %64, %invertfor.cond46.preheader ]
  %"'de.0" = phi double [ 0.000000e+00, %invertfor.cond.cleanup43 ], [ %46, %invertfor.cond46.preheader ]
  %"iv3'ac.0" = phi i64 [ %_unwrap30, %invertfor.cond.cleanup43 ], [ %65, %invertfor.cond46.preheader ]
  %42 = phi <2 x double> [ zeroinitializer, %invertfor.cond.cleanup43 ], [ %56, %invertfor.cond46.preheader ]
  %arrayidx67_unwrap = getelementptr inbounds i32, i32* %4, i64 %"iv3'ac.0"
  %_unwrap10 = load i32, i32* %arrayidx67_unwrap, align 4, !tbaa !6, !invariant.group !20
  %cmp68_unwrap = icmp eq i32 %_unwrap9.pre, %_unwrap10
  %diffeadd69 = select i1 %cmp68_unwrap, double %"spec.select'de.0", double 0.000000e+00
  %diffeout.1128 = select i1 %cmp68_unwrap, double 0.000000e+00, double %"spec.select'de.0"
  %43 = fadd fast double %diffeout.1128, %diffeadd69
  %mul54_unwrap = mul i64 %"iv3'ac.0", 3
  %_unwrap15 = and i64 %mul54_unwrap, 4294967295
  %_unwrap16 = add nuw nsw i64 %_unwrap15, 2
  %arrayidx57.2_unwrap = getelementptr inbounds double, double* %parts, i64 %_unwrap16
  %_unwrap17 = load double, double* %arrayidx57.2_unwrap, align 8, !tbaa !2, !invariant.group !21
  %sub58.2_unwrap = fsub double %_unwrap14, %_unwrap17
  %44 = fadd fast double %sub58.2_unwrap, %sub58.2_unwrap
  %45 = fmul fast double %diffeadd69, %44
  %46 = fadd fast double %"'de.0", %45
  %"arrayidx57.2'ipg_unwrap" = getelementptr inbounds double, double* %"parts'", i64 %_unwrap16
  %47 = load double, double* %"arrayidx57.2'ipg_unwrap", align 8
  %48 = fsub fast double %47, %45
  store double %48, double* %"arrayidx57.2'ipg_unwrap", align 8
  %arrayidx57_unwrap = getelementptr inbounds double, double* %parts, i64 %_unwrap15
  %49 = bitcast double* %arrayidx57_unwrap to <2 x double>*
  %50 = load <2 x double>, <2 x double>* %49, align 8, !tbaa !2
  %51 = fsub <2 x double> %67, %50
  %52 = fadd fast <2 x double> %51, %51
  %53 = insertelement <2 x double> undef, double %diffeadd69, i32 0
  %54 = shufflevector <2 x double> %53, <2 x double> undef, <2 x i32> zeroinitializer
  %55 = fmul fast <2 x double> %54, %52
  %56 = fadd fast <2 x double> %42, %55
  %"arrayidx57'ipg_unwrap" = getelementptr inbounds double, double* %"parts'", i64 %_unwrap15
  %57 = bitcast double* %"arrayidx57'ipg_unwrap" to <2 x double>*
  %58 = load <2 x double>, <2 x double>* %57, align 8
  %59 = fsub fast <2 x double> %58, %55
  %60 = bitcast double* %"arrayidx57'ipg_unwrap" to <2 x double>*
  store <2 x double> %59, <2 x double>* %60, align 8
  %61 = icmp eq i64 %"iv3'ac.0", 0
  %62 = select i1 %61, double 0.000000e+00, double %43
  %63 = fadd fast double %"out.0132'de.0", %43
  %64 = select i1 %61, double %63, double %"out.0132'de.0"
  %65 = add nsw i64 %"iv3'ac.0", -1
  br i1 %61, label %invertfor.cond46.preheader.lr.ph, label %invertfor.cond46.preheader

invertfor.cond.cleanup43:                         ; preds = %invertfor.cond46.preheader.lr.ph, %invertdelete.notnull.loopexit
  %"spec.select'de.1" = phi double [ %differeturn, %invertdelete.notnull.loopexit ], [ %40, %invertfor.cond46.preheader.lr.ph ]
  %"iv1'ac.0" = phi i64 [ %_unwrap30, %invertdelete.notnull.loopexit ], [ %41, %invertfor.cond46.preheader.lr.ph ]
  %arrayidx65_unwrap.phi.trans.insert = getelementptr inbounds i32, i32* %4, i64 %"iv1'ac.0"
  %_unwrap9.pre = load i32, i32* %arrayidx65_unwrap.phi.trans.insert, align 4, !tbaa !6, !invariant.group !22
  %mul50_unwrap11 = mul i64 %"iv1'ac.0", 3
  %_unwrap12 = and i64 %mul50_unwrap11, 4294967295
  %_unwrap13 = add nuw nsw i64 %_unwrap12, 2
  %arrayidx53.2_unwrap = getelementptr inbounds double, double* %parts, i64 %_unwrap13
  %_unwrap14 = load double, double* %arrayidx53.2_unwrap, align 8, !tbaa !2, !invariant.group !23
  %arrayidx53_unwrap = getelementptr inbounds double, double* %parts, i64 %_unwrap12
  %66 = bitcast double* %arrayidx53_unwrap to <2 x double>*
  %67 = load <2 x double>, <2 x double>* %66, align 8, !tbaa !2
  br label %invertfor.cond46.preheader

invertdelete.notnull.loopexit:                    ; preds = %delete.notnull
  %wide.trip.count150_unwrap29 = zext i32 %n to i64
  %_unwrap30 = add nsw i64 %wide.trip.count150_unwrap29, -1
  br label %invertfor.cond.cleanup43
}

; Function Attrs: nounwind readnone speculatable
declare { i32, i1 } @llvm.umul.with.overflow.i32(i32, i32) #5

attributes #0 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }
attributes #3 = { uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { argmemonly nounwind }
attributes #5 = { nounwind readnone speculatable }
attributes #6 = { nobuiltin "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #7 = { nobuiltin nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #8 = { norecurse uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #9 = { builtin }
attributes #10 = { builtin nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0}
!3 = !{!"double", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C++ TBAA"}
!6 = !{!7, !7, i64 0}
!7 = !{!"int", !4, i64 0}
!8 = distinct !{!8, !9}
!9 = !{!"llvm.loop.isvectorized", i32 1}
!10 = distinct !{!10, !9}
!11 = !{!12, !12, i64 0}
!12 = !{!"long", !4, i64 0}
!13 = !{!14, !12, i64 4992}
!14 = !{!"_ZTSSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EE", !4, i64 0, !12, i64 4992}
!15 = !{!"branch_weights", i32 2000, i32 1}
!16 = distinct !{!16, !9}
!17 = distinct !{!17, !9}
!18 = distinct !{!18, !9}
!19 = distinct !{!19, !9}
!20 = distinct !{}
!21 = distinct !{}
!22 = distinct !{}
!23 = distinct !{}

; CHECK: define internal void @diffe_Z3fooPdi
; CHECK-NOT: malloc
