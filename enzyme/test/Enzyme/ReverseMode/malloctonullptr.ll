; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -simplifycfg -instcombine -early-cse -adce -S | FileCheck %s

source_filename = "/mnt/Data/git/Enzyme/enzyme/test/Integration/eigentensor.cpp"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }
%"struct.std::array.6" = type { [2 x i64] }
%"struct.std::array" = type { [4 x i64] }
%struct.timespec = type { i64, i64 }
%"class.Eigen::Tensor" = type { %"class.Eigen::TensorStorage" }
%"class.Eigen::TensorStorage" = type { float*, %"struct.Eigen::DSizes" }
%"struct.Eigen::DSizes" = type { %"struct.std::array" }
%"class.Eigen::Tensor.1" = type { %"class.Eigen::TensorStorage.4" }
%"class.Eigen::TensorStorage.4" = type { float*, %"struct.Eigen::DSizes.5" }
%"struct.Eigen::DSizes.5" = type { %"struct.std::array.6" }
%"struct.Eigen::TensorEvaluator.9" = type { float*, %"struct.Eigen::DSizes", %"struct.Eigen::DefaultDevice"*, %"class.Eigen::Tensor"* }
%"struct.Eigen::DefaultDevice" = type { i8 }
%"struct.Eigen::TensorEvaluator.10" = type { %"struct.std::array", %"struct.std::array", %"struct.std::array.6", %"struct.std::array.6", %"struct.Eigen::TensorEvaluator.11", %"struct.Eigen::TensorEvaluator.12", %"struct.Eigen::DSizes", %"class.Eigen::Tensor.1", float*, i8, %"struct.Eigen::DefaultDevice"* }
%"struct.Eigen::TensorEvaluator.11" = type { float*, %"struct.Eigen::DSizes", %"struct.Eigen::DefaultDevice"*, %"class.Eigen::Tensor"* }
%"struct.Eigen::TensorEvaluator.12" = type { float*, %"struct.Eigen::DSizes.5", %"struct.Eigen::DefaultDevice"*, %"class.Eigen::Tensor.1"* }

@stderr = external dso_local local_unnamed_addr global %struct._IO_FILE*, align 8
@.str.1 = private unnamed_addr constant [60 x i8] c"kernelp(si=%d, sj=%d)=%f, expected_kernel(si=%d, sj=%d)=%f\0A\00", align 1
@.str.2 = private unnamed_addr constant [68 x i8] c"Assertion Failed: fabs( [%s = %g] - [%s = %g] ) > %g at %s:%d (%s)\0A\00", align 1
@.str.3 = private unnamed_addr constant [16 x i8] c"kernelp(si, sj)\00", align 1
@.str.4 = private unnamed_addr constant [24 x i8] c"expected_kernel(si, sj)\00", align 1
@.str.5 = private unnamed_addr constant [61 x i8] c"/mnt/Data/git/Enzyme/enzyme/test/Integration/eigentensor.cpp\00", align 1
@__PRETTY_FUNCTION__.main = private unnamed_addr constant [23 x i8] c"int main(int, char **)\00", align 1
@_ZZL6matvecPKN5Eigen6TensorIfLi2ELi0ElEEPKNS0_IfLi4ELi0ElEEPS4_E4dims = private unnamed_addr constant %"struct.std::array.6" { [2 x i64] [i64 1, i64 2] }, align 8
@str = private unnamed_addr constant [13 x i8] c"did original\00"

; Function Attrs: norecurse nounwind uwtable
define dso_local void @_Z6memcpyPfS_m(float* noalias nocapture %dst, float* noalias nocapture readonly %src, i64 %count) local_unnamed_addr #0 {
entry:
  %cmp6 = icmp ult i64 %count, 4
  br i1 %cmp6, label %for.cond.cleanup, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  %src9 = bitcast float* %src to i8*
  %dst8 = bitcast float* %dst to i8*
  %0 = and i64 %count, -4
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %dst8, i8* align 4 %src9, i64 %0, i1 false)
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.body.preheader, %entry
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

; Function Attrs: norecurse nounwind uwtable
define dso_local void @_Z6memcpyPdS_m(double* noalias nocapture %dst, double* noalias nocapture readonly %src, i64 %count) local_unnamed_addr #0 {
entry:
  %cmp6 = icmp ult i64 %count, 8
  br i1 %cmp6, label %for.cond.cleanup, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  %src9 = bitcast double* %src to i8*
  %dst8 = bitcast double* %dst to i8*
  %0 = and i64 %count, -8
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %dst8, i8* align 8 %src9, i64 %0, i1 false)
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.body.preheader, %entry
  ret void
}

; Function Attrs: norecurse nounwind uwtable
define dso_local i32 @main(i32 %argc, i8** nocapture readnone %argv) local_unnamed_addr #0 {
entry:
  %dims.i.i.i.i361 = alloca <2 x i64>, align 16
  %tmpcast = bitcast <2 x i64>* %dims.i.i.i.i361 to %"struct.std::array.6"*
  %dims.i.i.i290 = alloca %"struct.std::array", align 8
  %ts.i.i.i.i.i291 = alloca %struct.timespec, align 8
  %dims.i.i.i.i267 = alloca %"struct.std::array.6", align 8
  %dims.i.i.i.i = alloca %"struct.std::array", align 8
  %dims.i.i.i205 = alloca %"struct.std::array.6", align 8
  %ts.i.i.i.i.i206 = alloca %struct.timespec, align 8
  %dims.i.i.i = alloca %"struct.std::array", align 8
  %ts.i.i.i.i.i = alloca %struct.timespec, align 8
  %input = alloca %"class.Eigen::Tensor", align 8
  %kernel = alloca %"class.Eigen::Tensor.1", align 8
  %output = alloca %"class.Eigen::Tensor", align 8
  %inputp = alloca %"class.Eigen::Tensor", align 8
  %kernelp = alloca %"class.Eigen::Tensor.1", align 8
  %outputp = alloca %"class.Eigen::Tensor", align 8
  %0 = bitcast %"class.Eigen::Tensor"* %input to i8*
  call void @llvm.lifetime.start.p0i8(i64 40, i8* nonnull %0) #7
  %ref.tmp.sroa.0.0..sroa_idx.i.i.i = getelementptr inbounds %"class.Eigen::Tensor", %"class.Eigen::Tensor"* %input, i64 0, i32 0, i32 1, i32 0, i32 0, i64 0
  %ref.tmp.sroa.4.0..sroa_idx8.i.i.i = getelementptr inbounds %"class.Eigen::Tensor", %"class.Eigen::Tensor"* %input, i64 0, i32 0, i32 1, i32 0, i32 0, i64 1
  %1 = bitcast i64* %ref.tmp.sroa.0.0..sroa_idx.i.i.i to <2 x i64>*
  store <2 x i64> <i64 3, i64 3>, <2 x i64>* %1, align 8, !tbaa !2
  %ref.tmp.sroa.5.0..sroa_idx10.i.i.i = getelementptr inbounds %"class.Eigen::Tensor", %"class.Eigen::Tensor"* %input, i64 0, i32 0, i32 1, i32 0, i32 0, i64 2
  %ref.tmp.sroa.6.0..sroa_idx12.i.i.i = getelementptr inbounds %"class.Eigen::Tensor", %"class.Eigen::Tensor"* %input, i64 0, i32 0, i32 1, i32 0, i32 0, i64 3
  %2 = bitcast i64* %ref.tmp.sroa.5.0..sroa_idx10.i.i.i to <2 x i64>*
  store <2 x i64> <i64 7, i64 11>, <2 x i64>* %2, align 8, !tbaa !2
  %call.i.i.i.i.i = tail call noalias i8* @malloc(i64 2772) #7
  %3 = bitcast %"class.Eigen::Tensor"* %input to i8**
  store i8* %call.i.i.i.i.i, i8** %3, align 8, !tbaa !7
  %4 = bitcast %"class.Eigen::Tensor.1"* %kernel to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %4) #7
  %ref.tmp.sroa.0.0..sroa_idx.i.i.i195 = getelementptr inbounds %"class.Eigen::Tensor.1", %"class.Eigen::Tensor.1"* %kernel, i64 0, i32 0, i32 1, i32 0, i32 0, i64 0
  %ref.tmp.sroa.4.0..sroa_idx3.i.i.i = getelementptr inbounds %"class.Eigen::Tensor.1", %"class.Eigen::Tensor.1"* %kernel, i64 0, i32 0, i32 1, i32 0, i32 0, i64 1
  %5 = bitcast i64* %ref.tmp.sroa.0.0..sroa_idx.i.i.i195 to <2 x i64>*
  store <2 x i64> <i64 2, i64 2>, <2 x i64>* %5, align 8, !tbaa !11
  %call.i.i.i.i.i196 = tail call noalias i8* @malloc(i64 16) #7
  %6 = bitcast %"class.Eigen::Tensor.1"* %kernel to i8**
  store i8* %call.i.i.i.i.i196, i8** %6, align 8, !tbaa !13
  %7 = bitcast %"class.Eigen::Tensor"* %output to i8*
  call void @llvm.lifetime.start.p0i8(i64 40, i8* nonnull %7) #7
  %ref.tmp.sroa.0.0..sroa_idx.i.i.i198 = getelementptr inbounds %"class.Eigen::Tensor", %"class.Eigen::Tensor"* %output, i64 0, i32 0, i32 1, i32 0, i32 0, i64 0
  %8 = bitcast i64* %ref.tmp.sroa.0.0..sroa_idx.i.i.i198 to <2 x i64>*
  store <2 x i64> <i64 3, i64 2>, <2 x i64>* %8, align 8, !tbaa !2
  %ref.tmp.sroa.5.0..sroa_idx10.i.i.i200 = getelementptr inbounds %"class.Eigen::Tensor", %"class.Eigen::Tensor"* %output, i64 0, i32 0, i32 1, i32 0, i32 0, i64 2
  %9 = bitcast i64* %ref.tmp.sroa.5.0..sroa_idx10.i.i.i200 to <2 x i64>*
  store <2 x i64> <i64 6, i64 11>, <2 x i64>* %9, align 8, !tbaa !2
  %call.i.i.i.i.i202 = tail call noalias i8* @malloc(i64 1584) #7
  %10 = bitcast %"class.Eigen::Tensor"* %output to i8**
  store i8* %call.i.i.i.i.i202, i8** %10, align 8, !tbaa !7
  %11 = bitcast %struct.timespec* %ts.i.i.i.i.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %11) #7, !noalias !16
  %call.i.i.i.i.i204 = call i32 @clock_gettime(i32 0, %struct.timespec* nonnull %ts.i.i.i.i.i) #7, !noalias !16
  %call1.i.i.i.i.i = call i64 @random() #7, !noalias !16
  %tv_nsec.i.i.i.i.i = getelementptr inbounds %struct.timespec, %struct.timespec* %ts.i.i.i.i.i, i64 0, i32 1
  %12 = load i64, i64* %tv_nsec.i.i.i.i.i, align 8, !tbaa !19, !noalias !16
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %11) #7, !noalias !16
  %ref.tmp.sroa.8.72..sroa_cast.i.i = getelementptr inbounds %"class.Eigen::Tensor", %"class.Eigen::Tensor"* %input, i64 0, i32 0, i32 1
  %13 = bitcast %"struct.Eigen::DSizes"* %ref.tmp.sroa.8.72..sroa_cast.i.i to i8*
  %14 = bitcast %"struct.std::array"* %dims.i.i.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %14) #7
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 %14, i8* nonnull align 8 %13, i64 32, i1 false) #7
  %15 = bitcast i8* %call.i.i.i.i.i to float*
  %arrayidx.i.i.i.i.i.i = getelementptr inbounds %"struct.std::array", %"struct.std::array"* %dims.i.i.i, i64 0, i32 0, i64 0
  %16 = load i64, i64* %arrayidx.i.i.i.i.i.i, align 8, !tbaa !21
  %arrayidx.i.i.i.i.i.i.1 = getelementptr inbounds %"struct.std::array", %"struct.std::array"* %dims.i.i.i, i64 0, i32 0, i64 1
  %17 = load i64, i64* %arrayidx.i.i.i.i.i.i.1, align 8, !tbaa !21
  %mul.i.i.i3.i.1 = mul nsw i64 %17, %16
  %arrayidx.i.i.i.i.i.i.2 = getelementptr inbounds %"struct.std::array", %"struct.std::array"* %dims.i.i.i, i64 0, i32 0, i64 2
  %18 = load i64, i64* %arrayidx.i.i.i.i.i.i.2, align 8, !tbaa !21
  %mul.i.i.i3.i.2 = mul nsw i64 %18, %mul.i.i.i3.i.1
  %arrayidx.i.i.i.i.i.i.3 = getelementptr inbounds %"struct.std::array", %"struct.std::array"* %dims.i.i.i, i64 0, i32 0, i64 3
  %19 = load i64, i64* %arrayidx.i.i.i.i.i.i.3, align 8, !tbaa !21
  %mul.i.i.i3.i.3 = mul nsw i64 %19, %mul.i.i.i3.i.2
  %m_data.i.i = getelementptr inbounds %"class.Eigen::Tensor", %"class.Eigen::Tensor"* %input, i64 0, i32 0, i32 0
  %xor.i.i.i.i.i = xor i64 %12, %call1.i.i.i.i.i
  %agg.tmp.sroa.0.sroa.0.0.agg.tmp.sroa.0.0..sroa_cast.sroa_idx.i.i.i.i.i.i.i = getelementptr inbounds %"struct.Eigen::DSizes", %"struct.Eigen::DSizes"* %ref.tmp.sroa.8.72..sroa_cast.i.i, i64 0, i32 0, i32 0, i64 0
  %cmp.i.i.i.i.i = icmp eq i64 %mul.i.i.i3.i.3, 693
  br i1 %cmp.i.i.i.i.i, label %_ZN5Eigen6TensorIfLi4ELi0ElE6resizeERKNS_6DSizesIlLi4EEE.exit.i.i, label %if.then.i.i.i.i.i

if.then.i.i.i.i.i:                                ; preds = %entry
  call void @free(i8* %call.i.i.i.i.i) #7
  %tobool.i.i.i.i.i = icmp eq i64 %mul.i.i.i3.i.3, 0
  br i1 %tobool.i.i.i.i.i, label %if.else.i.i.i.i.i, label %_ZN5Eigen8internal28conditional_aligned_new_autoIfLb1EEEPT_m.exit.i.i.i.i.i

_ZN5Eigen8internal28conditional_aligned_new_autoIfLb1EEEPT_m.exit.i.i.i.i.i: ; preds = %if.then.i.i.i.i.i
  %mul.i.i.i.i.i.i = shl i64 %mul.i.i.i3.i.3, 2
  %call.i.i.i.i.i.i.i.i = call noalias i8* @malloc(i64 %mul.i.i.i.i.i.i) #7
  store i8* %call.i.i.i.i.i.i.i.i, i8** %3, align 8, !tbaa !7
  %20 = bitcast i8* %call.i.i.i.i.i.i.i.i to float*
  br label %_ZN5Eigen6TensorIfLi4ELi0ElE6resizeERKNS_6DSizesIlLi4EEE.exit.i.i

if.else.i.i.i.i.i:                                ; preds = %if.then.i.i.i.i.i
  store float* null, float** %m_data.i.i, align 8, !tbaa !7
  br label %_ZN5Eigen6TensorIfLi4ELi0ElE6resizeERKNS_6DSizesIlLi4EEE.exit.i.i

_ZN5Eigen6TensorIfLi4ELi0ElE6resizeERKNS_6DSizesIlLi4EEE.exit.i.i: ; preds = %if.else.i.i.i.i.i, %_ZN5Eigen8internal28conditional_aligned_new_autoIfLb1EEEPT_m.exit.i.i.i.i.i, %entry
  %21 = phi float* [ %20, %_ZN5Eigen8internal28conditional_aligned_new_autoIfLb1EEEPT_m.exit.i.i.i.i.i ], [ null, %if.else.i.i.i.i.i ], [ %15, %entry ]
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 %13, i8* nonnull align 8 %14, i64 32, i1 false) #7, !tbaa !2, !tbaa.struct !22
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %14) #7
  %evaluator.sroa.11.72.copyload.i.i.i = load i64, i64* %agg.tmp.sroa.0.sroa.0.0.agg.tmp.sroa.0.0..sroa_cast.sroa_idx.i.i.i.i.i.i.i, align 8, !tbaa !24
  %evaluator.sroa.13.72.copyload.i.i.i = load i64, i64* %ref.tmp.sroa.4.0..sroa_idx8.i.i.i, align 8, !tbaa !24
  %evaluator.sroa.14.72.copyload.i.i.i = load i64, i64* %ref.tmp.sroa.5.0..sroa_idx10.i.i.i, align 8, !tbaa !24
  %evaluator.sroa.15.72.copyload.i.i.i = load i64, i64* %ref.tmp.sroa.6.0..sroa_idx12.i.i.i, align 8, !tbaa !24
  %mul.i.i.i.i.i.i7.i.i.i = mul i64 %evaluator.sroa.13.72.copyload.i.i.i, %evaluator.sroa.11.72.copyload.i.i.i
  %mul.i.i.i.i.i.i.i.i = mul i64 %mul.i.i.i.i.i.i7.i.i.i, %evaluator.sroa.14.72.copyload.i.i.i
  %mul.i.i.i.i.i.i.i = mul i64 %mul.i.i.i.i.i.i.i.i, %evaluator.sroa.15.72.copyload.i.i.i
  %cmp28.i.i.i = icmp sgt i64 %mul.i.i.i.i.i.i.i, 0
  br i1 %cmp28.i.i.i, label %for.body.i.i.i.preheader, label %_ZN5Eigen10TensorBaseINS_6TensorIfLi4ELi0ElEELi1EE9setRandomEv.exit

for.body.i.i.i.preheader:                         ; preds = %_ZN5Eigen6TensorIfLi4ELi0ElE6resizeERKNS_6DSizesIlLi4EEE.exit.i.i
  br label %for.body.i.i.i

for.body.i.i.i:                                   ; preds = %for.body.i.i.i.preheader, %for.body.i.i.i
  %i.030.i.i.i = phi i64 [ %inc.i.i.i, %for.body.i.i.i ], [ 0, %for.body.i.i.i.preheader ]
  %evaluator.sroa.7.029.i.i.in.in.i = phi i64 [ %add.i.i.i.i.i.i.i, %for.body.i.i.i ], [ %xor.i.i.i.i.i, %for.body.i.i.i.preheader ]
  %evaluator.sroa.7.029.i.i.in.i = mul i64 %evaluator.sroa.7.029.i.i.in.in.i, 6364136223846793005
  %evaluator.sroa.7.029.i.i.i = add nsw i64 %i.030.i.i.i, -2720673578348880933
  %add.i.i.i.i.i.i.i = add i64 %evaluator.sroa.7.029.i.i.i, %evaluator.sroa.7.029.i.i.in.i
  %shr.i.i.i.i.i.i.i.i.i = lshr i64 %add.i.i.i.i.i.i.i, 22
  %xor.i.i.i.i.i.i.i.i.i = xor i64 %shr.i.i.i.i.i.i.i.i.i, %add.i.i.i.i.i.i.i
  %shr1.i.i.i.i.i.i.i.i.i = lshr i64 %add.i.i.i.i.i.i.i, 61
  %add2.i.i.i.i.i.i.i.i.i = add nuw nsw i64 %shr1.i.i.i.i.i.i.i.i.i, 22
  %shr3.i.i.i.i.i.i.i.i.i = lshr i64 %xor.i.i.i.i.i.i.i.i.i, %add2.i.i.i.i.i.i.i.i.i
  %conv.i.i.i.i.i.i.i.i.i = trunc i64 %shr3.i.i.i.i.i.i.i.i.i to i32
  %and.i.i.i.i.i.i.i.i = and i32 %conv.i.i.i.i.i.i.i.i.i, 8388607
  %or.i.i.i.i.i.i.i.i = or i32 %and.i.i.i.i.i.i.i.i, 1065353216
  %22 = bitcast i32 %or.i.i.i.i.i.i.i.i to float
  %sub.i.i.i.i.i.i.i.i = fadd fast float %22, -1.000000e+00
  %arrayidx.i.i.i.i.i = getelementptr inbounds float, float* %21, i64 %i.030.i.i.i
  store float %sub.i.i.i.i.i.i.i.i, float* %arrayidx.i.i.i.i.i, align 4, !tbaa !25
  %inc.i.i.i = add nuw nsw i64 %i.030.i.i.i, 1
  %exitcond.i.i.i = icmp eq i64 %inc.i.i.i, %mul.i.i.i.i.i.i.i
  br i1 %exitcond.i.i.i, label %_ZN5Eigen10TensorBaseINS_6TensorIfLi4ELi0ElEELi1EE9setRandomEv.exit.loopexit, label %for.body.i.i.i

_ZN5Eigen10TensorBaseINS_6TensorIfLi4ELi0ElEELi1EE9setRandomEv.exit.loopexit: ; preds = %for.body.i.i.i
  br label %_ZN5Eigen10TensorBaseINS_6TensorIfLi4ELi0ElEELi1EE9setRandomEv.exit

_ZN5Eigen10TensorBaseINS_6TensorIfLi4ELi0ElEELi1EE9setRandomEv.exit: ; preds = %_ZN5Eigen10TensorBaseINS_6TensorIfLi4ELi0ElEELi1EE9setRandomEv.exit.loopexit, %_ZN5Eigen6TensorIfLi4ELi0ElE6resizeERKNS_6DSizesIlLi4EEE.exit.i.i
  %23 = bitcast %struct.timespec* %ts.i.i.i.i.i206 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %23) #7, !noalias !27
  %call.i.i.i.i.i207 = call i32 @clock_gettime(i32 0, %struct.timespec* nonnull %ts.i.i.i.i.i206) #7, !noalias !27
  %call1.i.i.i.i.i208 = call i64 @random() #7, !noalias !27
  %tv_nsec.i.i.i.i.i209 = getelementptr inbounds %struct.timespec, %struct.timespec* %ts.i.i.i.i.i206, i64 0, i32 1
  %24 = load i64, i64* %tv_nsec.i.i.i.i.i209, align 8, !tbaa !19, !noalias !27
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %23) #7, !noalias !27
  %ref.tmp.sroa.8.56..sroa_cast.i.i = getelementptr inbounds %"class.Eigen::Tensor.1", %"class.Eigen::Tensor.1"* %kernel, i64 0, i32 0, i32 1
  %25 = bitcast %"struct.Eigen::DSizes.5"* %ref.tmp.sroa.8.56..sroa_cast.i.i to i8*
  %26 = bitcast %"struct.std::array.6"* %dims.i.i.i205 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %26) #7
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 %26, i8* nonnull align 8 %25, i64 16, i1 false) #7
  %arrayidx.i.i.i.i.i.i213 = getelementptr inbounds %"struct.std::array.6", %"struct.std::array.6"* %dims.i.i.i205, i64 0, i32 0, i64 0
  %27 = load i64, i64* %arrayidx.i.i.i.i.i.i213, align 8, !tbaa !21
  %arrayidx.i.i.i.i.i.i213.1 = getelementptr inbounds %"struct.std::array.6", %"struct.std::array.6"* %dims.i.i.i205, i64 0, i32 0, i64 1
  %28 = load i64, i64* %arrayidx.i.i.i.i.i.i213.1, align 8, !tbaa !21
  %mul.i.i.i3.i214.1 = mul nsw i64 %28, %27
  %xor.i.i.i.i.i218 = xor i64 %24, %call1.i.i.i.i.i208
  %agg.tmp.sroa.0.0..sroa_idx.i.i.i.i.i.i.i = getelementptr inbounds %"struct.Eigen::DSizes.5", %"struct.Eigen::DSizes.5"* %ref.tmp.sroa.8.56..sroa_cast.i.i, i64 0, i32 0, i32 0, i64 0
  %agg.tmp.sroa.0.0.copyload.i.i.i.i.i.i.i = load i64, i64* %agg.tmp.sroa.0.0..sroa_idx.i.i.i.i.i.i.i, align 8, !tbaa !11
  %agg.tmp.sroa.2.0.copyload.i.i.i.i.i.i.i = load i64, i64* %ref.tmp.sroa.4.0..sroa_idx3.i.i.i, align 8, !tbaa !11
  %mul.i.i.i.i.i.i.i4.i.i220 = mul nsw i64 %agg.tmp.sroa.2.0.copyload.i.i.i.i.i.i.i, %agg.tmp.sroa.0.0.copyload.i.i.i.i.i.i.i
  %cmp.i.i.i.i.i221 = icmp eq i64 %mul.i.i.i.i.i.i.i4.i.i220, %mul.i.i.i3.i214.1
  br i1 %cmp.i.i.i.i.i221, label %for.end.i.i._ZN5Eigen6TensorIfLi2ELi0ElE6resizeERKNS_6DSizesIlLi2EEE.exit_crit_edge.i.i, label %if.then.i.i.i.i.i226

for.end.i.i._ZN5Eigen6TensorIfLi2ELi0ElE6resizeERKNS_6DSizesIlLi2EEE.exit_crit_edge.i.i: ; preds = %_ZN5Eigen10TensorBaseINS_6TensorIfLi4ELi0ElEELi1EE9setRandomEv.exit
  %.phi.trans.insert.i.i223 = getelementptr inbounds %"class.Eigen::Tensor.1", %"class.Eigen::Tensor.1"* %kernel, i64 0, i32 0, i32 0
  %.pre.i.i224 = load float*, float** %.phi.trans.insert.i.i223, align 8, !tbaa !13
  br label %_ZN5Eigen6TensorIfLi2ELi0ElE6resizeERKNS_6DSizesIlLi2EEE.exit.i.i

if.then.i.i.i.i.i226:                             ; preds = %_ZN5Eigen10TensorBaseINS_6TensorIfLi4ELi0ElEELi1EE9setRandomEv.exit
  %29 = load i8*, i8** %6, align 8, !tbaa !13
  call void @free(i8* %29) #7
  %tobool.i.i.i.i.i225 = icmp eq i64 %mul.i.i.i3.i214.1, 0
  br i1 %tobool.i.i.i.i.i225, label %if.else.i.i.i.i.i231, label %_ZN5Eigen8internal28conditional_aligned_new_autoIfLb1EEEPT_m.exit.i.i.i.i.i229

_ZN5Eigen8internal28conditional_aligned_new_autoIfLb1EEEPT_m.exit.i.i.i.i.i229: ; preds = %if.then.i.i.i.i.i226
  %mul.i.i.i.i.i.i227 = shl i64 %mul.i.i.i3.i214.1, 2
  %call.i.i.i.i.i.i.i.i228 = call noalias i8* @malloc(i64 %mul.i.i.i.i.i.i227) #7
  store i8* %call.i.i.i.i.i.i.i.i228, i8** %6, align 8, !tbaa !13
  %30 = bitcast i8* %call.i.i.i.i.i.i.i.i228 to float*
  br label %_ZN5Eigen6TensorIfLi2ELi0ElE6resizeERKNS_6DSizesIlLi2EEE.exit.i.i

if.else.i.i.i.i.i231:                             ; preds = %if.then.i.i.i.i.i226
  %m_data.i.i.i.i.i230 = getelementptr inbounds %"class.Eigen::Tensor.1", %"class.Eigen::Tensor.1"* %kernel, i64 0, i32 0, i32 0
  store float* null, float** %m_data.i.i.i.i.i230, align 8, !tbaa !13
  br label %_ZN5Eigen6TensorIfLi2ELi0ElE6resizeERKNS_6DSizesIlLi2EEE.exit.i.i

_ZN5Eigen6TensorIfLi2ELi0ElE6resizeERKNS_6DSizesIlLi2EEE.exit.i.i: ; preds = %if.else.i.i.i.i.i231, %_ZN5Eigen8internal28conditional_aligned_new_autoIfLb1EEEPT_m.exit.i.i.i.i.i229, %for.end.i.i._ZN5Eigen6TensorIfLi2ELi0ElE6resizeERKNS_6DSizesIlLi2EEE.exit_crit_edge.i.i
  %31 = phi float* [ %.pre.i.i224, %for.end.i.i._ZN5Eigen6TensorIfLi2ELi0ElE6resizeERKNS_6DSizesIlLi2EEE.exit_crit_edge.i.i ], [ %30, %_ZN5Eigen8internal28conditional_aligned_new_autoIfLb1EEEPT_m.exit.i.i.i.i.i229 ], [ null, %if.else.i.i.i.i.i231 ]
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 %25, i8* nonnull align 8 %26, i64 16, i1 false) #7, !tbaa !11, !tbaa.struct !30
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %26) #7
  %evaluator.sroa.11.56.copyload.i.i.i = load i64, i64* %agg.tmp.sroa.0.0..sroa_idx.i.i.i.i.i.i.i, align 8, !tbaa !32
  %evaluator.sroa.13.56.copyload.i.i.i = load i64, i64* %ref.tmp.sroa.4.0..sroa_idx3.i.i.i, align 8, !tbaa !32
  %mul.i.i.i.i.i.i.i232 = mul i64 %evaluator.sroa.13.56.copyload.i.i.i, %evaluator.sroa.11.56.copyload.i.i.i
  %cmp23.i.i.i = icmp sgt i64 %mul.i.i.i.i.i.i.i232, 0
  br i1 %cmp23.i.i.i, label %for.body.i.i.i246.preheader, label %_ZN5Eigen10TensorBaseINS_6TensorIfLi2ELi0ElEELi1EE9setRandomEv.exit

for.body.i.i.i246.preheader:                      ; preds = %_ZN5Eigen6TensorIfLi2ELi0ElE6resizeERKNS_6DSizesIlLi2EEE.exit.i.i
  br label %for.body.i.i.i246

for.body.i.i.i246:                                ; preds = %for.body.i.i.i246.preheader, %for.body.i.i.i246
  %i.025.i.i.i = phi i64 [ %inc.i.i.i244, %for.body.i.i.i246 ], [ 0, %for.body.i.i.i246.preheader ]
  %evaluator.sroa.7.024.i.i.in.in.i = phi i64 [ %add.i.i.i.i.i.i.i233, %for.body.i.i.i246 ], [ %xor.i.i.i.i.i218, %for.body.i.i.i246.preheader ]
  %evaluator.sroa.7.024.i.i.in.i = mul i64 %evaluator.sroa.7.024.i.i.in.in.i, 6364136223846793005
  %evaluator.sroa.7.024.i.i.i = add nsw i64 %i.025.i.i.i, -2720673578348880933
  %add.i.i.i.i.i.i.i233 = add i64 %evaluator.sroa.7.024.i.i.i, %evaluator.sroa.7.024.i.i.in.i
  %shr.i.i.i.i.i.i.i.i.i234 = lshr i64 %add.i.i.i.i.i.i.i233, 22
  %xor.i.i.i.i.i.i.i.i.i235 = xor i64 %shr.i.i.i.i.i.i.i.i.i234, %add.i.i.i.i.i.i.i233
  %shr1.i.i.i.i.i.i.i.i.i236 = lshr i64 %add.i.i.i.i.i.i.i233, 61
  %add2.i.i.i.i.i.i.i.i.i237 = add nuw nsw i64 %shr1.i.i.i.i.i.i.i.i.i236, 22
  %shr3.i.i.i.i.i.i.i.i.i238 = lshr i64 %xor.i.i.i.i.i.i.i.i.i235, %add2.i.i.i.i.i.i.i.i.i237
  %conv.i.i.i.i.i.i.i.i.i239 = trunc i64 %shr3.i.i.i.i.i.i.i.i.i238 to i32
  %and.i.i.i.i.i.i.i.i240 = and i32 %conv.i.i.i.i.i.i.i.i.i239, 8388607
  %or.i.i.i.i.i.i.i.i241 = or i32 %and.i.i.i.i.i.i.i.i240, 1065353216
  %32 = bitcast i32 %or.i.i.i.i.i.i.i.i241 to float
  %sub.i.i.i.i.i.i.i.i242 = fadd fast float %32, -1.000000e+00
  %arrayidx.i.i.i.i.i243 = getelementptr inbounds float, float* %31, i64 %i.025.i.i.i
  store float %sub.i.i.i.i.i.i.i.i242, float* %arrayidx.i.i.i.i.i243, align 4, !tbaa !25
  %inc.i.i.i244 = add nuw nsw i64 %i.025.i.i.i, 1
  %exitcond.i.i.i245 = icmp eq i64 %inc.i.i.i244, %mul.i.i.i.i.i.i.i232
  br i1 %exitcond.i.i.i245, label %_ZN5Eigen10TensorBaseINS_6TensorIfLi2ELi0ElEELi1EE9setRandomEv.exit.loopexit, label %for.body.i.i.i246

_ZN5Eigen10TensorBaseINS_6TensorIfLi2ELi0ElEELi1EE9setRandomEv.exit.loopexit: ; preds = %for.body.i.i.i246
  br label %_ZN5Eigen10TensorBaseINS_6TensorIfLi2ELi0ElEELi1EE9setRandomEv.exit

_ZN5Eigen10TensorBaseINS_6TensorIfLi2ELi0ElEELi1EE9setRandomEv.exit: ; preds = %_ZN5Eigen10TensorBaseINS_6TensorIfLi2ELi0ElEELi1EE9setRandomEv.exit.loopexit, %_ZN5Eigen6TensorIfLi2ELi0ElE6resizeERKNS_6DSizesIlLi2EEE.exit.i.i
  %33 = bitcast %"class.Eigen::Tensor"* %inputp to i8*
  call void @llvm.lifetime.start.p0i8(i64 40, i8* nonnull %33) #7
  %ref.tmp.sroa.0.0..sroa_idx.i.i.i247 = getelementptr inbounds %"class.Eigen::Tensor", %"class.Eigen::Tensor"* %inputp, i64 0, i32 0, i32 1, i32 0, i32 0, i64 0
  %ref.tmp.sroa.4.0..sroa_idx8.i.i.i248 = getelementptr inbounds %"class.Eigen::Tensor", %"class.Eigen::Tensor"* %inputp, i64 0, i32 0, i32 1, i32 0, i32 0, i64 1
  %34 = bitcast i64* %ref.tmp.sroa.0.0..sroa_idx.i.i.i247 to <2 x i64>*
  store <2 x i64> <i64 3, i64 3>, <2 x i64>* %34, align 8, !tbaa !2
  %ref.tmp.sroa.5.0..sroa_idx10.i.i.i249 = getelementptr inbounds %"class.Eigen::Tensor", %"class.Eigen::Tensor"* %inputp, i64 0, i32 0, i32 1, i32 0, i32 0, i64 2
  %ref.tmp.sroa.6.0..sroa_idx12.i.i.i250 = getelementptr inbounds %"class.Eigen::Tensor", %"class.Eigen::Tensor"* %inputp, i64 0, i32 0, i32 1, i32 0, i32 0, i64 3
  %35 = bitcast i64* %ref.tmp.sroa.5.0..sroa_idx10.i.i.i249 to <2 x i64>*
  store <2 x i64> <i64 7, i64 11>, <2 x i64>* %35, align 8, !tbaa !2
  %call.i.i.i.i.i251 = call noalias i8* @malloc(i64 2772) #7
  %36 = bitcast %"class.Eigen::Tensor"* %inputp to i8**
  store i8* %call.i.i.i.i.i251, i8** %36, align 8, !tbaa !7
  %37 = bitcast %"class.Eigen::Tensor.1"* %kernelp to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %37) #7
  %ref.tmp.sroa.0.0..sroa_idx.i.i.i253 = getelementptr inbounds %"class.Eigen::Tensor.1", %"class.Eigen::Tensor.1"* %kernelp, i64 0, i32 0, i32 1, i32 0, i32 0, i64 0
  %ref.tmp.sroa.4.0..sroa_idx3.i.i.i254 = getelementptr inbounds %"class.Eigen::Tensor.1", %"class.Eigen::Tensor.1"* %kernelp, i64 0, i32 0, i32 1, i32 0, i32 0, i64 1
  %38 = bitcast i64* %ref.tmp.sroa.0.0..sroa_idx.i.i.i253 to <2 x i64>*
  store <2 x i64> <i64 2, i64 2>, <2 x i64>* %38, align 8, !tbaa !11
  %call.i.i.i.i.i255 = call noalias i8* @malloc(i64 16) #7
  %39 = bitcast %"class.Eigen::Tensor.1"* %kernelp to i8**
  store i8* %call.i.i.i.i.i255, i8** %39, align 8, !tbaa !13
  %40 = bitcast %"class.Eigen::Tensor"* %outputp to i8*
  call void @llvm.lifetime.start.p0i8(i64 40, i8* nonnull %40) #7
  %ref.tmp.sroa.0.0..sroa_idx.i.i.i257 = getelementptr inbounds %"class.Eigen::Tensor", %"class.Eigen::Tensor"* %outputp, i64 0, i32 0, i32 1, i32 0, i32 0, i64 0
  %ref.tmp.sroa.4.0..sroa_idx8.i.i.i258 = getelementptr inbounds %"class.Eigen::Tensor", %"class.Eigen::Tensor"* %outputp, i64 0, i32 0, i32 1, i32 0, i32 0, i64 1
  %41 = bitcast i64* %ref.tmp.sroa.0.0..sroa_idx.i.i.i257 to <2 x i64>*
  store <2 x i64> <i64 3, i64 2>, <2 x i64>* %41, align 8, !tbaa !2
  %ref.tmp.sroa.5.0..sroa_idx10.i.i.i259 = getelementptr inbounds %"class.Eigen::Tensor", %"class.Eigen::Tensor"* %outputp, i64 0, i32 0, i32 1, i32 0, i32 0, i64 2
  %ref.tmp.sroa.6.0..sroa_idx12.i.i.i260 = getelementptr inbounds %"class.Eigen::Tensor", %"class.Eigen::Tensor"* %outputp, i64 0, i32 0, i32 1, i32 0, i32 0, i64 3
  %42 = bitcast i64* %ref.tmp.sroa.5.0..sroa_idx10.i.i.i259 to <2 x i64>*
  store <2 x i64> <i64 6, i64 11>, <2 x i64>* %42, align 8, !tbaa !2
  %call.i.i.i.i.i261 = call noalias i8* @malloc(i64 1584) #7
  %43 = bitcast %"class.Eigen::Tensor"* %outputp to i8**
  store i8* %call.i.i.i.i.i261, i8** %43, align 8, !tbaa !7
  %ref.tmp.sroa.8.72..sroa_cast.i.i.i = getelementptr inbounds %"class.Eigen::Tensor", %"class.Eigen::Tensor"* %inputp, i64 0, i32 0, i32 1
  %44 = bitcast %"struct.Eigen::DSizes"* %ref.tmp.sroa.8.72..sroa_cast.i.i.i to i8*
  %45 = bitcast %"struct.std::array"* %dims.i.i.i.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %45) #7
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 %45, i8* nonnull align 8 %44, i64 32, i1 false) #7
  %arrayidx.i.i.i.i.i.i.i = getelementptr inbounds %"struct.std::array", %"struct.std::array"* %dims.i.i.i.i, i64 0, i32 0, i64 0
  %46 = load i64, i64* %arrayidx.i.i.i.i.i.i.i, align 8, !tbaa !21
  %arrayidx.i.i.i.i.i.i.i.1 = getelementptr inbounds %"struct.std::array", %"struct.std::array"* %dims.i.i.i.i, i64 0, i32 0, i64 1
  %47 = load i64, i64* %arrayidx.i.i.i.i.i.i.i.1, align 8, !tbaa !21
  %mul.i.i.i.i.i.1 = mul nsw i64 %47, %46
  %arrayidx.i.i.i.i.i.i.i.2 = getelementptr inbounds %"struct.std::array", %"struct.std::array"* %dims.i.i.i.i, i64 0, i32 0, i64 2
  %48 = load i64, i64* %arrayidx.i.i.i.i.i.i.i.2, align 8, !tbaa !21
  %mul.i.i.i.i.i.2 = mul nsw i64 %48, %mul.i.i.i.i.i.1
  %arrayidx.i.i.i.i.i.i.i.3 = getelementptr inbounds %"struct.std::array", %"struct.std::array"* %dims.i.i.i.i, i64 0, i32 0, i64 3
  %49 = load i64, i64* %arrayidx.i.i.i.i.i.i.i.3, align 8, !tbaa !21
  %mul.i.i.i.i.i.3 = mul nsw i64 %49, %mul.i.i.i.i.i.2
  %m_data.i.i256 = getelementptr inbounds %"class.Eigen::Tensor.1", %"class.Eigen::Tensor.1"* %kernelp, i64 0, i32 0, i32 0
  %m_data.i.i262 = getelementptr inbounds %"class.Eigen::Tensor", %"class.Eigen::Tensor"* %outputp, i64 0, i32 0, i32 0
  %agg.tmp.sroa.0.sroa.0.0.agg.tmp.sroa.0.0..sroa_cast.sroa_idx.i.i.i.i.i.i.i.i = getelementptr inbounds %"struct.Eigen::DSizes", %"struct.Eigen::DSizes"* %ref.tmp.sroa.8.72..sroa_cast.i.i.i, i64 0, i32 0, i32 0, i64 0
  %cmp.i.i.i.i.i.i = icmp eq i64 %mul.i.i.i.i.i.3, 693
  br i1 %cmp.i.i.i.i.i.i, label %_ZN5Eigen6TensorIfLi4ELi0ElE6resizeERKNS_6DSizesIlLi4EEE.exit.i.i.i, label %if.then.i.i.i.i.i.i

if.then.i.i.i.i.i.i:                              ; preds = %_ZN5Eigen10TensorBaseINS_6TensorIfLi2ELi0ElEELi1EE9setRandomEv.exit
  call void @free(i8* %call.i.i.i.i.i251) #7
  %tobool.i.i.i.i.i.i = icmp eq i64 %mul.i.i.i.i.i.3, 0
  br i1 %tobool.i.i.i.i.i.i, label %if.else.i.i.i.i.i.i, label %_ZN5Eigen8internal28conditional_aligned_new_autoIfLb1EEEPT_m.exit.i.i.i.i.i.i

_ZN5Eigen8internal28conditional_aligned_new_autoIfLb1EEEPT_m.exit.i.i.i.i.i.i: ; preds = %if.then.i.i.i.i.i.i
  %mul.i.i.i.i.i.i.i264 = shl i64 %mul.i.i.i.i.i.3, 2
  %call.i.i.i.i.i.i.i.i.i = call noalias i8* @malloc(i64 %mul.i.i.i.i.i.i.i264) #7
  store i8* %call.i.i.i.i.i.i.i.i.i, i8** %36, align 8, !tbaa !7
  br label %_ZN5Eigen6TensorIfLi4ELi0ElE6resizeERKNS_6DSizesIlLi4EEE.exit.i.i.i

if.else.i.i.i.i.i.i:                              ; preds = %if.then.i.i.i.i.i.i
  %m_data.i.i.i.i.i.i = getelementptr inbounds %"class.Eigen::Tensor", %"class.Eigen::Tensor"* %inputp, i64 0, i32 0, i32 0
  store float* null, float** %m_data.i.i.i.i.i.i, align 8, !tbaa !7
  br label %_ZN5Eigen6TensorIfLi4ELi0ElE6resizeERKNS_6DSizesIlLi4EEE.exit.i.i.i

_ZN5Eigen6TensorIfLi4ELi0ElE6resizeERKNS_6DSizesIlLi4EEE.exit.i.i.i: ; preds = %if.else.i.i.i.i.i.i, %_ZN5Eigen8internal28conditional_aligned_new_autoIfLb1EEEPT_m.exit.i.i.i.i.i.i, %_ZN5Eigen10TensorBaseINS_6TensorIfLi2ELi0ElEELi1EE9setRandomEv.exit
  %50 = phi i8* [ %call.i.i.i.i.i.i.i.i.i, %_ZN5Eigen8internal28conditional_aligned_new_autoIfLb1EEEPT_m.exit.i.i.i.i.i.i ], [ null, %if.else.i.i.i.i.i.i ], [ %call.i.i.i.i.i251, %_ZN5Eigen10TensorBaseINS_6TensorIfLi2ELi0ElEELi1EE9setRandomEv.exit ]
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 %44, i8* nonnull align 8 %45, i64 32, i1 false) #7, !tbaa !2, !tbaa.struct !22
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %45) #7
  %evaluator.sroa.10.72.copyload.i.i.i.i = load i64, i64* %agg.tmp.sroa.0.sroa.0.0.agg.tmp.sroa.0.0..sroa_cast.sroa_idx.i.i.i.i.i.i.i.i, align 8, !tbaa !24
  %evaluator.sroa.12.72.copyload.i.i.i.i = load i64, i64* %ref.tmp.sroa.4.0..sroa_idx8.i.i.i248, align 8, !tbaa !24
  %evaluator.sroa.13.72.copyload.i.i.i.i = load i64, i64* %ref.tmp.sroa.5.0..sroa_idx10.i.i.i249, align 8, !tbaa !24
  %evaluator.sroa.14.72.copyload.i.i.i.i = load i64, i64* %ref.tmp.sroa.6.0..sroa_idx12.i.i.i250, align 8, !tbaa !24
  %mul.i.i.i.i.i.i.i.i.i.i265 = mul i64 %evaluator.sroa.12.72.copyload.i.i.i.i, %evaluator.sroa.10.72.copyload.i.i.i.i
  %mul.i.i.i.i.i.i.i.i.i = mul i64 %mul.i.i.i.i.i.i.i.i.i.i265, %evaluator.sroa.13.72.copyload.i.i.i.i
  %mul.i.i.i.i.i.i.i.i266 = mul i64 %mul.i.i.i.i.i.i.i.i.i, %evaluator.sroa.14.72.copyload.i.i.i.i
  %cmp29.i.i.i.i = icmp sgt i64 %mul.i.i.i.i.i.i.i.i266, 0
  br i1 %cmp29.i.i.i.i, label %for.body.i.i.i.preheader.i, label %_ZN5Eigen10TensorBaseINS_6TensorIfLi4ELi0ElEELi1EE7setZeroEv.exit

for.body.i.i.i.preheader.i:                       ; preds = %_ZN5Eigen6TensorIfLi4ELi0ElE6resizeERKNS_6DSizesIlLi4EEE.exit.i.i.i
  %51 = shl i64 %mul.i.i.i.i.i.i.i.i266, 2
  call void @llvm.memset.p0i8.i64(i8* align 4 %50, i8 0, i64 %51, i1 false) #7
  br label %_ZN5Eigen10TensorBaseINS_6TensorIfLi4ELi0ElEELi1EE7setZeroEv.exit

_ZN5Eigen10TensorBaseINS_6TensorIfLi4ELi0ElEELi1EE7setZeroEv.exit: ; preds = %for.body.i.i.i.preheader.i, %_ZN5Eigen6TensorIfLi4ELi0ElE6resizeERKNS_6DSizesIlLi4EEE.exit.i.i.i
  %ref.tmp.sroa.8.56..sroa_cast.i.i.i = getelementptr inbounds %"class.Eigen::Tensor.1", %"class.Eigen::Tensor.1"* %kernelp, i64 0, i32 0, i32 1
  %52 = bitcast %"struct.Eigen::DSizes.5"* %ref.tmp.sroa.8.56..sroa_cast.i.i.i to i8*
  %53 = bitcast %"struct.std::array.6"* %dims.i.i.i.i267 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %53) #7
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 %53, i8* nonnull align 8 %52, i64 16, i1 false) #7
  %arrayidx.i.i.i.i.i.i.i270 = getelementptr inbounds %"struct.std::array.6", %"struct.std::array.6"* %dims.i.i.i.i267, i64 0, i32 0, i64 0
  %54 = load i64, i64* %arrayidx.i.i.i.i.i.i.i270, align 8, !tbaa !21
  %arrayidx.i.i.i.i.i.i.i270.1 = getelementptr inbounds %"struct.std::array.6", %"struct.std::array.6"* %dims.i.i.i.i267, i64 0, i32 0, i64 1
  %55 = load i64, i64* %arrayidx.i.i.i.i.i.i.i270.1, align 8, !tbaa !21
  %mul.i.i.i.i.i271.1 = mul nsw i64 %55, %54
  %agg.tmp.sroa.0.0..sroa_idx.i.i.i.i.i.i.i.i = getelementptr inbounds %"struct.Eigen::DSizes.5", %"struct.Eigen::DSizes.5"* %ref.tmp.sroa.8.56..sroa_cast.i.i.i, i64 0, i32 0, i32 0, i64 0
  %agg.tmp.sroa.0.0.copyload.i.i.i.i.i.i.i.i = load i64, i64* %agg.tmp.sroa.0.0..sroa_idx.i.i.i.i.i.i.i.i, align 8, !tbaa !11
  %agg.tmp.sroa.2.0.copyload.i.i.i.i.i.i.i.i = load i64, i64* %ref.tmp.sroa.4.0..sroa_idx3.i.i.i254, align 8, !tbaa !11
  %mul.i.i.i.i.i.i.i.i.i.i277 = mul nsw i64 %agg.tmp.sroa.2.0.copyload.i.i.i.i.i.i.i.i, %agg.tmp.sroa.0.0.copyload.i.i.i.i.i.i.i.i
  %cmp.i.i.i.i.i.i278 = icmp eq i64 %mul.i.i.i.i.i.i.i.i.i.i277, %mul.i.i.i.i.i271.1
  %.pre.i.i3.i279 = load i8*, i8** %39, align 8, !tbaa !13
  br i1 %cmp.i.i.i.i.i.i278, label %_ZN5Eigen6TensorIfLi2ELi0ElE6resizeERKNS_6DSizesIlLi2EEE.exit.i.i.i, label %if.then.i.i.i.i.i.i282

if.then.i.i.i.i.i.i282:                           ; preds = %_ZN5Eigen10TensorBaseINS_6TensorIfLi4ELi0ElEELi1EE7setZeroEv.exit
  call void @free(i8* %.pre.i.i3.i279) #7
  %tobool.i.i.i.i.i.i281 = icmp eq i64 %mul.i.i.i.i.i271.1, 0
  br i1 %tobool.i.i.i.i.i.i281, label %if.else.i.i.i.i.i.i287, label %_ZN5Eigen8internal28conditional_aligned_new_autoIfLb1EEEPT_m.exit.i.i.i.i.i.i285

_ZN5Eigen8internal28conditional_aligned_new_autoIfLb1EEEPT_m.exit.i.i.i.i.i.i285: ; preds = %if.then.i.i.i.i.i.i282
  %mul.i.i.i.i.i.i.i283 = shl i64 %mul.i.i.i.i.i271.1, 2
  %call.i.i.i.i.i.i.i.i.i284 = call noalias i8* @malloc(i64 %mul.i.i.i.i.i.i.i283) #7
  store i8* %call.i.i.i.i.i.i.i.i.i284, i8** %39, align 8, !tbaa !13
  br label %_ZN5Eigen6TensorIfLi2ELi0ElE6resizeERKNS_6DSizesIlLi2EEE.exit.i.i.i

if.else.i.i.i.i.i.i287:                           ; preds = %if.then.i.i.i.i.i.i282
  store float* null, float** %m_data.i.i256, align 8, !tbaa !13
  br label %_ZN5Eigen6TensorIfLi2ELi0ElE6resizeERKNS_6DSizesIlLi2EEE.exit.i.i.i

_ZN5Eigen6TensorIfLi2ELi0ElE6resizeERKNS_6DSizesIlLi2EEE.exit.i.i.i: ; preds = %if.else.i.i.i.i.i.i287, %_ZN5Eigen8internal28conditional_aligned_new_autoIfLb1EEEPT_m.exit.i.i.i.i.i.i285, %_ZN5Eigen10TensorBaseINS_6TensorIfLi4ELi0ElEELi1EE7setZeroEv.exit
  %56 = phi i8* [ %call.i.i.i.i.i.i.i.i.i284, %_ZN5Eigen8internal28conditional_aligned_new_autoIfLb1EEEPT_m.exit.i.i.i.i.i.i285 ], [ null, %if.else.i.i.i.i.i.i287 ], [ %.pre.i.i3.i279, %_ZN5Eigen10TensorBaseINS_6TensorIfLi4ELi0ElEELi1EE7setZeroEv.exit ]
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 %52, i8* nonnull align 8 %53, i64 16, i1 false) #7, !tbaa !11, !tbaa.struct !30
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %53) #7
  %evaluator.sroa.10.56.copyload.i.i.i.i = load i64, i64* %agg.tmp.sroa.0.0..sroa_idx.i.i.i.i.i.i.i.i, align 8, !tbaa !32
  %evaluator.sroa.12.56.copyload.i.i.i.i = load i64, i64* %ref.tmp.sroa.4.0..sroa_idx3.i.i.i254, align 8, !tbaa !32
  %mul.i.i.i.i.i.i.i.i288 = mul i64 %evaluator.sroa.12.56.copyload.i.i.i.i, %evaluator.sroa.10.56.copyload.i.i.i.i
  %cmp25.i.i.i.i = icmp sgt i64 %mul.i.i.i.i.i.i.i.i288, 0
  br i1 %cmp25.i.i.i.i, label %for.body.i.i.i.preheader.i289, label %_ZN5Eigen10TensorBaseINS_6TensorIfLi2ELi0ElEELi1EE7setZeroEv.exit

for.body.i.i.i.preheader.i289:                    ; preds = %_ZN5Eigen6TensorIfLi2ELi0ElE6resizeERKNS_6DSizesIlLi2EEE.exit.i.i.i
  %57 = shl i64 %mul.i.i.i.i.i.i.i.i288, 2
  call void @llvm.memset.p0i8.i64(i8* align 4 %56, i8 0, i64 %57, i1 false) #7
  br label %_ZN5Eigen10TensorBaseINS_6TensorIfLi2ELi0ElEELi1EE7setZeroEv.exit

_ZN5Eigen10TensorBaseINS_6TensorIfLi2ELi0ElEELi1EE7setZeroEv.exit: ; preds = %for.body.i.i.i.preheader.i289, %_ZN5Eigen6TensorIfLi2ELi0ElE6resizeERKNS_6DSizesIlLi2EEE.exit.i.i.i
  %58 = bitcast %struct.timespec* %ts.i.i.i.i.i291 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %58) #7, !noalias !33
  %call.i.i.i.i.i292 = call i32 @clock_gettime(i32 0, %struct.timespec* nonnull %ts.i.i.i.i.i291) #7, !noalias !33
  %call1.i.i.i.i.i293 = call i64 @random() #7, !noalias !33
  %tv_nsec.i.i.i.i.i294 = getelementptr inbounds %struct.timespec, %struct.timespec* %ts.i.i.i.i.i291, i64 0, i32 1
  %59 = load i64, i64* %tv_nsec.i.i.i.i.i294, align 8, !tbaa !19, !noalias !33
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %58) #7, !noalias !33
  %ref.tmp.sroa.8.72..sroa_cast.i.i296 = getelementptr inbounds %"class.Eigen::Tensor", %"class.Eigen::Tensor"* %outputp, i64 0, i32 0, i32 1
  %60 = bitcast %"struct.Eigen::DSizes"* %ref.tmp.sroa.8.72..sroa_cast.i.i296 to i8*
  %61 = bitcast %"struct.std::array"* %dims.i.i.i290 to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %61) #7
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 %61, i8* nonnull align 8 %60, i64 32, i1 false) #7
  %arrayidx.i.i.i.i.i.i299 = getelementptr inbounds %"struct.std::array", %"struct.std::array"* %dims.i.i.i290, i64 0, i32 0, i64 0
  %62 = load i64, i64* %arrayidx.i.i.i.i.i.i299, align 8, !tbaa !21
  %arrayidx.i.i.i.i.i.i299.1 = getelementptr inbounds %"struct.std::array", %"struct.std::array"* %dims.i.i.i290, i64 0, i32 0, i64 1
  %63 = load i64, i64* %arrayidx.i.i.i.i.i.i299.1, align 8, !tbaa !21
  %mul.i.i.i3.i300.1 = mul nsw i64 %63, %62
  %arrayidx.i.i.i.i.i.i299.2 = getelementptr inbounds %"struct.std::array", %"struct.std::array"* %dims.i.i.i290, i64 0, i32 0, i64 2
  %64 = load i64, i64* %arrayidx.i.i.i.i.i.i299.2, align 8, !tbaa !21
  %mul.i.i.i3.i300.2 = mul nsw i64 %64, %mul.i.i.i3.i300.1
  %arrayidx.i.i.i.i.i.i299.3 = getelementptr inbounds %"struct.std::array", %"struct.std::array"* %dims.i.i.i290, i64 0, i32 0, i64 3
  %65 = load i64, i64* %arrayidx.i.i.i.i.i.i299.3, align 8, !tbaa !21
  %mul.i.i.i3.i300.3 = mul nsw i64 %65, %mul.i.i.i3.i300.2
  %xor.i.i.i.i.i304 = xor i64 %59, %call1.i.i.i.i.i293
  %agg.tmp.sroa.0.sroa.0.0.agg.tmp.sroa.0.0..sroa_cast.sroa_idx.i.i.i.i.i.i.i306 = getelementptr inbounds %"struct.Eigen::DSizes", %"struct.Eigen::DSizes"* %ref.tmp.sroa.8.72..sroa_cast.i.i296, i64 0, i32 0, i32 0, i64 0
  %agg.tmp.sroa.0.sroa.0.0.copyload.i.i.i.i.i.i.i307 = load i64, i64* %agg.tmp.sroa.0.sroa.0.0.agg.tmp.sroa.0.0..sroa_cast.sroa_idx.i.i.i.i.i.i.i306, align 8, !tbaa !2
  %agg.tmp.sroa.0.sroa.2.0.copyload.i.i.i.i.i.i.i309 = load i64, i64* %ref.tmp.sroa.4.0..sroa_idx8.i.i.i258, align 8, !tbaa !2
  %agg.tmp.sroa.0.sroa.3.0.copyload.i.i.i.i.i.i.i311 = load i64, i64* %ref.tmp.sroa.5.0..sroa_idx10.i.i.i259, align 8, !tbaa !2
  %agg.tmp.sroa.0.sroa.4.0.copyload.i.i.i.i.i.i.i313 = load i64, i64* %ref.tmp.sroa.6.0..sroa_idx12.i.i.i260, align 8, !tbaa !2
  %mul.i.i.i.i.i.i.i.i.i.i.i314 = mul nsw i64 %agg.tmp.sroa.0.sroa.2.0.copyload.i.i.i.i.i.i.i309, %agg.tmp.sroa.0.sroa.0.0.copyload.i.i.i.i.i.i.i307
  %mul.i.i.i.i.i.i.i.i.i.i315 = mul nsw i64 %mul.i.i.i.i.i.i.i.i.i.i.i314, %agg.tmp.sroa.0.sroa.3.0.copyload.i.i.i.i.i.i.i311
  %mul.i.i.i.i.i.i.i4.i.i316 = mul nsw i64 %mul.i.i.i.i.i.i.i.i.i.i315, %agg.tmp.sroa.0.sroa.4.0.copyload.i.i.i.i.i.i.i313
  %cmp.i.i.i.i.i317 = icmp eq i64 %mul.i.i.i.i.i.i.i4.i.i316, %mul.i.i.i3.i300.3
  br i1 %cmp.i.i.i.i.i317, label %for.end.i.i._ZN5Eigen6TensorIfLi4ELi0ElE6resizeERKNS_6DSizesIlLi4EEE.exit_crit_edge.i.i321, label %if.then.i.i.i.i.i323

for.end.i.i._ZN5Eigen6TensorIfLi4ELi0ElE6resizeERKNS_6DSizesIlLi4EEE.exit_crit_edge.i.i321: ; preds = %_ZN5Eigen10TensorBaseINS_6TensorIfLi2ELi0ElEELi1EE7setZeroEv.exit
  %.pre.i.i320 = load float*, float** %m_data.i.i262, align 8, !tbaa !7
  br label %_ZN5Eigen6TensorIfLi4ELi0ElE6resizeERKNS_6DSizesIlLi4EEE.exit.i.i337

if.then.i.i.i.i.i323:                             ; preds = %_ZN5Eigen10TensorBaseINS_6TensorIfLi2ELi0ElEELi1EE7setZeroEv.exit
  %66 = load i8*, i8** %43, align 8, !tbaa !7
  call void @free(i8* %66) #7
  %tobool.i.i.i.i.i322 = icmp eq i64 %mul.i.i.i3.i300.3, 0
  br i1 %tobool.i.i.i.i.i322, label %if.else.i.i.i.i.i328, label %_ZN5Eigen8internal28conditional_aligned_new_autoIfLb1EEEPT_m.exit.i.i.i.i.i326

_ZN5Eigen8internal28conditional_aligned_new_autoIfLb1EEEPT_m.exit.i.i.i.i.i326: ; preds = %if.then.i.i.i.i.i323
  %mul.i.i.i.i.i.i324 = shl i64 %mul.i.i.i3.i300.3, 2
  %call.i.i.i.i.i.i.i.i325 = call noalias i8* @malloc(i64 %mul.i.i.i.i.i.i324) #7
  store i8* %call.i.i.i.i.i.i.i.i325, i8** %43, align 8, !tbaa !7
  %67 = bitcast i8* %call.i.i.i.i.i.i.i.i325 to float*
  br label %_ZN5Eigen6TensorIfLi4ELi0ElE6resizeERKNS_6DSizesIlLi4EEE.exit.i.i337

if.else.i.i.i.i.i328:                             ; preds = %if.then.i.i.i.i.i323
  store float* null, float** %m_data.i.i262, align 8, !tbaa !7
  br label %_ZN5Eigen6TensorIfLi4ELi0ElE6resizeERKNS_6DSizesIlLi4EEE.exit.i.i337

_ZN5Eigen6TensorIfLi4ELi0ElE6resizeERKNS_6DSizesIlLi4EEE.exit.i.i337: ; preds = %if.else.i.i.i.i.i328, %_ZN5Eigen8internal28conditional_aligned_new_autoIfLb1EEEPT_m.exit.i.i.i.i.i326, %for.end.i.i._ZN5Eigen6TensorIfLi4ELi0ElE6resizeERKNS_6DSizesIlLi4EEE.exit_crit_edge.i.i321
  %68 = phi float* [ %.pre.i.i320, %for.end.i.i._ZN5Eigen6TensorIfLi4ELi0ElE6resizeERKNS_6DSizesIlLi4EEE.exit_crit_edge.i.i321 ], [ %67, %_ZN5Eigen8internal28conditional_aligned_new_autoIfLb1EEEPT_m.exit.i.i.i.i.i326 ], [ null, %if.else.i.i.i.i.i328 ]
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 %60, i8* nonnull align 8 %61, i64 32, i1 false) #7, !tbaa !2, !tbaa.struct !22
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %61) #7
  %evaluator.sroa.11.72.copyload.i.i.i329 = load i64, i64* %agg.tmp.sroa.0.sroa.0.0.agg.tmp.sroa.0.0..sroa_cast.sroa_idx.i.i.i.i.i.i.i306, align 8, !tbaa !24
  %evaluator.sroa.13.72.copyload.i.i.i330 = load i64, i64* %ref.tmp.sroa.4.0..sroa_idx8.i.i.i258, align 8, !tbaa !24
  %evaluator.sroa.14.72.copyload.i.i.i331 = load i64, i64* %ref.tmp.sroa.5.0..sroa_idx10.i.i.i259, align 8, !tbaa !24
  %evaluator.sroa.15.72.copyload.i.i.i332 = load i64, i64* %ref.tmp.sroa.6.0..sroa_idx12.i.i.i260, align 8, !tbaa !24
  %mul.i.i.i.i.i.i7.i.i.i333 = mul i64 %evaluator.sroa.13.72.copyload.i.i.i330, %evaluator.sroa.11.72.copyload.i.i.i329
  %mul.i.i.i.i.i.i.i.i334 = mul i64 %mul.i.i.i.i.i.i7.i.i.i333, %evaluator.sroa.14.72.copyload.i.i.i331
  %mul.i.i.i.i.i.i.i335 = mul i64 %mul.i.i.i.i.i.i.i.i334, %evaluator.sroa.15.72.copyload.i.i.i332
  %cmp28.i.i.i336 = icmp sgt i64 %mul.i.i.i.i.i.i.i335, 0
  br i1 %cmp28.i.i.i336, label %for.body.i.i.i355.preheader, label %_ZN5Eigen10TensorBaseINS_6TensorIfLi4ELi0ElEELi1EE9setRandomEv.exit356

for.body.i.i.i355.preheader:                      ; preds = %_ZN5Eigen6TensorIfLi4ELi0ElE6resizeERKNS_6DSizesIlLi4EEE.exit.i.i337
  br label %for.body.i.i.i355

for.body.i.i.i355:                                ; preds = %for.body.i.i.i355.preheader, %for.body.i.i.i355
  %i.030.i.i.i338 = phi i64 [ %inc.i.i.i353, %for.body.i.i.i355 ], [ 0, %for.body.i.i.i355.preheader ]
  %evaluator.sroa.7.029.i.i.in.in.i339 = phi i64 [ %add.i.i.i.i.i.i.i342, %for.body.i.i.i355 ], [ %xor.i.i.i.i.i304, %for.body.i.i.i355.preheader ]
  %evaluator.sroa.7.029.i.i.in.i340 = mul i64 %evaluator.sroa.7.029.i.i.in.in.i339, 6364136223846793005
  %evaluator.sroa.7.029.i.i.i341 = add nsw i64 %i.030.i.i.i338, -2720673578348880933
  %add.i.i.i.i.i.i.i342 = add i64 %evaluator.sroa.7.029.i.i.i341, %evaluator.sroa.7.029.i.i.in.i340
  %shr.i.i.i.i.i.i.i.i.i343 = lshr i64 %add.i.i.i.i.i.i.i342, 22
  %xor.i.i.i.i.i.i.i.i.i344 = xor i64 %shr.i.i.i.i.i.i.i.i.i343, %add.i.i.i.i.i.i.i342
  %shr1.i.i.i.i.i.i.i.i.i345 = lshr i64 %add.i.i.i.i.i.i.i342, 61
  %add2.i.i.i.i.i.i.i.i.i346 = add nuw nsw i64 %shr1.i.i.i.i.i.i.i.i.i345, 22
  %shr3.i.i.i.i.i.i.i.i.i347 = lshr i64 %xor.i.i.i.i.i.i.i.i.i344, %add2.i.i.i.i.i.i.i.i.i346
  %conv.i.i.i.i.i.i.i.i.i348 = trunc i64 %shr3.i.i.i.i.i.i.i.i.i347 to i32
  %and.i.i.i.i.i.i.i.i349 = and i32 %conv.i.i.i.i.i.i.i.i.i348, 8388607
  %or.i.i.i.i.i.i.i.i350 = or i32 %and.i.i.i.i.i.i.i.i349, 1065353216
  %69 = bitcast i32 %or.i.i.i.i.i.i.i.i350 to float
  %sub.i.i.i.i.i.i.i.i351 = fadd fast float %69, -1.000000e+00
  %arrayidx.i.i.i.i.i352 = getelementptr inbounds float, float* %68, i64 %i.030.i.i.i338
  store float %sub.i.i.i.i.i.i.i.i351, float* %arrayidx.i.i.i.i.i352, align 4, !tbaa !25
  %inc.i.i.i353 = add nuw nsw i64 %i.030.i.i.i338, 1
  %exitcond.i.i.i354 = icmp eq i64 %inc.i.i.i353, %mul.i.i.i.i.i.i.i335
  br i1 %exitcond.i.i.i354, label %_ZN5Eigen10TensorBaseINS_6TensorIfLi4ELi0ElEELi1EE9setRandomEv.exit356.loopexit, label %for.body.i.i.i355

_ZN5Eigen10TensorBaseINS_6TensorIfLi4ELi0ElEELi1EE9setRandomEv.exit356.loopexit: ; preds = %for.body.i.i.i355
  br label %_ZN5Eigen10TensorBaseINS_6TensorIfLi4ELi0ElEELi1EE9setRandomEv.exit356

_ZN5Eigen10TensorBaseINS_6TensorIfLi4ELi0ElEELi1EE9setRandomEv.exit356: ; preds = %_ZN5Eigen10TensorBaseINS_6TensorIfLi4ELi0ElEELi1EE9setRandomEv.exit356.loopexit, %_ZN5Eigen6TensorIfLi4ELi0ElE6resizeERKNS_6DSizesIlLi4EEE.exit.i.i337
  %call.i.i.i.i.i359 = call noalias i8* @malloc(i64 16) #7
  %70 = bitcast <2 x i64>* %dims.i.i.i.i361 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %70) #7
  store <2 x i64> <i64 2, i64 2>, <2 x i64>* %dims.i.i.i.i361, align 16
  %arrayidx.i.i.i.i.i.i.i365.phi.trans.insert = getelementptr inbounds %"struct.std::array.6", %"struct.std::array.6"* %tmpcast, i64 0, i32 0, i64 1
  %.pre = load i64, i64* %arrayidx.i.i.i.i.i.i.i365.phi.trans.insert, align 8, !tbaa !21
  %mul.i.i.i.i.i366 = shl i64 %.pre, 3
  %71 = bitcast i8* %call.i.i.i.i.i359 to float*
  %cmp.i.i.i.i.i.i377 = icmp eq i64 %.pre, 2
  br i1 %cmp.i.i.i.i.i.i377, label %_ZN5Eigen10TensorBaseINS_6TensorIfLi2ELi0ElEELi1EE7setZeroEv.exit393, label %if.then.i.i.i.i.i.i381

if.then.i.i.i.i.i.i381:                           ; preds = %_ZN5Eigen10TensorBaseINS_6TensorIfLi4ELi0ElEELi1EE9setRandomEv.exit356
  call void @free(i8* %call.i.i.i.i.i359) #7
  %tobool.i.i.i.i.i.i380 = icmp eq i64 %.pre, 0
  br i1 %tobool.i.i.i.i.i.i380, label %_ZN5Eigen10TensorBaseINS_6TensorIfLi2ELi0ElEELi1EE7setZeroEv.exit393, label %_ZN5Eigen8internal28conditional_aligned_new_autoIfLb1EEEPT_m.exit.i.i.i.i.i.i384

_ZN5Eigen8internal28conditional_aligned_new_autoIfLb1EEEPT_m.exit.i.i.i.i.i.i384: ; preds = %if.then.i.i.i.i.i.i381
  %call.i.i.i.i.i.i.i.i.i383 = call noalias i8* @malloc(i64 %mul.i.i.i.i.i366) #7
  %72 = bitcast i8* %call.i.i.i.i.i.i.i.i.i383 to float*
  br label %_ZN5Eigen10TensorBaseINS_6TensorIfLi2ELi0ElEELi1EE7setZeroEv.exit393

_ZN5Eigen10TensorBaseINS_6TensorIfLi2ELi0ElEELi1EE7setZeroEv.exit393: ; preds = %_ZN5Eigen8internal28conditional_aligned_new_autoIfLb1EEEPT_m.exit.i.i.i.i.i.i384, %if.then.i.i.i.i.i.i381, %_ZN5Eigen10TensorBaseINS_6TensorIfLi4ELi0ElEELi1EE9setRandomEv.exit356
  %73 = phi i8* [ %call.i.i.i.i.i359, %_ZN5Eigen10TensorBaseINS_6TensorIfLi4ELi0ElEELi1EE9setRandomEv.exit356 ], [ %call.i.i.i.i.i.i.i.i.i383, %_ZN5Eigen8internal28conditional_aligned_new_autoIfLb1EEEPT_m.exit.i.i.i.i.i.i384 ], [ null, %if.then.i.i.i.i.i.i381 ]
  %expected_kernel.sroa.0.0 = phi float* [ %71, %_ZN5Eigen10TensorBaseINS_6TensorIfLi4ELi0ElEELi1EE9setRandomEv.exit356 ], [ %72, %_ZN5Eigen8internal28conditional_aligned_new_autoIfLb1EEEPT_m.exit.i.i.i.i.i.i384 ], [ null, %if.then.i.i.i.i.i.i381 ]
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %70) #7
  call void @llvm.memset.p0i8.i64(i8* align 4 %73, i8 0, i64 16, i1 false) #7
  %74 = load float*, float** %m_data.i.i, align 8, !tbaa !7
  %75 = load i64, i64* %ref.tmp.sroa.0.0..sroa_idx.i.i.i, align 8, !tbaa !21
  %76 = load i64, i64* %ref.tmp.sroa.4.0..sroa_idx8.i.i.i, align 8, !tbaa !21
  %77 = load i64, i64* %ref.tmp.sroa.5.0..sroa_idx10.i.i.i, align 8, !tbaa !21
  %78 = load float*, float** %m_data.i.i262, align 8, !tbaa !7
  %79 = load i64, i64* %ref.tmp.sroa.0.0..sroa_idx.i.i.i257, align 8, !tbaa !21
  %80 = load i64, i64* %ref.tmp.sroa.4.0..sroa_idx8.i.i.i258, align 8, !tbaa !21
  %81 = load i64, i64* %ref.tmp.sroa.5.0..sroa_idx10.i.i.i259, align 8, !tbaa !21
  br label %for.cond6.preheader

for.cond6.preheader:                              ; preds = %for.cond.cleanup12.1, %_ZN5Eigen10TensorBaseINS_6TensorIfLi2ELi0ElEELi1EE7setZeroEv.exit393
  %indvars.iv569 = phi i64 [ 0, %_ZN5Eigen10TensorBaseINS_6TensorIfLi2ELi0ElEELi1EE7setZeroEv.exit393 ], [ %indvars.iv.next570, %for.cond.cleanup12.1 ]
  br label %for.cond14.preheader

for.cond.cleanup:                                 ; preds = %for.cond.cleanup12.1
  %arrayidx.i.i.i492.1.lcssa.lcssa.lcssa = phi float* [ %arrayidx.i.i.i492.1.lcssa.lcssa, %for.cond.cleanup12.1 ]
  %arrayidx.i.i.i492.122.lcssa.lcssa.lcssa = phi float* [ %arrayidx.i.i.i492.122.lcssa.lcssa, %for.cond.cleanup12.1 ]
  %arrayidx.i.i.i492.1.1.lcssa.lcssa.lcssa = phi float* [ %arrayidx.i.i.i492.1.1.lcssa.lcssa, %for.cond.cleanup12.1 ]
  call void @_ZL6matvecPKN5Eigen6TensorIfLi2ELi0ElEEPKNS0_IfLi4ELi0ElEEPS4_(%"class.Eigen::Tensor.1"* nonnull %kernel, %"class.Eigen::Tensor"* nonnull %input, %"class.Eigen::Tensor"* nonnull %output)
  %puts = call i32 @puts(i8* getelementptr inbounds ([13 x i8], [13 x i8]* @str, i64 0, i64 0))
  %call86 = call fast double @__enzyme_autodiff(i8* bitcast (void (%"class.Eigen::Tensor.1"*, %"class.Eigen::Tensor"*, %"class.Eigen::Tensor"*)* @_ZL6matvecPKN5Eigen6TensorIfLi2ELi0ElEEPKNS0_IfLi4ELi0ElEEPS4_ to i8*), %"class.Eigen::Tensor.1"* nonnull %kernel, %"class.Eigen::Tensor.1"* nonnull %kernelp, %"class.Eigen::Tensor"* nonnull %input, %"class.Eigen::Tensor"* nonnull %inputp, %"class.Eigen::Tensor"* nonnull %output, %"class.Eigen::Tensor"* nonnull %outputp) #7
  %.pre572.pre = load float*, float** %m_data.i.i256, align 8, !tbaa !13
  %82 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !36
  %83 = load float, float* %.pre572.pre, align 4, !tbaa !25
  %conv100 = fpext float %83 to double
  %84 = load float, float* %expected_kernel.sroa.0.0, align 4, !tbaa !25
  %conv104 = fpext float %84 to double
  %call105 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %82, i8* getelementptr inbounds ([60 x i8], [60 x i8]* @.str.1, i64 0, i64 0), i32 0, i32 0, double %conv100, i32 0, i32 0, double %conv104) #9
  %85 = load float*, float** %m_data.i.i256, align 8, !tbaa !13
  %86 = load float, float* %85, align 4, !tbaa !25
  %87 = load float, float* %expected_kernel.sroa.0.0, align 4, !tbaa !25
  %sub = fsub fast float %86, %87
  %88 = call fast float @llvm.fabs.f32(float %sub)
  %89 = fpext float %88 to double
  %cmp113 = fcmp fast ogt double %89, 1.000000e-04
  br i1 %cmp113, label %if.then, label %for.cond93

for.cond14.preheader:                             ; preds = %for.cond.cleanup16, %for.cond6.preheader
  %indvars.iv563 = phi i64 [ 0, %for.cond6.preheader ], [ %.lcssa1, %for.cond.cleanup16 ]
  br label %for.body17

for.cond.cleanup16:                               ; preds = %for.body17
  %.lcssa1 = phi i64 [ %93, %for.body17 ]
  %arrayidx.i.i.i492.1.lcssa = phi float* [ %arrayidx.i.i.i492.1, %for.body17 ]
  %arrayidx.i.i.i492.122.lcssa = phi float* [ %arrayidx.i.i.i492.122, %for.body17 ]
  %arrayidx.i.i.i492.1.1.lcssa = phi float* [ %arrayidx.i.i.i492.1.1, %for.body17 ]
  %exitcond565 = icmp eq i64 %.lcssa1, 6
  br i1 %exitcond565, label %for.cond14.preheader.1.preheader, label %for.cond14.preheader

for.cond14.preheader.1.preheader:                 ; preds = %for.cond.cleanup16
  %arrayidx.i.i.i492.1.lcssa.lcssa = phi float* [ %arrayidx.i.i.i492.1.lcssa, %for.cond.cleanup16 ]
  %arrayidx.i.i.i492.122.lcssa.lcssa = phi float* [ %arrayidx.i.i.i492.122.lcssa, %for.cond.cleanup16 ]
  %arrayidx.i.i.i492.1.1.lcssa.lcssa = phi float* [ %arrayidx.i.i.i492.1.1.lcssa, %for.cond.cleanup16 ]
  br label %for.cond14.preheader.1

for.body17:                                       ; preds = %for.body17, %for.cond14.preheader
  %indvars.iv560 = phi i64 [ 0, %for.cond14.preheader ], [ %indvars.iv.next561, %for.body17 ]
  %mul.i.i.i.i.i.i.i.i404 = mul nsw i64 %77, %indvars.iv560
  %mul.i.i.i.i.i.i.i.i468 = mul nsw i64 %81, %indvars.iv560
  %add.i.i.i.i.i.i.i.i469 = add nsw i64 %mul.i.i.i.i.i.i.i.i468, %indvars.iv563
  %mul.i.i.i.i.i.i.i470 = mul nsw i64 %add.i.i.i.i.i.i.i.i469, %80
  %mul.i.i.i.i.i.i472 = mul nsw i64 %mul.i.i.i.i.i.i.i470, %79
  %add.i.i.i.i.i.i473 = add nsw i64 %mul.i.i.i.i.i.i472, %indvars.iv569
  %arrayidx.i.i.i474 = getelementptr inbounds float, float* %78, i64 %add.i.i.i.i.i.i473
  %90 = load float, float* %arrayidx.i.i.i474, align 4, !tbaa !25
  %add.i.i.i.i.i.i.i.i482 = add nsw i64 %mul.i.i.i.i.i.i.i.i404, %indvars.iv563
  %mul.i.i.i.i.i.i.i483 = mul nsw i64 %add.i.i.i.i.i.i.i.i482, %76
  %mul.i.i.i.i.i.i485 = mul nsw i64 %mul.i.i.i.i.i.i.i483, %75
  %add.i.i.i.i.i.i486 = add nsw i64 %mul.i.i.i.i.i.i485, %indvars.iv569
  %arrayidx.i.i.i487 = getelementptr inbounds float, float* %74, i64 %add.i.i.i.i.i.i486
  %91 = load float, float* %arrayidx.i.i.i487, align 4, !tbaa !25
  %mul65 = fmul fast float %91, %90
  %92 = load float, float* %expected_kernel.sroa.0.0, align 4, !tbaa !25
  %add69 = fadd fast float %92, %mul65
  store float %add69, float* %expected_kernel.sroa.0.0, align 4, !tbaa !25
  %93 = add nuw nsw i64 %indvars.iv563, 1
  %add.i.i.i.i.i.i.i.i482.1 = add nsw i64 %mul.i.i.i.i.i.i.i.i404, %93
  %mul.i.i.i.i.i.i.i483.1 = mul nsw i64 %add.i.i.i.i.i.i.i.i482.1, %76
  %mul.i.i.i.i.i.i485.1 = mul nsw i64 %mul.i.i.i.i.i.i.i483.1, %75
  %add.i.i.i.i.i.i486.1 = add nsw i64 %mul.i.i.i.i.i.i485.1, %indvars.iv569
  %arrayidx.i.i.i487.1 = getelementptr inbounds float, float* %74, i64 %add.i.i.i.i.i.i486.1
  %94 = load float, float* %arrayidx.i.i.i487.1, align 4, !tbaa !25
  %mul65.1 = fmul fast float %94, %90
  %arrayidx.i.i.i492.1 = getelementptr inbounds float, float* %expected_kernel.sroa.0.0, i64 2
  %95 = load float, float* %arrayidx.i.i.i492.1, align 4, !tbaa !25
  %add69.1 = fadd fast float %95, %mul65.1
  store float %add69.1, float* %arrayidx.i.i.i492.1, align 4, !tbaa !25
  %add.i.i.i.i.i.i.i484.117 = add nsw i64 %mul.i.i.i.i.i.i.i483, 1
  %mul.i.i.i.i.i.i485.118 = mul nsw i64 %add.i.i.i.i.i.i.i484.117, %75
  %add.i.i.i.i.i.i486.119 = add nsw i64 %mul.i.i.i.i.i.i485.118, %indvars.iv569
  %arrayidx.i.i.i487.120 = getelementptr inbounds float, float* %74, i64 %add.i.i.i.i.i.i486.119
  %96 = load float, float* %arrayidx.i.i.i487.120, align 4, !tbaa !25
  %mul65.121 = fmul fast float %96, %90
  %arrayidx.i.i.i492.122 = getelementptr inbounds float, float* %expected_kernel.sroa.0.0, i64 1
  %97 = load float, float* %arrayidx.i.i.i492.122, align 4, !tbaa !25
  %add69.123 = fadd fast float %97, %mul65.121
  store float %add69.123, float* %arrayidx.i.i.i492.122, align 4, !tbaa !25
  %add.i.i.i.i.i.i.i484.1.1 = add nsw i64 %mul.i.i.i.i.i.i.i483.1, 1
  %mul.i.i.i.i.i.i485.1.1 = mul nsw i64 %add.i.i.i.i.i.i.i484.1.1, %75
  %add.i.i.i.i.i.i486.1.1 = add nsw i64 %mul.i.i.i.i.i.i485.1.1, %indvars.iv569
  %arrayidx.i.i.i487.1.1 = getelementptr inbounds float, float* %74, i64 %add.i.i.i.i.i.i486.1.1
  %98 = load float, float* %arrayidx.i.i.i487.1.1, align 4, !tbaa !25
  %mul65.1.1 = fmul fast float %98, %90
  %arrayidx.i.i.i492.1.1 = getelementptr inbounds float, float* %expected_kernel.sroa.0.0, i64 3
  %99 = load float, float* %arrayidx.i.i.i492.1.1, align 4, !tbaa !25
  %add69.1.1 = fadd fast float %99, %mul65.1.1
  store float %add69.1.1, float* %arrayidx.i.i.i492.1.1, align 4, !tbaa !25
  %indvars.iv.next561 = add nuw nsw i64 %indvars.iv560, 1
  %exitcond562 = icmp eq i64 %indvars.iv.next561, 11
  br i1 %exitcond562, label %for.cond.cleanup16, label %for.body17

for.cond93:                                       ; preds = %for.cond.cleanup
  %100 = load i64, i64* %ref.tmp.sroa.0.0..sroa_idx.i.i.i253, align 8, !tbaa !21
  %101 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !36
  %arrayidx.i.i.i497.1 = getelementptr inbounds float, float* %85, i64 %100
  %102 = load float, float* %arrayidx.i.i.i497.1, align 4, !tbaa !25
  %conv100.1 = fpext float %102 to double
  %103 = load float, float* %arrayidx.i.i.i492.1.lcssa.lcssa.lcssa, align 4, !tbaa !25
  %conv104.1 = fpext float %103 to double
  %call105.1 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %101, i8* getelementptr inbounds ([60 x i8], [60 x i8]* @.str.1, i64 0, i64 0), i32 0, i32 1, double %conv100.1, i32 0, i32 1, double %conv104.1) #9
  %104 = load float*, float** %m_data.i.i256, align 8, !tbaa !13
  %105 = load i64, i64* %ref.tmp.sroa.0.0..sroa_idx.i.i.i253, align 8, !tbaa !21
  %arrayidx.i.i.i517.1 = getelementptr inbounds float, float* %104, i64 %105
  %106 = load float, float* %arrayidx.i.i.i517.1, align 4, !tbaa !25
  %107 = load float, float* %arrayidx.i.i.i492.1.lcssa.lcssa.lcssa, align 4, !tbaa !25
  %sub.1 = fsub fast float %106, %107
  %108 = call fast float @llvm.fabs.f32(float %sub.1)
  %109 = fpext float %108 to double
  %cmp113.1 = fcmp fast ogt double %109, 1.000000e-04
  br i1 %cmp113.1, label %if.then, label %for.body96.1

if.then:                                          ; preds = %for.cond93.116, %for.body96.1, %for.cond93, %for.cond.cleanup
  %.lcssa2 = phi float [ %86, %for.cond.cleanup ], [ %106, %for.cond93 ], [ %115, %for.body96.1 ], [ %125, %for.cond93.116 ]
  %.lcssa = phi float [ %87, %for.cond.cleanup ], [ %107, %for.cond93 ], [ %116, %for.body96.1 ], [ %126, %for.cond93.116 ]
  %110 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !36
  %conv117 = fpext float %.lcssa2 to double
  %conv121 = fpext float %.lcssa to double
  %call122 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %110, i8* getelementptr inbounds ([68 x i8], [68 x i8]* @.str.2, i64 0, i64 0), i8* getelementptr inbounds ([16 x i8], [16 x i8]* @.str.3, i64 0, i64 0), double %conv117, i8* getelementptr inbounds ([24 x i8], [24 x i8]* @.str.4, i64 0, i64 0), double %conv121, double 1.000000e-04, i8* getelementptr inbounds ([61 x i8], [61 x i8]* @.str.5, i64 0, i64 0), i32 415, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @__PRETTY_FUNCTION__.main, i64 0, i64 0)) #9
  call void @abort() #10
  unreachable

for.body96.1:                                     ; preds = %for.cond93
  %111 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !36
  %arrayidx.i.i.i497.18 = getelementptr inbounds float, float* %104, i64 1
  %112 = load float, float* %arrayidx.i.i.i497.18, align 4, !tbaa !25
  %conv100.19 = fpext float %112 to double
  %113 = load float, float* %arrayidx.i.i.i492.122.lcssa.lcssa.lcssa, align 4, !tbaa !25
  %conv104.111 = fpext float %113 to double
  %call105.112 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %111, i8* getelementptr inbounds ([60 x i8], [60 x i8]* @.str.1, i64 0, i64 0), i32 1, i32 0, double %conv100.19, i32 1, i32 0, double %conv104.111) #9
  %114 = load float*, float** %m_data.i.i256, align 8, !tbaa !13
  %arrayidx.i.i.i517.113 = getelementptr inbounds float, float* %114, i64 1
  %115 = load float, float* %arrayidx.i.i.i517.113, align 4, !tbaa !25
  %116 = load float, float* %arrayidx.i.i.i492.122.lcssa.lcssa.lcssa, align 4, !tbaa !25
  %sub.114 = fsub fast float %115, %116
  %117 = call fast float @llvm.fabs.f32(float %sub.114)
  %118 = fpext float %117 to double
  %cmp113.115 = fcmp fast ogt double %118, 1.000000e-04
  br i1 %cmp113.115, label %if.then, label %for.cond93.116

for.cond93.116:                                   ; preds = %for.body96.1
  %119 = load i64, i64* %ref.tmp.sroa.0.0..sroa_idx.i.i.i253, align 8, !tbaa !21
  %120 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !36
  %add.i.i.i.i.i.i496.1.1 = add nsw i64 %119, 1
  %arrayidx.i.i.i497.1.1 = getelementptr inbounds float, float* %114, i64 %add.i.i.i.i.i.i496.1.1
  %121 = load float, float* %arrayidx.i.i.i497.1.1, align 4, !tbaa !25
  %conv100.1.1 = fpext float %121 to double
  %122 = load float, float* %arrayidx.i.i.i492.1.1.lcssa.lcssa.lcssa, align 4, !tbaa !25
  %conv104.1.1 = fpext float %122 to double
  %call105.1.1 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %120, i8* getelementptr inbounds ([60 x i8], [60 x i8]* @.str.1, i64 0, i64 0), i32 1, i32 1, double %conv100.1.1, i32 1, i32 1, double %conv104.1.1) #9
  %123 = load float*, float** %m_data.i.i256, align 8, !tbaa !13
  %124 = load i64, i64* %ref.tmp.sroa.0.0..sroa_idx.i.i.i253, align 8, !tbaa !21
  %add.i.i.i.i.i.i516.1.1 = add nsw i64 %124, 1
  %arrayidx.i.i.i517.1.1 = getelementptr inbounds float, float* %123, i64 %add.i.i.i.i.i.i516.1.1
  %125 = load float, float* %arrayidx.i.i.i517.1.1, align 4, !tbaa !25
  %126 = load float, float* %arrayidx.i.i.i492.1.1.lcssa.lcssa.lcssa, align 4, !tbaa !25
  %sub.1.1 = fsub fast float %125, %126
  %127 = call fast float @llvm.fabs.f32(float %sub.1.1)
  %128 = fpext float %127 to double
  %cmp113.1.1 = fcmp fast ogt double %128, 1.000000e-04
  br i1 %cmp113.1.1, label %if.then, label %for.cond93.1.1

for.cond93.1.1:                                   ; preds = %for.cond93.116
  call void @free(i8* %73) #7
  %129 = load i8*, i8** %43, align 8, !tbaa !7
  call void @free(i8* %129) #7
  call void @llvm.lifetime.end.p0i8(i64 40, i8* nonnull %40) #7
  %130 = load i8*, i8** %39, align 8, !tbaa !13
  call void @free(i8* %130) #7
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %37) #7
  %131 = load i8*, i8** %36, align 8, !tbaa !7
  call void @free(i8* %131) #7
  call void @llvm.lifetime.end.p0i8(i64 40, i8* nonnull %33) #7
  %132 = load i8*, i8** %10, align 8, !tbaa !7
  call void @free(i8* %132) #7
  call void @llvm.lifetime.end.p0i8(i64 40, i8* nonnull %7) #7
  %133 = load i8*, i8** %6, align 8, !tbaa !13
  call void @free(i8* %133) #7
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %4) #7
  %134 = load i8*, i8** %3, align 8, !tbaa !7
  call void @free(i8* %134) #7
  call void @llvm.lifetime.end.p0i8(i64 40, i8* nonnull %0) #7
  ret i32 0

for.cond14.preheader.1:                           ; preds = %for.cond14.preheader.1.preheader, %for.cond.cleanup16.1
  %indvars.iv563.1 = phi i64 [ %.lcssa3, %for.cond.cleanup16.1 ], [ 0, %for.cond14.preheader.1.preheader ]
  br label %for.body17.1

for.body17.1:                                     ; preds = %for.body17.1, %for.cond14.preheader.1
  %indvars.iv560.1 = phi i64 [ 0, %for.cond14.preheader.1 ], [ %indvars.iv.next561.1, %for.body17.1 ]
  %mul.i.i.i.i.i.i.i.i404.1 = mul nsw i64 %77, %indvars.iv560.1
  %mul.i.i.i.i.i.i.i.i468.1 = mul nsw i64 %81, %indvars.iv560.1
  %add.i.i.i.i.i.i.i.i469.1 = add nsw i64 %mul.i.i.i.i.i.i.i.i468.1, %indvars.iv563.1
  %mul.i.i.i.i.i.i.i470.1 = mul nsw i64 %add.i.i.i.i.i.i.i.i469.1, %80
  %add.i.i.i.i.i.i.i471.1 = add nsw i64 %mul.i.i.i.i.i.i.i470.1, 1
  %mul.i.i.i.i.i.i472.1 = mul nsw i64 %add.i.i.i.i.i.i.i471.1, %79
  %add.i.i.i.i.i.i473.1 = add nsw i64 %mul.i.i.i.i.i.i472.1, %indvars.iv569
  %arrayidx.i.i.i474.1 = getelementptr inbounds float, float* %78, i64 %add.i.i.i.i.i.i473.1
  %135 = load float, float* %arrayidx.i.i.i474.1, align 4, !tbaa !25
  %add.i.i.i.i.i.i.i.i482.124 = add nsw i64 %mul.i.i.i.i.i.i.i.i404.1, %indvars.iv563.1
  %mul.i.i.i.i.i.i.i483.125 = mul nsw i64 %add.i.i.i.i.i.i.i.i482.124, %76
  %add.i.i.i.i.i.i.i484.126 = add nsw i64 %mul.i.i.i.i.i.i.i483.125, 1
  %mul.i.i.i.i.i.i485.127 = mul nsw i64 %add.i.i.i.i.i.i.i484.126, %75
  %add.i.i.i.i.i.i486.128 = add nsw i64 %mul.i.i.i.i.i.i485.127, %indvars.iv569
  %arrayidx.i.i.i487.129 = getelementptr inbounds float, float* %74, i64 %add.i.i.i.i.i.i486.128
  %136 = load float, float* %arrayidx.i.i.i487.129, align 4, !tbaa !25
  %mul65.130 = fmul fast float %136, %135
  %137 = load float, float* %expected_kernel.sroa.0.0, align 4, !tbaa !25
  %add69.131 = fadd fast float %137, %mul65.130
  store float %add69.131, float* %expected_kernel.sroa.0.0, align 4, !tbaa !25
  %138 = add nuw nsw i64 %indvars.iv563.1, 1
  %add.i.i.i.i.i.i.i.i482.1.132 = add nsw i64 %mul.i.i.i.i.i.i.i.i404.1, %138
  %mul.i.i.i.i.i.i.i483.1.133 = mul nsw i64 %add.i.i.i.i.i.i.i.i482.1.132, %76
  %add.i.i.i.i.i.i.i484.1.134 = add nsw i64 %mul.i.i.i.i.i.i.i483.1.133, 1
  %mul.i.i.i.i.i.i485.1.135 = mul nsw i64 %add.i.i.i.i.i.i.i484.1.134, %75
  %add.i.i.i.i.i.i486.1.136 = add nsw i64 %mul.i.i.i.i.i.i485.1.135, %indvars.iv569
  %arrayidx.i.i.i487.1.137 = getelementptr inbounds float, float* %74, i64 %add.i.i.i.i.i.i486.1.136
  %139 = load float, float* %arrayidx.i.i.i487.1.137, align 4, !tbaa !25
  %mul65.1.138 = fmul fast float %139, %135
  %140 = load float, float* %arrayidx.i.i.i492.1.lcssa.lcssa, align 4, !tbaa !25
  %add69.1.140 = fadd fast float %140, %mul65.1.138
  store float %add69.1.140, float* %arrayidx.i.i.i492.1.lcssa.lcssa, align 4, !tbaa !25
  %add.i.i.i.i.i.i.i484.117.1 = add nsw i64 %mul.i.i.i.i.i.i.i483.125, 2
  %mul.i.i.i.i.i.i485.118.1 = mul nsw i64 %add.i.i.i.i.i.i.i484.117.1, %75
  %add.i.i.i.i.i.i486.119.1 = add nsw i64 %mul.i.i.i.i.i.i485.118.1, %indvars.iv569
  %arrayidx.i.i.i487.120.1 = getelementptr inbounds float, float* %74, i64 %add.i.i.i.i.i.i486.119.1
  %141 = load float, float* %arrayidx.i.i.i487.120.1, align 4, !tbaa !25
  %mul65.121.1 = fmul fast float %141, %135
  %142 = load float, float* %arrayidx.i.i.i492.122.lcssa.lcssa, align 4, !tbaa !25
  %add69.123.1 = fadd fast float %142, %mul65.121.1
  store float %add69.123.1, float* %arrayidx.i.i.i492.122.lcssa.lcssa, align 4, !tbaa !25
  %add.i.i.i.i.i.i.i484.1.1.1 = add nsw i64 %mul.i.i.i.i.i.i.i483.1.133, 2
  %mul.i.i.i.i.i.i485.1.1.1 = mul nsw i64 %add.i.i.i.i.i.i.i484.1.1.1, %75
  %add.i.i.i.i.i.i486.1.1.1 = add nsw i64 %mul.i.i.i.i.i.i485.1.1.1, %indvars.iv569
  %arrayidx.i.i.i487.1.1.1 = getelementptr inbounds float, float* %74, i64 %add.i.i.i.i.i.i486.1.1.1
  %143 = load float, float* %arrayidx.i.i.i487.1.1.1, align 4, !tbaa !25
  %mul65.1.1.1 = fmul fast float %143, %135
  %144 = load float, float* %arrayidx.i.i.i492.1.1.lcssa.lcssa, align 4, !tbaa !25
  %add69.1.1.1 = fadd fast float %144, %mul65.1.1.1
  store float %add69.1.1.1, float* %arrayidx.i.i.i492.1.1.lcssa.lcssa, align 4, !tbaa !25
  %indvars.iv.next561.1 = add nuw nsw i64 %indvars.iv560.1, 1
  %exitcond562.1 = icmp eq i64 %indvars.iv.next561.1, 11
  br i1 %exitcond562.1, label %for.cond.cleanup16.1, label %for.body17.1

for.cond.cleanup16.1:                             ; preds = %for.body17.1
  %.lcssa3 = phi i64 [ %138, %for.body17.1 ]
  %exitcond565.1 = icmp eq i64 %.lcssa3, 6
  br i1 %exitcond565.1, label %for.cond.cleanup12.1, label %for.cond14.preheader.1

for.cond.cleanup12.1:                             ; preds = %for.cond.cleanup16.1
  %indvars.iv.next570 = add nuw nsw i64 %indvars.iv569, 1
  %exitcond571 = icmp eq i64 %indvars.iv.next570, 3
  br i1 %exitcond571, label %for.cond.cleanup, label %for.cond6.preheader
}

; Function Attrs: noinline nounwind uwtable
define internal void @_ZL6matvecPKN5Eigen6TensorIfLi2ELi0ElEEPKNS0_IfLi4ELi0ElEEPS4_(%"class.Eigen::Tensor.1"* noalias %K, %"class.Eigen::Tensor"* noalias %In, %"class.Eigen::Tensor"* %Out) #2 {
entry:
  %result.i.i.i = alloca float, align 4
  %left_evaluator.i = alloca %"struct.Eigen::TensorEvaluator.9", align 8
  %right_evaluator.i = alloca %"struct.Eigen::TensorEvaluator.10", align 8
  %dims.i.i = alloca %"struct.std::array", align 8
  %right_ref.tmp.i = alloca %"struct.Eigen::TensorEvaluator.10", align 8
  %0 = bitcast %"class.Eigen::Tensor"* %Out to i64*
  %m_dimensions.i.i.i.i.i = getelementptr inbounds %"class.Eigen::Tensor", %"class.Eigen::Tensor"* %Out, i64 0, i32 0, i32 1
  %1 = bitcast %"struct.Eigen::DSizes"* %m_dimensions.i.i.i.i.i to i8*
  %m_data.i.i = getelementptr inbounds %"struct.Eigen::TensorEvaluator.10", %"struct.Eigen::TensorEvaluator.10"* %right_ref.tmp.i, i64 0, i32 7, i32 0, i32 0
  %m_data5.i.i = getelementptr inbounds %"class.Eigen::Tensor.1", %"class.Eigen::Tensor.1"* %K, i64 0, i32 0, i32 0
  %a2 = load float*, float** %m_data5.i.i, align 8, !tbaa !13
  call void @subfn(float** nonnull %m_data.i.i, float* %a2) #7
  %2 = bitcast %"struct.std::array"* %dims.i.i to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 %1, i8* nonnull align 8 %2, i64 32, i1 false) #7, !tbaa !2, !tbaa.struct !22
  %m_kernelArg.i.i.i = getelementptr inbounds %"struct.Eigen::TensorEvaluator.10", %"struct.Eigen::TensorEvaluator.10"* %right_ref.tmp.i, i64 0, i32 7
  %3 = bitcast %"class.Eigen::Tensor.1"* %m_kernelArg.i.i.i to i8**
  %4 = load i8*, i8** %3, align 8, !tbaa !13
  call void @free(i8* %4) #7
  %5 = load i64, i64* %0, align 8, !tbaa !7
  %6 = bitcast %"struct.Eigen::TensorEvaluator.9"* %left_evaluator.i to i64*
  store i64 %5, i64* %6, align 8, !tbaa !37
  %m_dims.i.i.i1 = getelementptr inbounds %"struct.Eigen::TensorEvaluator.9", %"struct.Eigen::TensorEvaluator.9"* %left_evaluator.i, i64 0, i32 1
  %7 = bitcast %"struct.Eigen::DSizes"* %m_dims.i.i.i1 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 %7, i8* nonnull align 8 %1, i64 32, i1 false) #7, !tbaa !24
  %m_impl.i.i.i4 = getelementptr inbounds %"struct.Eigen::TensorEvaluator.9", %"struct.Eigen::TensorEvaluator.9"* %left_evaluator.i, i64 0, i32 3
  store %"class.Eigen::Tensor"* %Out, %"class.Eigen::Tensor"** %m_impl.i.i.i4, align 8, !tbaa !36
  %m_inputImpl.i = getelementptr inbounds %"struct.Eigen::TensorEvaluator.10", %"struct.Eigen::TensorEvaluator.10"* %right_evaluator.i, i64 0, i32 4
  %8 = bitcast %"class.Eigen::Tensor"* %In to i64*
  %9 = load i64, i64* %8, align 8, !tbaa !7
  %10 = bitcast %"struct.Eigen::TensorEvaluator.11"* %m_inputImpl.i to i64*
  store i64 %9, i64* %10, align 8, !tbaa !39
  %m_dims.i132.i = getelementptr inbounds %"struct.Eigen::TensorEvaluator.10", %"struct.Eigen::TensorEvaluator.10"* %right_evaluator.i, i64 0, i32 4, i32 1
  %m_dimensions.i.i.i.i = getelementptr inbounds %"class.Eigen::Tensor", %"class.Eigen::Tensor"* %In, i64 0, i32 0, i32 1
  %11 = bitcast %"struct.Eigen::DSizes"* %m_dims.i132.i to i8*
  %12 = bitcast %"struct.Eigen::DSizes"* %m_dimensions.i.i.i.i to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 %11, i8* nonnull align 8 %12, i64 32, i1 false) #7, !tbaa !24
  %m_impl.i.i = getelementptr inbounds %"struct.Eigen::TensorEvaluator.10", %"struct.Eigen::TensorEvaluator.10"* %right_evaluator.i, i64 0, i32 4, i32 3
  store %"class.Eigen::Tensor"* %In, %"class.Eigen::Tensor"** %m_impl.i.i, align 8, !tbaa !36
  %m_kernelImpl.i = getelementptr inbounds %"struct.Eigen::TensorEvaluator.10", %"struct.Eigen::TensorEvaluator.10"* %right_evaluator.i, i64 0, i32 5
  %13 = bitcast %"class.Eigen::Tensor.1"* %K to i64*
  %14 = load i64, i64* %13, align 8, !tbaa !13
  %15 = bitcast %"struct.Eigen::TensorEvaluator.12"* %m_kernelImpl.i to i64*
  store i64 %14, i64* %15, align 8, !tbaa !41
  %m_dims.i133.i = getelementptr inbounds %"struct.Eigen::TensorEvaluator.10", %"struct.Eigen::TensorEvaluator.10"* %right_evaluator.i, i64 0, i32 5, i32 1
  %m_dimensions.i.i.i134.i = getelementptr inbounds %"class.Eigen::Tensor.1", %"class.Eigen::Tensor.1"* %K, i64 0, i32 0, i32 1
  %16 = bitcast %"struct.Eigen::DSizes.5"* %m_dims.i133.i to i8*
  %17 = bitcast %"struct.Eigen::DSizes.5"* %m_dimensions.i.i.i134.i to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 %16, i8* nonnull align 8 %17, i64 16, i1 false) #7, !tbaa !32
  %m_impl.i136.i = getelementptr inbounds %"struct.Eigen::TensorEvaluator.10", %"struct.Eigen::TensorEvaluator.10"* %right_evaluator.i, i64 0, i32 5, i32 3
  store %"class.Eigen::Tensor.1"* %K, %"class.Eigen::Tensor.1"** %m_impl.i136.i, align 8, !tbaa !36
  %m_dimensions.i = getelementptr inbounds %"struct.Eigen::TensorEvaluator.10", %"struct.Eigen::TensorEvaluator.10"* %right_evaluator.i, i64 0, i32 6
  %this6.i.i = bitcast %"struct.Eigen::DSizes"* %m_dimensions.i to i8*
  %m_data.i.i.i1 = getelementptr inbounds %"struct.Eigen::TensorEvaluator.10", %"struct.Eigen::TensorEvaluator.10"* %right_evaluator.i, i64 0, i32 7, i32 0, i32 0
  %agg.tmp.sroa.0.0..sroa_idx.i.i.i.i.i = getelementptr inbounds %"struct.Eigen::DSizes.5", %"struct.Eigen::DSizes.5"* %m_dimensions.i.i.i134.i, i64 0, i32 0, i32 0, i64 0
  %agg.tmp.sroa.0.0.copyload.i.i.i.i.i = load i64, i64* %agg.tmp.sroa.0.0..sroa_idx.i.i.i.i.i, align 8, !tbaa !11
  %agg.tmp.sroa.2.0..sroa_idx1.i.i.i.i.i = getelementptr inbounds %"class.Eigen::Tensor.1", %"class.Eigen::Tensor.1"* %K, i64 0, i32 0, i32 1, i32 0, i32 0, i64 1
  %agg.tmp.sroa.2.0.copyload.i.i.i.i.i = load i64, i64* %agg.tmp.sroa.2.0..sroa_idx1.i.i.i.i.i, align 8, !tbaa !11
  %mul.i.i.i.i.i.i.i = mul nsw i64 %agg.tmp.sroa.2.0.copyload.i.i.i.i.i, %agg.tmp.sroa.0.0.copyload.i.i.i.i.i
  %mul.i.i.i.i = shl i64 %mul.i.i.i.i.i.i.i, 2
  %call.i.i.i.i.i.i = call noalias i8* @malloc(i64 %mul.i.i.i.i) #7
  %18 = bitcast i8* %call.i.i.i.i.i.i to float*
  %19 = bitcast float** %m_data.i.i.i1 to i8**
  store i8* %call.i.i.i.i.i.i, i8** %19, align 8, !tbaa !13
  %20 = load float*, float** %m_data5.i.i, align 8, !tbaa !13
  br label %for.body.i.i.i.i.i

for.body.i.i.i.i.i:                               ; preds = %for.body.i.i.i.i.i, %entry
  %add.ptr11.i.i.i.i.i = phi float* [ %add.ptr.i.i.i.i.i, %for.body.i.i.i.i.i ], [ %20, %entry ]
  %idx.ext10.i.i.i.i.i = phi i64 [ %idx.ext.i.i.i.i.i, %for.body.i.i.i.i.i ], [ 0, %entry ]
  %i.09.i.i.i.i.i = phi i32 [ %inc.i.i.i.i.i, %for.body.i.i.i.i.i ], [ 0, %entry ]
  %21 = bitcast float* %add.ptr11.i.i.i.i.i to i32*
  %22 = load i32, i32* %21, align 4, !tbaa !25
  %arrayidx2.i.i.i.i.i = getelementptr inbounds float, float* %18, i64 %idx.ext10.i.i.i.i.i
  %23 = bitcast float* %arrayidx2.i.i.i.i.i to i32*
  store i32 %22, i32* %23, align 4, !tbaa !25
  %inc.i.i.i.i.i = add i32 %i.09.i.i.i.i.i, 1
  %idx.ext.i.i.i.i.i = zext i32 %inc.i.i.i.i.i to i64
  %add.ptr.i.i.i.i.i = getelementptr inbounds float, float* %20, i64 %idx.ext.i.i.i.i.i
  %cmp.i.i.i.i.i = icmp eq i64 %mul.i.i.i.i.i.i.i, %idx.ext.i.i.i.i.i
  br i1 %cmp.i.i.i.i.i, label %inl_ZN5Eigen15TensorEvaluatorIKNS_19TensorConvolutionOpIKSt5arrayIlLm2EEKNS_6TensorIfLi4ELi0ElEEKNS5_IfLi2ELi0ElEEEENS_13DefaultDeviceEEC2ERSB_RKSC_.exit, label %for.body.i.i.i.i.i

inl_ZN5Eigen15TensorEvaluatorIKNS_19TensorConvolutionOpIKSt5arrayIlLm2EEKNS_6TensorIfLi4ELi0ElEEKNS5_IfLi2ELi0ElEEEENS_13DefaultDeviceEEC2ERSB_RKSC_.exit: ; preds = %for.body.i.i.i.i.i
  %arrayidx.i.i129.i = getelementptr inbounds %"struct.Eigen::TensorEvaluator.10", %"struct.Eigen::TensorEvaluator.10"* %right_evaluator.i, i64 0, i32 0, i32 0, i64 0
  store i64 1, i64* %arrayidx.i.i129.i, align 8, !tbaa !21
  %arrayidx.i.i127.i = getelementptr inbounds %"struct.Eigen::TensorEvaluator.10", %"struct.Eigen::TensorEvaluator.10"* %right_evaluator.i, i64 0, i32 4, i32 1, i32 0, i32 0, i64 0
  %24 = load i64, i64* %arrayidx.i.i127.i, align 8, !tbaa !21
  %arrayidx.i.i126.i = getelementptr inbounds %"struct.Eigen::TensorEvaluator.10", %"struct.Eigen::TensorEvaluator.10"* %right_evaluator.i, i64 0, i32 0, i32 0, i64 1
  store i64 %24, i64* %arrayidx.i.i126.i, align 8, !tbaa !21
  %arrayidx.i.i127.1.i = getelementptr inbounds %"struct.Eigen::TensorEvaluator.10", %"struct.Eigen::TensorEvaluator.10"* %right_evaluator.i, i64 0, i32 4, i32 1, i32 0, i32 0, i64 1
  %25 = load i64, i64* %arrayidx.i.i127.1.i, align 8, !tbaa !21
  %mul.1.i = mul nsw i64 %25, %24
  %arrayidx.i.i126.1.i = getelementptr inbounds %"struct.Eigen::TensorEvaluator.10", %"struct.Eigen::TensorEvaluator.10"* %right_evaluator.i, i64 0, i32 0, i32 0, i64 2
  store i64 %mul.1.i, i64* %arrayidx.i.i126.1.i, align 8, !tbaa !21
  %arrayidx.i.i127.2.i = getelementptr inbounds %"struct.Eigen::TensorEvaluator.10", %"struct.Eigen::TensorEvaluator.10"* %right_evaluator.i, i64 0, i32 4, i32 1, i32 0, i32 0, i64 2
  %26 = load i64, i64* %arrayidx.i.i127.2.i, align 8, !tbaa !21
  %mul.2.i = mul nsw i64 %26, %mul.1.i
  %arrayidx.i.i126.2.i = getelementptr inbounds %"struct.Eigen::TensorEvaluator.10", %"struct.Eigen::TensorEvaluator.10"* %right_evaluator.i, i64 0, i32 0, i32 0, i64 3
  store i64 %mul.2.i, i64* %arrayidx.i.i126.2.i, align 8, !tbaa !21
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 %this6.i.i, i8* nonnull align 8 %11, i64 32, i1 false) #7, !tbaa !43
  %arrayidx.i.i117.i = getelementptr inbounds %"struct.Eigen::TensorEvaluator.10", %"struct.Eigen::TensorEvaluator.10"* %right_evaluator.i, i64 0, i32 3, i32 0, i64 0
  %arrayidx.i.i122157.i = getelementptr inbounds %"struct.Eigen::TensorEvaluator.10", %"struct.Eigen::TensorEvaluator.10"* %right_evaluator.i, i64 0, i32 5, i32 1, i32 0, i32 0, i64 0
  %27 = load i64, i64* %arrayidx.i.i122157.i, align 8, !tbaa !21
  %sub32158.i = add i64 %25, 1
  %add159.i = sub i64 %sub32158.i, %27
  %arrayidx.i.i121160.i = getelementptr inbounds %"struct.Eigen::TensorEvaluator.10", %"struct.Eigen::TensorEvaluator.10"* %right_evaluator.i, i64 0, i32 6, i32 0, i32 0, i64 1
  store i64 %add159.i, i64* %arrayidx.i.i121160.i, align 8, !tbaa !21
  store i64 1, i64* %arrayidx.i.i117.i, align 8, !tbaa !21
  %arrayidx.i.i115.i = getelementptr inbounds %"struct.Eigen::TensorEvaluator.10", %"struct.Eigen::TensorEvaluator.10"* %right_evaluator.i, i64 0, i32 2, i32 0, i64 0
  store i64 %24, i64* %arrayidx.i.i115.i, align 8, !tbaa !21
  %arrayidx.i.i122.i = getelementptr inbounds %"struct.Eigen::TensorEvaluator.10", %"struct.Eigen::TensorEvaluator.10"* %right_evaluator.i, i64 0, i32 5, i32 1, i32 0, i32 0, i64 1
  %28 = load i64, i64* %arrayidx.i.i122.i, align 8, !tbaa !21
  %sub32.i = add i64 %26, 1
  %add.i = sub i64 %sub32.i, %28
  %arrayidx.i.i121.i = getelementptr inbounds %"struct.Eigen::TensorEvaluator.10", %"struct.Eigen::TensorEvaluator.10"* %right_evaluator.i, i64 0, i32 6, i32 0, i32 0, i64 2
  store i64 %add.i, i64* %arrayidx.i.i121.i, align 8, !tbaa !21
  %arrayidx.i.i118.i = getelementptr inbounds %"struct.Eigen::TensorEvaluator.10", %"struct.Eigen::TensorEvaluator.10"* %right_evaluator.i, i64 0, i32 3, i32 0, i64 1
  store i64 %27, i64* %arrayidx.i.i118.i, align 8, !tbaa !21
  %arrayidx.i.i115152.i = getelementptr inbounds %"struct.Eigen::TensorEvaluator.10", %"struct.Eigen::TensorEvaluator.10"* %right_evaluator.i, i64 0, i32 2, i32 0, i64 1
  store i64 %mul.1.i, i64* %arrayidx.i.i115152.i, align 8, !tbaa !21
  %arrayidx.i.i125.i = getelementptr inbounds %"struct.Eigen::TensorEvaluator.10", %"struct.Eigen::TensorEvaluator.10"* %right_evaluator.i, i64 0, i32 1, i32 0, i64 0
  store i64 1, i64* %arrayidx.i.i125.i, align 8, !tbaa !21
  %arrayidx.i.i113.i = getelementptr inbounds %"struct.Eigen::TensorEvaluator.10", %"struct.Eigen::TensorEvaluator.10"* %right_evaluator.i, i64 0, i32 6, i32 0, i32 0, i64 0
  %29 = load i64, i64* %arrayidx.i.i113.i, align 8, !tbaa !21
  %arrayidx.i.i.i2 = getelementptr inbounds %"struct.Eigen::TensorEvaluator.10", %"struct.Eigen::TensorEvaluator.10"* %right_evaluator.i, i64 0, i32 1, i32 0, i64 1
  store i64 %29, i64* %arrayidx.i.i.i2, align 8, !tbaa !21
  %mul72.1.i = mul i64 %add159.i, %29
  %arrayidx.i.i.1.i = getelementptr inbounds %"struct.Eigen::TensorEvaluator.10", %"struct.Eigen::TensorEvaluator.10"* %right_evaluator.i, i64 0, i32 1, i32 0, i64 2
  store i64 %mul72.1.i, i64* %arrayidx.i.i.1.i, align 8, !tbaa !21
  %mul72.2.i = mul i64 %add.i, %mul72.1.i
  %arrayidx.i.i.2.i = getelementptr inbounds %"struct.Eigen::TensorEvaluator.10", %"struct.Eigen::TensorEvaluator.10"* %right_evaluator.i, i64 0, i32 1, i32 0, i64 3
  store i64 %mul72.2.i, i64* %arrayidx.i.i.2.i, align 8, !tbaa !21
  %m_data.i.i.i.i.i = getelementptr inbounds %"struct.Eigen::TensorEvaluator.10", %"struct.Eigen::TensorEvaluator.10"* %right_evaluator.i, i64 0, i32 5, i32 0
  %30 = bitcast float** %m_data.i.i.i.i.i to i64*
  %31 = load i64, i64* %30, align 8, !tbaa !41
  %m_kernel.i.i.i.i = getelementptr inbounds %"struct.Eigen::TensorEvaluator.10", %"struct.Eigen::TensorEvaluator.10"* %right_evaluator.i, i64 0, i32 8
  %32 = bitcast float** %m_kernel.i.i.i.i to i64*
  store i64 %31, i64* %32, align 8, !tbaa !44
  %m_local_kernel8.i.i.i.i = getelementptr inbounds %"struct.Eigen::TensorEvaluator.10", %"struct.Eigen::TensorEvaluator.10"* %right_evaluator.i, i64 0, i32 9
  store i8 0, i8* %m_local_kernel8.i.i.i.i, align 8, !tbaa !48
  %agg.tmp.sroa.0.sroa.4.0.agg.tmp.sroa.0.0..sroa_cast.sroa_idx14.i.i.i = getelementptr inbounds %"struct.Eigen::TensorEvaluator.10", %"struct.Eigen::TensorEvaluator.10"* %right_evaluator.i, i64 0, i32 6, i32 0, i32 0, i64 3
  %agg.tmp.sroa.0.sroa.4.0.copyload.i.i.i = load i64, i64* %agg.tmp.sroa.0.sroa.4.0.agg.tmp.sroa.0.0..sroa_cast.sroa_idx14.i.i.i, align 8, !tbaa !2
  %mul.i.i.i.i.i = mul i64 %mul72.2.i, %agg.tmp.sroa.0.sroa.4.0.copyload.i.i.i
  %33 = bitcast float* %result.i.i.i to i32*
  %m_data.i.i.i = getelementptr inbounds %"struct.Eigen::TensorEvaluator.9", %"struct.Eigen::TensorEvaluator.9"* %left_evaluator.i, i64 0, i32 0
  br label %for.body.i

for.body.i:                                       ; preds = %_ZNK5Eigen15TensorEvaluatorIKNS_19TensorConvolutionOpIKSt5arrayIlLm2EEKNS_6TensorIfLi4ELi0ElEEKNS5_IfLi2ELi0ElEEEENS_13DefaultDeviceEE8convolveElliRf.exit, %inl_ZN5Eigen15TensorEvaluatorIKNS_19TensorConvolutionOpIKSt5arrayIlLm2EEKNS_6TensorIfLi4ELi0ElEEKNS5_IfLi2ELi0ElEEEENS_13DefaultDeviceEEC2ERSB_RKSC_.exit
  %i.011.i = phi i64 [ 0, %inl_ZN5Eigen15TensorEvaluatorIKNS_19TensorConvolutionOpIKSt5arrayIlLm2EEKNS_6TensorIfLi4ELi0ElEEKNS5_IfLi2ELi0ElEEEENS_13DefaultDeviceEEC2ERSB_RKSC_.exit ], [ %inc.i, %_ZNK5Eigen15TensorEvaluatorIKNS_19TensorConvolutionOpIKSt5arrayIlLm2EEKNS_6TensorIfLi4ELi0ElEEKNS5_IfLi2ELi0ElEEEENS_13DefaultDeviceEE8convolveElliRf.exit ]
  store float 0.000000e+00, float* %result.i.i.i, align 4, !tbaa !25
  %34 = load i64, i64* %arrayidx.i.i.2.i, align 8, !tbaa !21
  %div.i.i.i.i = sdiv i64 %i.011.i, %34
  %35 = load i64, i64* %arrayidx.i.i126.2.i, align 8, !tbaa !21
  %mul.i.i.i8.i = mul nsw i64 %35, %div.i.i.i.i
  %mul7.i.i.i.i = mul nsw i64 %div.i.i.i.i, %34
  %sub.i.i.i.i = sub nsw i64 %i.011.i, %mul7.i.i.i.i
  %36 = load i64, i64* %arrayidx.i.i.1.i, align 8, !tbaa !21
  %div.i.i.i.i.1 = sdiv i64 %sub.i.i.i.i, %36
  %37 = load i64, i64* %arrayidx.i.i126.1.i, align 8, !tbaa !21
  %mul.i.i.i8.i.1 = mul nsw i64 %37, %div.i.i.i.i.1
  %add.i.i.i.i.1 = add nsw i64 %mul.i.i.i8.i.1, %mul.i.i.i8.i
  %mul7.i.i.i.i.1 = mul nsw i64 %div.i.i.i.i.1, %36
  %sub.i.i.i.i.1 = sub nsw i64 %sub.i.i.i.i, %mul7.i.i.i.i.1
  %38 = load i64, i64* %arrayidx.i.i.i2, align 8, !tbaa !21
  %div.i.i.i.i.2 = sdiv i64 %sub.i.i.i.i.1, %38
  %39 = load i64, i64* %arrayidx.i.i126.i, align 8, !tbaa !21
  %mul.i.i.i8.i.2 = mul nsw i64 %39, %div.i.i.i.i.2
  %add.i.i.i.i.2 = add nsw i64 %mul.i.i.i8.i.2, %add.i.i.i.i.1
  %mul7.i.i.i.i.2 = mul nsw i64 %div.i.i.i.i.2, %38
  %sub.i.i.i.i.2 = sub nsw i64 %sub.i.i.i.i.1, %mul7.i.i.i.i.2
  %a21 = load float*, float** %m_data.i.i.i, align 8, !tbaa !37
  %add8.i.i.i.i = add nsw i64 %add.i.i.i.i.2, %sub.i.i.i.i.2
  %40 = load i64, i64* %arrayidx.i.i122.i, align 8, !tbaa !21
  %41 = load i64, i64* %arrayidx.i.i115152.i, align 8, !tbaa !21
  %42 = load i64, i64* %arrayidx.i.i118.i, align 8, !tbaa !21
  br label %for.body.us.i

for.body.us.i:                                    ; preds = %Z_ZNK5Eigen15TensorEvaluatorIKNS_19TensorConvolutionOpIKSt5arrayIlLm2EEKNS_6TensorIfLi4ELi0ElEEKNS5_IfLi2ELi0ElEEEENS_13DefaultDeviceEE8convolveElliRf.exit.i, %for.body.i
  %indvars.iv.i = phi i64 [ %indvars.iv.next.i, %Z_ZNK5Eigen15TensorEvaluatorIKNS_19TensorConvolutionOpIKSt5arrayIlLm2EEKNS_6TensorIfLi4ELi0ElEEKNS5_IfLi2ELi0ElEEEENS_13DefaultDeviceEE8convolveElliRf.exit.i ], [ 0, %for.body.i ]
  %mul.us.i = mul nsw i64 %41, %indvars.iv.i
  %add.us.i = add nsw i64 %mul.us.i, %add8.i.i.i.i
  %mul10.us.i = mul nsw i64 %42, %indvars.iv.i
  %43 = load i64, i64* %arrayidx.i.i122157.i, align 8, !tbaa !21
  %44 = load i64, i64* %arrayidx.i.i115.i, align 8, !tbaa !21
  %45 = load i64, i64* %arrayidx.i.i117.i, align 8, !tbaa !21
  %m_data.i.i.i2 = getelementptr inbounds %"struct.Eigen::TensorEvaluator.10", %"struct.Eigen::TensorEvaluator.10"* %right_evaluator.i, i64 0, i32 4, i32 0
  %.pre37.i.i = load float*, float** %m_data.i.i.i2, align 8, !tbaa !39
  %.pre38.i.i = load float*, float** %m_kernel.i.i.i.i, align 8, !tbaa !44
  %.pre39.i.i = load float, float* %result.i.i.i, align 4, !tbaa !25
  br label %for.body.i.i

for.body.i.i:                                     ; preds = %for.body.i.i, %for.body.us.i
  %a4.i.i = phi float [ %.pre39.i.i, %for.body.us.i ], [ %add15.i.i, %for.body.i.i ]
  %indvars.iv34.i.i = phi i64 [ 0, %for.body.us.i ], [ %indvars.iv.next35.i.i, %for.body.i.i ]
  %mul.i.i = mul nsw i64 %44, %indvars.iv34.i.i
  %add.i.i = add nsw i64 %mul.i.i, %add.us.i
  %mul10.i.i = mul nsw i64 %45, %indvars.iv34.i.i
  %add11.i.i = add nsw i64 %mul10.i.i, %mul10.us.i
  %add.ptr.i.i.i = getelementptr inbounds float, float* %.pre37.i.i, i64 %add.i.i
  %add.ptr.val.i.i.i = load float, float* %add.ptr.i.i.i, align 4, !tbaa !25
  %arrayidx.i.i = getelementptr inbounds float, float* %.pre38.i.i, i64 %add11.i.i
  %a5.i.i = load float, float* %arrayidx.i.i, align 4, !tbaa !25
  %mul14.i.i = fmul fast float %a5.i.i, %add.ptr.val.i.i.i
  %add15.i.i = fadd fast float %a4.i.i, %mul14.i.i
  store float %add15.i.i, float* %result.i.i.i, align 4, !tbaa !25
  %indvars.iv.next35.i.i = add nuw nsw i64 %indvars.iv34.i.i, 1
  %cmp.i.i = icmp sgt i64 %43, %indvars.iv.next35.i.i
  br i1 %cmp.i.i, label %for.body.i.i, label %Z_ZNK5Eigen15TensorEvaluatorIKNS_19TensorConvolutionOpIKSt5arrayIlLm2EEKNS_6TensorIfLi4ELi0ElEEKNS5_IfLi2ELi0ElEEEENS_13DefaultDeviceEE8convolveElliRf.exit.i

Z_ZNK5Eigen15TensorEvaluatorIKNS_19TensorConvolutionOpIKSt5arrayIlLm2EEKNS_6TensorIfLi4ELi0ElEEKNS5_IfLi2ELi0ElEEEENS_13DefaultDeviceEE8convolveElliRf.exit.i: ; preds = %for.body.i.i
  %indvars.iv.next.i = add nuw nsw i64 %indvars.iv.i, 1
  %cmp.us.i = icmp sgt i64 %40, %indvars.iv.next.i
  br i1 %cmp.us.i, label %for.body.us.i, label %_ZNK5Eigen15TensorEvaluatorIKNS_19TensorConvolutionOpIKSt5arrayIlLm2EEKNS_6TensorIfLi4ELi0ElEEKNS5_IfLi2ELi0ElEEEENS_13DefaultDeviceEE8convolveElliRf.exit

_ZNK5Eigen15TensorEvaluatorIKNS_19TensorConvolutionOpIKSt5arrayIlLm2EEKNS_6TensorIfLi4ELi0ElEEKNS5_IfLi2ELi0ElEEEENS_13DefaultDeviceEE8convolveElliRf.exit: ; preds = %Z_ZNK5Eigen15TensorEvaluatorIKNS_19TensorConvolutionOpIKSt5arrayIlLm2EEKNS_6TensorIfLi4ELi0ElEEKNS5_IfLi2ELi0ElEEEENS_13DefaultDeviceEE8convolveElliRf.exit.i
  %a20 = load i32, i32* %33, align 4, !tbaa !25
  %arrayidx.i.i.i = getelementptr inbounds float, float* %a21, i64 %i.011.i
  %a22 = bitcast float* %arrayidx.i.i.i to i32*
  store i32 %a20, i32* %a22, align 4, !tbaa !25
  %inc.i = add nuw nsw i64 %i.011.i, 1
  %exitcond.i = icmp eq i64 %inc.i, %mul.i.i.i.i.i
  br i1 %exitcond.i, label %if.end.loopexit.i, label %for.body.i

if.end.loopexit.i:                                ; preds = %_ZNK5Eigen15TensorEvaluatorIKNS_19TensorConvolutionOpIKSt5arrayIlLm2EEKNS_6TensorIfLi4ELi0ElEEKNS5_IfLi2ELi0ElEEEENS_13DefaultDeviceEE8convolveElliRf.exit
  ret void
}

declare dso_local double @__enzyme_autodiff(i8*, %"class.Eigen::Tensor.1"*, %"class.Eigen::Tensor.1"*, %"class.Eigen::Tensor"*, %"class.Eigen::Tensor"*, %"class.Eigen::Tensor"*, %"class.Eigen::Tensor"*) local_unnamed_addr #3

; Function Attrs: nounwind
declare dso_local i32 @fprintf(%struct._IO_FILE* nocapture, i8* nocapture readonly, ...) local_unnamed_addr #4

; Function Attrs: noreturn nounwind
declare dso_local void @abort() local_unnamed_addr #5

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1) #1

; Function Attrs: nounwind
declare dso_local void @free(i8* nocapture) local_unnamed_addr #4

; Function Attrs: nounwind
declare dso_local noalias i8* @malloc(i64) local_unnamed_addr #4

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local void @subfn(float** %m_data.i.i, float* %K) unnamed_addr #6 align 2 {
entry:
  %call = tail call noalias nonnull dereferenceable(36) dereferenceable_or_null(36) i8* @malloc(i64 36) #7
  %a0 = bitcast i8* %call to float*
  store float* %a0, float** %m_data.i.i, align 8, !tbaa !13
  br label %for

for:
  %a4 = load float, float* %K, align 4, !tbaa !25
  store float %a4, float* %a0, align 4, !tbaa !25
  br label %exit

exit:
  ret void
}

; Function Attrs: nounwind
declare dso_local i32 @clock_gettime(i32, %struct.timespec*) local_unnamed_addr #4

; Function Attrs: nounwind
declare dso_local i64 @random() local_unnamed_addr #4

; Function Attrs: nounwind
declare i32 @puts(i8* nocapture readonly) local_unnamed_addr #7

; Function Attrs: nounwind readnone speculatable
declare float @llvm.fabs.f32(float) #8

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1) #1

attributes #0 = { norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #3 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #4 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #5 = { noreturn nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #6 = { inlinehint nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #7 = { nounwind }
attributes #8 = { nounwind readnone speculatable }
attributes #9 = { cold }
attributes #10 = { noreturn nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0, i64 32}
!3 = !{!4, i64 32, !"_ZTSSt5arrayIlLm4EE", !6, i64 0, i64 32}
!4 = !{!5, i64 1, !"omnipotent char"}
!5 = !{!"Simple C++ TBAA"}
!6 = !{!4, i64 8, !"long"}
!7 = !{!8, !9, i64 0, i64 8}
!8 = !{!4, i64 40, !"_ZTSN5Eigen13TensorStorageIfNS_6DSizesIlLi4EEELi0EEE", !9, i64 0, i64 8, !10, i64 8, i64 32}
!9 = !{!4, i64 8, !"any pointer"}
!10 = !{!4, i64 32, !"_ZTSN5Eigen6DSizesIlLi4EEE"}
!11 = !{!12, !12, i64 0, i64 16}
!12 = !{!4, i64 16, !"_ZTSSt5arrayIlLm2EE", !6, i64 0, i64 16}
!13 = !{!14, !9, i64 0, i64 8}
!14 = !{!4, i64 24, !"_ZTSN5Eigen13TensorStorageIfNS_6DSizesIlLi2EEELi0EEE", !9, i64 0, i64 8, !15, i64 8, i64 16}
!15 = !{!4, i64 16, !"_ZTSN5Eigen6DSizesIlLi2EEE"}
!16 = !{!17}
!17 = distinct !{!17, !18, !"_ZNK5Eigen10TensorBaseINS_6TensorIfLi4ELi0ElEELi0EE6randomEv: %agg.result"}
!18 = distinct !{!18, !"_ZNK5Eigen10TensorBaseINS_6TensorIfLi4ELi0ElEELi0EE6randomEv"}
!19 = !{!20, !6, i64 8, i64 8}
!20 = !{!4, i64 16, !"_ZTS8timespec", !6, i64 0, i64 8, !6, i64 8, i64 8}
!21 = !{!6, !6, i64 0, i64 8}
!22 = !{i64 0, i64 32, !23}
!23 = !{!6, !6, i64 0, i64 32}
!24 = !{!10, !10, i64 0, i64 32}
!25 = !{!26, !26, i64 0, i64 4}
!26 = !{!4, i64 4, !"float"}
!27 = !{!28}
!28 = distinct !{!28, !29, !"_ZNK5Eigen10TensorBaseINS_6TensorIfLi2ELi0ElEELi0EE6randomEv: %agg.result"}
!29 = distinct !{!29, !"_ZNK5Eigen10TensorBaseINS_6TensorIfLi2ELi0ElEELi0EE6randomEv"}
!30 = !{i64 0, i64 16, !31}
!31 = !{!6, !6, i64 0, i64 16}
!32 = !{!15, !15, i64 0, i64 16}
!33 = !{!34}
!34 = distinct !{!34, !35, !"_ZNK5Eigen10TensorBaseINS_6TensorIfLi4ELi0ElEELi0EE6randomEv: %agg.result"}
!35 = distinct !{!35, !"_ZNK5Eigen10TensorBaseINS_6TensorIfLi4ELi0ElEELi0EE6randomEv"}
!36 = !{!9, !9, i64 0, i64 8}
!37 = !{!38, !9, i64 0, i64 8}
!38 = !{!4, i64 56, !"_ZTSN5Eigen15TensorEvaluatorINS_6TensorIfLi4ELi0ElEENS_13DefaultDeviceEEE", !9, i64 0, i64 8, !10, i64 8, i64 32, !9, i64 40, i64 8, !9, i64 48, i64 8}
!39 = !{!40, !9, i64 0, i64 8}
!40 = !{!4, i64 56, !"_ZTSN5Eigen15TensorEvaluatorIKNS_6TensorIfLi4ELi0ElEENS_13DefaultDeviceEEE", !9, i64 0, i64 8, !10, i64 8, i64 32, !9, i64 40, i64 8, !9, i64 48, i64 8}
!41 = !{!42, !9, i64 0, i64 8}
!42 = !{!4, i64 40, !"_ZTSN5Eigen15TensorEvaluatorIKNS_6TensorIfLi2ELi0ElEENS_13DefaultDeviceEEE", !9, i64 0, i64 8, !15, i64 8, i64 16, !9, i64 24, i64 8, !9, i64 32, i64 8}
!43 = !{!4, !4, i64 0, i64 0}
!44 = !{!45, !9, i64 248, i64 8}
!45 = !{!4, i64 272, !"_ZTSN5Eigen15TensorEvaluatorIKNS_19TensorConvolutionOpIKSt5arrayIlLm2EEKNS_6TensorIfLi4ELi0ElEEKNS5_IfLi2ELi0ElEEEENS_13DefaultDeviceEEE", !3, i64 0, i64 32, !3, i64 32, i64 32, !12, i64 64, i64 16, !12, i64 80, i64 16, !40, i64 96, i64 56, !42, i64 152, i64 40, !10, i64 192, i64 32, !46, i64 224, i64 24, !9, i64 248, i64 8, !47, i64 256, i64 1, !9, i64 264, i64 8}
!46 = !{!4, i64 24, !"_ZTSN5Eigen6TensorIfLi2ELi0ElEE", !14, i64 0, i64 24}
!47 = !{!4, i64 1, !"bool"}
!48 = !{!45, !47, i64 256, i64 1}
!49 = distinct !{!49, !50}
!50 = !{!"llvm.loop.unroll.disable"}

; CHECK: define internal i8* @augmented_subfn(float** %m_data.i.i, float** %"m_data.i.i'", float* %K, float* %"K'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %call = tail call noalias nonnull dereferenceable(36) dereferenceable_or_null(36) i8* @malloc(i64 36) #7
; CHECK-NEXT:   %"call'mi" = tail call noalias nonnull dereferenceable(36) dereferenceable_or_null(36) i8* @malloc(i64 36) #7
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull align 1 dereferenceable(36) dereferenceable_or_null(36) %"call'mi", i8 0, i64 36, i1 false)
; CHECK-NEXT:   %0 = bitcast float** %"m_data.i.i'" to i8**
; CHECK-NEXT:   store i8* %"call'mi", i8** %0, align 8
; CHECK-NEXT:   %1 = bitcast float** %m_data.i.i to i8**
; CHECK-NEXT:   store i8* %call, i8** %1, align 8, !tbaa !13
; CHECK-NEXT:   %2 = bitcast float* %K to i32*
; CHECK-NEXT:   %a41 = load i32, i32* %2, align 4, !tbaa !
; CHECK-NEXT:   %3 = bitcast i8* %call to i32*
; CHECK-NEXT:   store i32 %a41, i32* %3, align 4, !tbaa !
; CHECK-NEXT:   ret i8* %"call'mi"
; CHECK-NEXT: }

; CHECK: define internal void @diffesubfn(float** %m_data.i.i, float** %"m_data.i.i'", float* %K, float* %"K'", i8* %"call'mi")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[a0ipc:.+]] = bitcast i8* %"call'mi" to float*
; CHECK-NEXT:   %0 = load float, float* %[[a0ipc]], align 4
; CHECK-NEXT:   store float 0.000000e+00, float* %[[a0ipc]], align 4
; CHECK-NEXT:   %1 = load float, float* %"K'", align 4
; CHECK-NEXT:   %2 = fadd fast float %1, %0
; CHECK-NEXT:   store float %2, float* %"K'", align 4
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[callpmi:.+]])
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
