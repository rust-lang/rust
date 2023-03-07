; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -early-cse -simplifycfg -correlated-propagation -instsimplify -adce -S | FileCheck %s
source_filename = "/mnt/Data/git/Enzyme/enzyme/test/Integration/eigensumsqdyn.cpp"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }
%"class.Eigen::CwiseBinaryOp.30" = type <{ %"class.Eigen::Transpose", %"class.Eigen::Block.47", %"struct.Eigen::internal::scalar_product_op", [7 x i8] }>
%"class.Eigen::Transpose" = type { %"class.Eigen::Block" }
%"class.Eigen::Block" = type { %"class.Eigen::BlockImpl" }
%"class.Eigen::BlockImpl" = type { %"class.Eigen::internal::BlockImpl_dense" }
%"class.Eigen::internal::BlockImpl_dense" = type { %"class.Eigen::MapBase", %"class.Eigen::Matrix"*, %"class.Eigen::internal::variable_if_dynamic", %"class.Eigen::internal::variable_if_dynamic", i64 }
%"class.Eigen::MapBase" = type { double*, %"class.Eigen::internal::variable_if_dynamic.46", %"class.Eigen::internal::variable_if_dynamic" }
%"class.Eigen::internal::variable_if_dynamic.46" = type { i8 }
%"class.Eigen::Matrix" = type { %"class.Eigen::PlainObjectBase" }
%"class.Eigen::PlainObjectBase" = type { %"class.Eigen::DenseStorage" }
%"class.Eigen::DenseStorage" = type { double*, i64, i64 }
%"class.Eigen::internal::variable_if_dynamic" = type { i64 }
%"class.Eigen::Block.47" = type { %"class.Eigen::BlockImpl.48" }
%"class.Eigen::BlockImpl.48" = type { %"class.Eigen::internal::BlockImpl_dense.49" }
%"class.Eigen::internal::BlockImpl_dense.49" = type { %"class.Eigen::MapBase.base", %"class.Eigen::Matrix"*, %"class.Eigen::internal::variable_if_dynamic", %"class.Eigen::internal::variable_if_dynamic", i64 }
%"class.Eigen::MapBase.base" = type <{ double*, %"class.Eigen::internal::variable_if_dynamic", %"class.Eigen::internal::variable_if_dynamic.46" }>
%"struct.Eigen::internal::scalar_product_op" = type { i8 }
%"class.Eigen::MatrixBase.36" = type { i8 }

$_ZN5Eigen9TransposeIKNS_5BlockIKNS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1ELin1ELb0EEEEC2ERS6_ = comdat any

@stderr = external dso_local local_unnamed_addr global %struct._IO_FILE*, align 8
@.str = private unnamed_addr constant [68 x i8] c"Assertion Failed: fabs( [%s = %g] - [%s = %g] ) > %g at %s:%d (%s)\0A\00", align 1
@.str.1 = private unnamed_addr constant [9 x i8] c"Wp(i, o)\00", align 1
@.str.2 = private unnamed_addr constant [4 x i8] c"-8.\00", align 1
@.str.3 = private unnamed_addr constant [63 x i8] c"/mnt/Data/git/Enzyme/enzyme/test/Integration/eigensumsqdyn.cpp\00", align 1
@__PRETTY_FUNCTION__.main = private unnamed_addr constant [23 x i8] c"int main(int, char **)\00", align 1
@.str.4 = private unnamed_addr constant [19 x i8] c"Wp(o=%d, i=%d)=%f\0A\00", align 1
@.str.5 = private unnamed_addr constant [9 x i8] c"Mp(i, o)\00", align 1
@.str.6 = private unnamed_addr constant [3 x i8] c"8.\00", align 1
@.str.7 = private unnamed_addr constant [19 x i8] c"Mp(o=%d, i=%d)=%f\0A\00", align 1
@_ZZN5Eigen8internal15queryCacheSizesERiS1_S1_E12GenuineIntel = private unnamed_addr constant [3 x i32] [i32 1970169159, i32 1231384169, i32 1818588270], align 4
@_ZZN5Eigen8internal15queryCacheSizesERiS1_S1_E12AuthenticAMD = private unnamed_addr constant [3 x i32] [i32 1752462657, i32 1769238117, i32 1145913699], align 4
@_ZZN5Eigen8internal15queryCacheSizesERiS1_S1_E12AMDisbetter_ = private unnamed_addr constant [3 x i32] [i32 1766083905, i32 1952801395, i32 561145204], align 4

; Function Attrs: norecurse nounwind uwtable
define dso_local i32 @main(i32 %argc, i8** nocapture readnone %argv) local_unnamed_addr #0 {
entry:
  %call.i.i.i.i.i.i.i = call noalias i8* @malloc(i64 128) #7
  %0 = bitcast i8* %call.i.i.i.i.i.i.i to double*
  br label %for.body.i.i

for.body.i.i:                                     ; preds = %for.body.i.i, %entry
  %i.07.i.i = phi i64 [ %inc.i.i, %for.body.i.i ], [ 0, %entry ]
  %arrayidx.i.i.i.i = getelementptr inbounds double, double* %0, i64 %i.07.i.i
  %1 = bitcast double* %arrayidx.i.i.i.i to i64*
  store i64 4607182418800017408, i64* %1, align 8, !tbaa !2
  %inc.i.i = add nuw nsw i64 %i.07.i.i, 1
  %exitcond.i.i = icmp eq i64 %inc.i.i, 16
  br i1 %exitcond.i.i, label %_ZN5Eigen8internal26call_dense_assignment_loopINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EENS0_9assign_opIddEEEEvRT_RKT0_RKT1_.exit, label %for.body.i.i

_ZN5Eigen8internal26call_dense_assignment_loopINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EENS0_9assign_opIddEEEEvRT_RKT0_RKT1_.exit: ; preds = %for.body.i.i
  %call.i.i.i.i.i.i.i12 = call noalias i8* @malloc(i64 128) #7
  %2 = bitcast i8* %call.i.i.i.i.i.i.i12 to double*
  br label %for.body.i.i51

for.body.i.i51:                                   ; preds = %for.body.i.i51, %_ZN5Eigen8internal26call_dense_assignment_loopINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EENS0_9assign_opIddEEEEvRT_RKT0_RKT1_.exit
  %i.07.i.i44 = phi i64 [ %inc.i.i49, %for.body.i.i51 ], [ 0, %_ZN5Eigen8internal26call_dense_assignment_loopINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EENS0_9assign_opIddEEEEvRT_RKT0_RKT1_.exit ]
  %arrayidx.i.i.i.i46 = getelementptr inbounds double, double* %2, i64 %i.07.i.i44
  %3 = bitcast double* %arrayidx.i.i.i.i46 to i64*
  store i64 4611686018427387904, i64* %3, align 8, !tbaa !2
  %inc.i.i49 = add nuw nsw i64 %i.07.i.i44, 1
  %exitcond.i.i50 = icmp eq i64 %inc.i.i49, 16
  br i1 %exitcond.i.i50, label %_ZN5Eigen8internal26call_dense_assignment_loopINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EENS0_9assign_opIddEEEEvRT_RKT0_RKT1_.exit52, label %for.body.i.i51

_ZN5Eigen8internal26call_dense_assignment_loopINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EENS0_9assign_opIddEEEEvRT_RKT0_RKT1_.exit52: ; preds = %for.body.i.i51
  %call.i.i.i.i.i.i.i24 = call noalias i8* @malloc(i64 128) #7
  %4 = bitcast i8* %call.i.i.i.i.i.i.i24 to double*
  br label %for.body.i.i93

for.body.i.i93:                                   ; preds = %for.body.i.i93, %_ZN5Eigen8internal26call_dense_assignment_loopINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EENS0_9assign_opIddEEEEvRT_RKT0_RKT1_.exit52
  %i.07.i.i86 = phi i64 [ %inc.i.i91, %for.body.i.i93 ], [ 0, %_ZN5Eigen8internal26call_dense_assignment_loopINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EENS0_9assign_opIddEEEEvRT_RKT0_RKT1_.exit52 ]
  %arrayidx.i.i.i.i88 = getelementptr inbounds double, double* %4, i64 %i.07.i.i86
  %5 = bitcast double* %arrayidx.i.i.i.i88 to i64*
  store i64 0, i64* %5, align 8, !tbaa !2
  %inc.i.i91 = add nuw nsw i64 %i.07.i.i86, 1
  %exitcond.i.i92 = icmp eq i64 %inc.i.i91, 16
  br i1 %exitcond.i.i92, label %_ZN5Eigen8internal26call_dense_assignment_loopINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EENS0_9assign_opIddEEEEvRT_RKT0_RKT1_.exit94, label %for.body.i.i93

_ZN5Eigen8internal26call_dense_assignment_loopINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EENS0_9assign_opIddEEEEvRT_RKT0_RKT1_.exit94: ; preds = %for.body.i.i93
  %call.i.i.i.i.i.i.i36 = call noalias i8* @malloc(i64 128) #7
  %6 = bitcast i8* %call.i.i.i.i.i.i.i36 to double*
  br label %for.body.i.i135

for.body.i.i135:                                  ; preds = %for.body.i.i135, %_ZN5Eigen8internal26call_dense_assignment_loopINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EENS0_9assign_opIddEEEEvRT_RKT0_RKT1_.exit94
  %i.07.i.i128 = phi i64 [ %inc.i.i133, %for.body.i.i135 ], [ 0, %_ZN5Eigen8internal26call_dense_assignment_loopINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EENS0_9assign_opIddEEEEvRT_RKT0_RKT1_.exit94 ]
  %arrayidx.i.i.i.i130 = getelementptr inbounds double, double* %6, i64 %i.07.i.i128
  %7 = bitcast double* %arrayidx.i.i.i.i130 to i64*
  store i64 0, i64* %7, align 8, !tbaa !2
  %inc.i.i133 = add nuw nsw i64 %i.07.i.i128, 1
  %exitcond.i.i134 = icmp eq i64 %inc.i.i133, 16
  br i1 %exitcond.i.i134, label %_ZN5Eigen8internal26call_dense_assignment_loopINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EENS0_9assign_opIddEEEEvRT_RKT0_RKT1_.exit136, label %for.body.i.i135

_ZN5Eigen8internal26call_dense_assignment_loopINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EENS0_9assign_opIddEEEEvRT_RKT0_RKT1_.exit136: ; preds = %for.body.i.i135
  %call = call double @__enzyme_autodiff(i8* bitcast (double (double*, double*)* @_ZL6matvecPKN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEES3_ to i8*), double* %0, double* %4, double* %2, double* %6)
  br label %for.cond8.preheader

for.cond8.preheader:                              ; preds = %for.cond.cleanup11, %_ZN5Eigen8internal26call_dense_assignment_loopINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EENS0_9assign_opIddEEEEvRT_RKT0_RKT1_.exit136
  %indvars.iv103 = phi i64 [ 0, %_ZN5Eigen8internal26call_dense_assignment_loopINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EENS0_9assign_opIddEEEEvRT_RKT0_RKT1_.exit136 ], [ %indvars.iv.next104, %for.cond.cleanup11 ]
  %8 = trunc i64 %indvars.iv103 to i32
  %mul.i.i = mul nsw i64 4, %indvars.iv103
  br label %for.body12

for.cond.cleanup11:                               ; preds = %if.end
  %indvars.iv.next104 = add nuw nsw i64 %indvars.iv103, 1
  %cmp = icmp ult i64 %indvars.iv.next104, 4
  br i1 %cmp, label %for.cond8.preheader, label %for.cond35.preheader

for.body12:                                       ; preds = %if.end, %for.cond8.preheader
  %indvars.iv101 = phi i64 [ 0, %for.cond8.preheader ], [ %indvars.iv.next102, %if.end ]
  %add.i.i = add nsw i64 %mul.i.i, %indvars.iv101
  %arrayidx.i.i = getelementptr inbounds double, double* %4, i64 %add.i.i
  %9 = load double, double* %arrayidx.i.i, align 8, !tbaa !2
  %sub = fadd double %9, 8.000000e+00
  %10 = call double @llvm.fabs.f64(double %sub)
  %cmp16 = fcmp ogt double %10, 1.000000e-10
  %11 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !6
  br i1 %cmp16, label %if.then, label %if.end

if.then:                                          ; preds = %for.body12
  %.lcssa6 = phi double [ %9, %for.body12 ]
  %.lcssa4 = phi %struct._IO_FILE* [ %11, %for.body12 ]
  %call20 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %.lcssa4, i8* getelementptr inbounds ([68 x i8], [68 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str.1, i64 0, i64 0), double %.lcssa6, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.2, i64 0, i64 0), double -8.000000e+00, double 1.000000e-10, i8* getelementptr inbounds ([63 x i8], [63 x i8]* @.str.3, i64 0, i64 0), i32 61, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @__PRETTY_FUNCTION__.main, i64 0, i64 0)) #9
  call void @abort() #10
  unreachable

if.end:                                           ; preds = %for.body12
  %12 = trunc i64 %indvars.iv101 to i32
  %call24 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %11, i8* getelementptr inbounds ([19 x i8], [19 x i8]* @.str.4, i64 0, i64 0), i32 %12, i32 %8, double %9) #9
  %indvars.iv.next102 = add nuw nsw i64 %indvars.iv101, 1
  %cmp10 = icmp ult i64 %indvars.iv.next102, 4
  br i1 %cmp10, label %for.body12, label %for.cond.cleanup11

for.cond35.preheader:                             ; preds = %for.cond.cleanup11, %for.cond.cleanup38
  %indvars.iv99 = phi i64 [ %indvars.iv.next100, %for.cond.cleanup38 ], [ 0, %for.cond.cleanup11 ]
  %13 = trunc i64 %indvars.iv99 to i32
  %mul.i.i25 = mul nsw i64 4, %indvars.iv99
  br label %for.body39

for.cond.cleanup32:                               ; preds = %for.cond.cleanup38
  call void @free(i8* %call.i.i.i.i.i.i.i36) #7
  call void @free(i8* %call.i.i.i.i.i.i.i24) #7
  call void @free(i8* %call.i.i.i.i.i.i.i12) #7
  call void @free(i8* %call.i.i.i.i.i.i.i) #7
  ret i32 0

for.cond.cleanup38:                               ; preds = %if.end50
  %indvars.iv.next100 = add nuw nsw i64 %indvars.iv99, 1
  %cmp31 = icmp ult i64 %indvars.iv.next100, 4
  br i1 %cmp31, label %for.cond35.preheader, label %for.cond.cleanup32

for.body39:                                       ; preds = %if.end50, %for.cond35.preheader
  %indvars.iv = phi i64 [ 0, %for.cond35.preheader ], [ %indvars.iv.next, %if.end50 ]
  %add.i.i26 = add nsw i64 %mul.i.i25, %indvars.iv
  %arrayidx.i.i27 = getelementptr inbounds double, double* %6, i64 %add.i.i26
  %14 = load double, double* %arrayidx.i.i27, align 8, !tbaa !2
  %sub43 = fadd double %14, -8.000000e+00
  %15 = call double @llvm.fabs.f64(double %sub43)
  %cmp44 = fcmp ogt double %15, 1.000000e-10
  %16 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !6
  br i1 %cmp44, label %if.then45, label %if.end50

if.then45:                                        ; preds = %for.body39
  %.lcssa2 = phi double [ %14, %for.body39 ]
  %.lcssa = phi %struct._IO_FILE* [ %16, %for.body39 ]
  %call49 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %.lcssa, i8* getelementptr inbounds ([68 x i8], [68 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str.5, i64 0, i64 0), double %.lcssa2, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.6, i64 0, i64 0), double 8.000000e+00, double 1.000000e-10, i8* getelementptr inbounds ([63 x i8], [63 x i8]* @.str.3, i64 0, i64 0), i32 67, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @__PRETTY_FUNCTION__.main, i64 0, i64 0)) #9
  call void @abort() #10
  unreachable

if.end50:                                         ; preds = %for.body39
  %17 = trunc i64 %indvars.iv to i32
  %call54 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %16, i8* getelementptr inbounds ([19 x i8], [19 x i8]* @.str.7, i64 0, i64 0), i32 %17, i32 %13, double %14) #9
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %cmp37 = icmp ult i64 %indvars.iv.next, 4
  br i1 %cmp37, label %for.body39, label %for.cond.cleanup38
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

declare dso_local double @__enzyme_autodiff(i8*, double*, double*, double*, double*)

; Function Attrs: alwaysinline nounwind uwtable
define internal double @_ZL6matvecPKN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEES3_(double* noalias %W, double* noalias %M) #2 {
entry:
  %call.i.i.i.i.i.i.i = call noalias i8* @malloc(i64 128) #7
  %0 = bitcast i8* %call.i.i.i.i.i.i.i to double*
  br label %for.body.i.i

for.body.i.i:                                     ; preds = %for.body.i.i, %entry
  %i.07.i.i = phi i64 [ %inc.i.i, %for.body.i.i ], [ 0, %entry ]
  %Oi = getelementptr inbounds double, double* %0, i64 %i.07.i.i
  %arrayidx.i.i.i.i.i = getelementptr inbounds double, double* %W, i64 %i.07.i.i
  %arrayidx.i2.i.i.i.i = getelementptr inbounds double, double* %M, i64 %i.07.i.i
  %1 = load double, double* %arrayidx.i.i.i.i.i, align 8, !tbaa !2
  %2 = load double, double* %arrayidx.i2.i.i.i.i, align 8, !tbaa !2
  %sub = fsub double %1, %2
  store double %sub, double* %Oi, align 8, !tbaa !2
  %inc.i.i = add nuw nsw i64 %i.07.i.i, 1
  %exitcond.i.i = icmp eq i64 %inc.i.i, 16
  br i1 %exitcond.i.i, label %_ZN5Eigen8internal26call_dense_assignment_loopINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_13CwiseBinaryOpINS0_20scalar_difference_opIddEEKS3_S7_EENS0_9assign_opIddEEEEvRT_RKT0_RKT1_.exit, label %for.body.i.i

_ZN5Eigen8internal26call_dense_assignment_loopINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_13CwiseBinaryOpINS0_20scalar_difference_opIddEEKS3_S7_EENS0_9assign_opIddEEEEvRT_RKT0_RKT1_.exit: ; preds = %for.body.i.i
  %call.i.i.i.i.i.i.i13 = call noalias i8* @malloc(i64 128) #7
  %3 = bitcast i8* %call.i.i.i.i.i.i.i13 to double*
  call void @subfn(double* %3, double* %0) #7
  br label %for.body.i

for.body.i:                                       ; preds = %for.body.i, %_ZN5Eigen8internal26call_dense_assignment_loopINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_13CwiseBinaryOpINS0_20scalar_difference_opIddEEKS3_S7_EENS0_9assign_opIddEEEEvRT_RKT0_RKT1_.exit
  %i.047.i = phi i64 [ 0, %_ZN5Eigen8internal26call_dense_assignment_loopINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_13CwiseBinaryOpINS0_20scalar_difference_opIddEEKS3_S7_EENS0_9assign_opIddEEEEvRT_RKT0_RKT1_.exit ], [ %inc.i, %for.body.i ]
  %inc.i = add nuw nsw i64 %i.047.i, 1
  %cmp.i = icmp slt i64 %inc.i, 4
  br i1 %cmp.i, label %for.body.i, label %for.cond10.preheader.i

for.cond10.preheader.i:                           ; preds = %for.body.i, %for.cond.cleanup13.i
  %res.i.sroa.0.1 = phi i64 [ %.lcssa, %for.cond.cleanup13.i ], [ 0, %for.body.i ]
  %i4.044.i = phi i64 [ %inc22.i, %for.cond.cleanup13.i ], [ 0, %for.body.i ]
  %mul.i.i = mul nsw i64 4, %i4.044.i
  br label %for.body14.i

for.cond.cleanup13.i:                             ; preds = %for.body14.i
  %add.i.i.lcssa = phi double [ %add.i.i, %for.body14.i ]
  %.lcssa = phi i64 [ %8, %for.body14.i ]
  %inc22.i = add nuw nsw i64 %i4.044.i, 1
  %cmp7.i = icmp slt i64 %inc22.i, 4
  br i1 %cmp7.i, label %for.cond10.preheader.i, label %_ZN5Eigen8internal10redux_implINS0_13scalar_sum_opIddEENS0_15redux_evaluatorINS_7ProductINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEES7_Li0EEEEELi0ELi0EE3runERKS9_RKS3_.exit

for.body14.i:                                     ; preds = %for.body14.i, %for.cond10.preheader.i
  %res.i.sroa.0.2 = phi i64 [ %res.i.sroa.0.1, %for.cond10.preheader.i ], [ %8, %for.body14.i ]
  %j.041.i = phi i64 [ %inc19.i, %for.body14.i ], [ 0, %for.cond10.preheader.i ]
  %add.i4.i = add nsw i64 %mul.i.i, %j.041.i
  %arrayidx.i.i = getelementptr inbounds double, double* %3, i64 %add.i4.i
  %4 = bitcast double* %arrayidx.i.i to i64*
  %5 = load i64, i64* %4, align 8, !tbaa !2
  %6 = bitcast i64 %res.i.sroa.0.2 to double
  %7 = bitcast i64 %5 to double
  %add.i.i = fadd double %6, %7
  %8 = bitcast double %add.i.i to i64
  %inc19.i = add nuw nsw i64 %j.041.i, 1
  %cmp12.i = icmp slt i64 %inc19.i, 4
  br i1 %cmp12.i, label %for.body14.i, label %for.cond.cleanup13.i

_ZN5Eigen8internal10redux_implINS0_13scalar_sum_opIddEENS0_15redux_evaluatorINS_7ProductINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEES7_Li0EEEEELi0ELi0EE3runERKS9_RKS3_.exit: ; preds = %for.cond.cleanup13.i
  %add.i.i.lcssa.lcssa = phi double [ %add.i.i.lcssa, %for.cond.cleanup13.i ]
  call void @free(i8* %call.i.i.i.i.i.i.i13) #7
  call void @free(i8* %call.i.i.i.i.i.i.i) #7
  ret double %add.i.i.lcssa.lcssa
}

; Function Attrs: nounwind readnone speculatable
declare double @llvm.fabs.f64(double) #3

; Function Attrs: nounwind
declare dso_local i32 @fprintf(%struct._IO_FILE* nocapture, i8* nocapture readonly, ...) local_unnamed_addr #4

; Function Attrs: noreturn nounwind
declare dso_local void @abort() local_unnamed_addr #5

; Function Attrs: nounwind
declare dso_local void @free(i8* nocapture) local_unnamed_addr #4

; Function Attrs: nounwind
declare dso_local noalias i8* @malloc(i64) local_unnamed_addr #4

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local void @subfn(double* %w3, double* %w9) local_unnamed_addr #6 {
entry:
  %false = call i1 @falser()
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.cond.cleanup4, %entry
  %outer.022 = phi i64 [ %inc7, %for.cond.cleanup4 ], [ 0, %entry ]
  %mul.i.i.i.i.i = mul nsw i64 4, %outer.022
  br label %for.body5

for.body5:                                        ; preds = %if.exit, %for.cond1.preheader
  %inner.019 = phi i64 [ %inc, %if.exit ], [ 0, %for.cond1.preheader ]
  %add.i.i.i = add nsw i64 %mul.i.i.i.i.i, %inner.019
  %arrayidx = getelementptr inbounds double, double* %w3, i64 %add.i.i.i
  %add.ptr = getelementptr inbounds double, double* %w9, i64 %inner.019
  br i1 %false, label %if.exit, label %if.end.i.i

if.end.i.i:                                       ; preds = %for.body5
  %call2.i.i.i = call double @sumsq(double* %add.ptr) #7
  br label %if.exit

if.exit:                                          ; preds = %if.end.i.i, %for.body5
  %retval = phi double [ %call2.i.i.i, %if.end.i.i ], [ 0.000000e+00, %for.body5 ]
  store double %retval, double* %arrayidx, align 8, !tbaa !2
  %inc = add nuw nsw i64 %inner.019, 1
  %cmp3 = icmp slt i64 %inc, 4
  br i1 %cmp3, label %for.body5, label %for.cond.cleanup4

for.cond.cleanup4:                                ; preds = %if.exit
  %inc7 = add nuw nsw i64 %outer.022, 1
  %cmp = icmp slt i64 %inc7, 4
  br i1 %cmp, label %for.cond1.preheader, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup4
  ret void
}

define i1 @falser() #22 {
entry:
  ret i1 false
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local void @_ZNK5Eigen10MatrixBaseINS_9TransposeIKNS_5BlockIKNS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1ELin1ELb0EEEEEE12cwiseProductINS2_IS5_Lin1ELi1ELb1EEEEEKNS_13CwiseBinaryOpINS_8internal17scalar_product_opIdNSD_6traitsIT_E6ScalarEEEKS8_KSG_EERKNS0_ISG_EE(%"class.Eigen::CwiseBinaryOp.30"* noalias %agg.result, %"class.Eigen::MatrixBase.36"* %this) #6 {
entry:
  %0 = bitcast %"class.Eigen::MatrixBase.36"* %this to %"class.Eigen::Transpose"*
  %m_lhs.i = getelementptr inbounds %"class.Eigen::CwiseBinaryOp.30", %"class.Eigen::CwiseBinaryOp.30"* %agg.result, i64 0, i32 0
  %m_matrix.i.i = getelementptr inbounds %"class.Eigen::Transpose", %"class.Eigen::Transpose"* %m_lhs.i, i64 0, i32 0
  %m_matrix2.i.i = getelementptr inbounds %"class.Eigen::Transpose", %"class.Eigen::Transpose"* %0, i64 0, i32 0
  %1 = getelementptr inbounds %"class.Eigen::Block", %"class.Eigen::Block"* %m_matrix.i.i, i64 0, i32 0
  %2 = getelementptr inbounds %"class.Eigen::Block", %"class.Eigen::Block"* %m_matrix2.i.i, i64 0, i32 0
  %3 = getelementptr inbounds %"class.Eigen::BlockImpl", %"class.Eigen::BlockImpl"* %1, i64 0, i32 0
  %4 = getelementptr inbounds %"class.Eigen::BlockImpl", %"class.Eigen::BlockImpl"* %2, i64 0, i32 0
  %5 = getelementptr inbounds %"class.Eigen::internal::BlockImpl_dense", %"class.Eigen::internal::BlockImpl_dense"* %3, i64 0, i32 0
  %6 = getelementptr inbounds %"class.Eigen::internal::BlockImpl_dense", %"class.Eigen::internal::BlockImpl_dense"* %4, i64 0, i32 0
  %7 = bitcast %"class.Eigen::MapBase"* %6 to i64*
  %8 = load i64, i64* %7, align 8, !tbaa !8
  %9 = bitcast %"class.Eigen::MapBase"* %5 to i64*
  store i64 %8, i64* %9, align 8, !tbaa !8
  %10 = getelementptr inbounds %"class.Eigen::MapBase", %"class.Eigen::MapBase"* %6, i64 0, i32 2, i32 0
  %11 = getelementptr inbounds %"class.Eigen::MapBase", %"class.Eigen::MapBase"* %5, i64 0, i32 2, i32 0
  %12 = load i64, i64* %10, align 8, !tbaa !13
  store i64 %12, i64* %11, align 8, !tbaa !13
  %m_xpr.i.i.i.i.i = getelementptr inbounds %"class.Eigen::internal::BlockImpl_dense", %"class.Eigen::internal::BlockImpl_dense"* %3, i64 0, i32 1
  %m_xpr2.i.i.i.i.i = getelementptr inbounds %"class.Eigen::internal::BlockImpl_dense", %"class.Eigen::internal::BlockImpl_dense"* %4, i64 0, i32 1
  %13 = bitcast %"class.Eigen::Matrix"** %m_xpr.i.i.i.i.i to i8*
  %14 = bitcast %"class.Eigen::Matrix"** %m_xpr2.i.i.i.i.i to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 %13, i8* nonnull align 8 %14, i64 32, i1 false) #7
  ret void
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local void @_ZN5Eigen9TransposeIKNS_5BlockIKNS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1ELin1ELb0EEEEC2ERS6_(%"class.Eigen::Transpose"* %this, %"class.Eigen::Block"* dereferenceable(56) %matrix) unnamed_addr #6 comdat align 2 {
entry:
  %m_matrix = getelementptr inbounds %"class.Eigen::Transpose", %"class.Eigen::Transpose"* %this, i64 0, i32 0
  %0 = getelementptr inbounds %"class.Eigen::Block", %"class.Eigen::Block"* %m_matrix, i64 0, i32 0
  %1 = getelementptr inbounds %"class.Eigen::Block", %"class.Eigen::Block"* %matrix, i64 0, i32 0
  %2 = getelementptr inbounds %"class.Eigen::BlockImpl", %"class.Eigen::BlockImpl"* %0, i64 0, i32 0
  %3 = getelementptr inbounds %"class.Eigen::BlockImpl", %"class.Eigen::BlockImpl"* %1, i64 0, i32 0
  %4 = getelementptr inbounds %"class.Eigen::internal::BlockImpl_dense", %"class.Eigen::internal::BlockImpl_dense"* %2, i64 0, i32 0
  %5 = getelementptr inbounds %"class.Eigen::internal::BlockImpl_dense", %"class.Eigen::internal::BlockImpl_dense"* %3, i64 0, i32 0
  %6 = bitcast %"class.Eigen::MapBase"* %5 to i64*
  %7 = load i64, i64* %6, align 8, !tbaa !8
  %8 = bitcast %"class.Eigen::MapBase"* %4 to i64*
  store i64 %7, i64* %8, align 8, !tbaa !8
  %9 = getelementptr inbounds %"class.Eigen::MapBase", %"class.Eigen::MapBase"* %5, i64 0, i32 2, i32 0
  %10 = getelementptr inbounds %"class.Eigen::MapBase", %"class.Eigen::MapBase"* %4, i64 0, i32 2, i32 0
  %11 = load i64, i64* %9, align 8, !tbaa !13
  store i64 %11, i64* %10, align 8, !tbaa !13
  %m_xpr.i.i.i = getelementptr inbounds %"class.Eigen::internal::BlockImpl_dense", %"class.Eigen::internal::BlockImpl_dense"* %2, i64 0, i32 1
  %m_xpr2.i.i.i = getelementptr inbounds %"class.Eigen::internal::BlockImpl_dense", %"class.Eigen::internal::BlockImpl_dense"* %3, i64 0, i32 1
  %12 = bitcast %"class.Eigen::Matrix"** %m_xpr.i.i.i to i8*
  %13 = bitcast %"class.Eigen::Matrix"** %m_xpr2.i.i.i to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 %12, i8* nonnull align 8 %13, i64 32, i1 false) #7
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1) #1

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local double @sumsq(double* %a3) local_unnamed_addr #6 {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %res.0 = phi double [ 0.000000e+00, %entry ], [ %add.i, %for.body ]
  %i.047 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %mul.i.i.i = mul nsw i64 4, %i.047
  %arrayidx.i.i.i = getelementptr inbounds double, double* %a3, i64 %mul.i.i.i
  %a6 = load double, double* %arrayidx.i.i.i, align 8, !tbaa !2
  %mul.i.i8 = fmul double %a6, %a6
  %add.i = fadd double %res.0, %mul.i.i8
  %inc = add nuw nsw i64 %i.047, 1
  %cmp = icmp slt i64 %inc, 4
  br i1 %cmp, label %for.body, label %for.cond.cleanup8

for.cond.cleanup8:                                ; preds = %for.body
  %add.i.lcssa = phi double [ %add.i, %for.body ]
  ret double %add.i.lcssa
}

; Function Attrs: nounwind
declare dso_local i32 @__cxa_guard_acquire(i64*) local_unnamed_addr #7

; Function Attrs: nounwind
declare dso_local void @__cxa_guard_release(i64*) local_unnamed_addr #7

; Function Attrs: inaccessiblemem_or_argmemonly nounwind
declare void @llvm.prefetch(i8* nocapture readonly, i32, i32, i32) #8

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1) #1

; Function Attrs: nounwind
declare i8* @llvm.stacksave() #7

; Function Attrs: nounwind
declare void @llvm.stackrestore(i8*) #7

attributes #0 = { norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { alwaysinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind readnone speculatable }
attributes #4 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { noreturn nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { inlinehint nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #7 = { nounwind }
attributes #8 = { inaccessiblemem_or_argmemonly nounwind }
attributes #9 = { cold }
attributes #10 = { noreturn nounwind }
attributes #22 = { readnone speculatable }

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
!8 = !{!9, !7, i64 0}
!9 = !{!"_ZTSN5Eigen7MapBaseINS_5BlockIKNS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi1ELin1ELb0EEELi0EEE", !7, i64 0, !10, i64 8, !11, i64 16}
!10 = !{!"_ZTSN5Eigen8internal19variable_if_dynamicIlLi1EEE"}
!11 = !{!"_ZTSN5Eigen8internal19variable_if_dynamicIlLin1EEE", !12, i64 0}
!12 = !{!"long", !4, i64 0}
!13 = !{!12, !12, i64 0}

; CHECK: define internal void @diffe_ZL6matvecPKN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEES3_(double* noalias %W, double* %"W'", double* noalias %M, double* %"M'", double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %call.i.i.i.i.i.i.i = call noalias nonnull dereferenceable(128) dereferenceable_or_null(128) i8* @malloc(i64 128)
; CHECK-NEXT:   %"call.i.i.i.i.i.i.i'mi" = call noalias nonnull dereferenceable(128) dereferenceable_or_null(128) i8* @malloc(i64 128)
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* {{(noundef )?}}nonnull dereferenceable(128) dereferenceable_or_null(128) %"call.i.i.i.i.i.i.i'mi", i8 0, i64 128, i1 false)
; CHECK-NEXT:   %"'ipc" = bitcast i8* %"call.i.i.i.i.i.i.i'mi" to double*
; CHECK-NEXT:   %0 = bitcast i8* %call.i.i.i.i.i.i.i to double*
; CHECK-NEXT:   br label %for.body.i.i

; CHECK: for.body.i.i:                                     ; preds = %for.body.i.i, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %for.body.i.i ], [ 0, %entry ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %Oi = getelementptr inbounds double, double* %0, i64 %iv
; CHECK-NEXT:   %arrayidx.i.i.i.i.i = getelementptr inbounds double, double* %W, i64 %iv
; CHECK-NEXT:   %arrayidx.i2.i.i.i.i = getelementptr inbounds double, double* %M, i64 %iv
; CHECK-NEXT:   %1 = load double, double* %arrayidx.i.i.i.i.i, align 8, !tbaa !2
; CHECK-NEXT:   %2 = load double, double* %arrayidx.i2.i.i.i.i, align 8, !tbaa !2
; CHECK-NEXT:   %sub = fsub double %1, %2
; CHECK-NEXT:   store double %sub, double* %Oi, align 8, !tbaa !2
; CHECK-NEXT:   %exitcond.i.i = icmp eq i64 %iv.next, 16
; CHECK-NEXT:   br i1 %exitcond.i.i, label %_ZN5Eigen8internal26call_dense_assignment_loopINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_13CwiseBinaryOpINS0_20scalar_difference_opIddEEKS3_S7_EENS0_9assign_opIddEEEEvRT_RKT0_RKT1_.exit, label %for.body.i.i

; CHECK: _ZN5Eigen8internal26call_dense_assignment_loopINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_13CwiseBinaryOpINS0_20scalar_difference_opIddEEKS3_S7_EENS0_9assign_opIddEEEEvRT_RKT0_RKT1_.exit: ; preds = %for.body.i.i
; CHECK-NEXT:   %call.i.i.i.i.i.i.i13 = call noalias nonnull dereferenceable(128) dereferenceable_or_null(128) i8* @malloc(i64 128)
; CHECK-NEXT:   %"call.i.i.i.i.i.i.i13'mi" = call noalias nonnull dereferenceable(128) dereferenceable_or_null(128) i8* @malloc(i64 128)
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* {{(noundef )?}}nonnull dereferenceable(128) dereferenceable_or_null(128) %"call.i.i.i.i.i.i.i13'mi", i8 0, i64 128, i1 false)
; CHECK-NEXT:   %[[ipc8:.+]] = bitcast i8* %"call.i.i.i.i.i.i.i13'mi" to double*
; CHECK-NEXT:   %[[unwrap:.+]] = bitcast i8* %call.i.i.i.i.i.i.i13 to double*
; CHECK-NEXT:   br label %for.body.i

; CHECK: for.body.i:                                       ; preds = %for.body.i, %_ZN5Eigen8internal26call_dense_assignment_loopINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_13CwiseBinaryOpINS0_20scalar_difference_opIddEEKS3_S7_EENS0_9assign_opIddEEEEvRT_RKT0_RKT1_.exit
; CHECK-NEXT:   %iv1 = phi i64 [ %iv.next2, %for.body.i ], [ 0, %_ZN5Eigen8internal26call_dense_assignment_loopINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_13CwiseBinaryOpINS0_20scalar_difference_opIddEEKS3_S7_EENS0_9assign_opIddEEEEvRT_RKT0_RKT1_.exit ]
; CHECK-NEXT:   %iv.next2 = add nuw nsw i64 %iv1, 1
; CHECK-NEXT:   %cmp.i = icmp ne i64 %iv.next2, 4
; CHECK-NEXT:   br i1 %cmp.i, label %for.body.i, label %for.cond10.preheader.i

; CHECK: for.cond10.preheader.i:
; CHECK-NEXT:   %iv3 = phi i64 [ %iv.next4, %for.cond.cleanup13.i ], [ 0, %for.body.i ]
; CHECK-NEXT:   %iv.next4 = add nuw nsw i64 %iv3, 1
; CHECK-NEXT:   br label %for.body14.i

; CHECK: for.cond.cleanup13.i:                             ; preds = %for.body14.i
; CHECK-NEXT:   %cmp7.i = icmp ne i64 %iv.next4, 4
; CHECK-NEXT:   br i1 %cmp7.i, label %for.cond10.preheader.i, label %invertfor.cond.cleanup13.i

; CHECK: for.body14.i:                                     ; preds = %for.body14.i, %for.cond10.preheader.i
; CHECK-NEXT:   %iv5 = phi i64 [ %iv.next6, %for.body14.i ], [ 0, %for.cond10.preheader.i ]
; CHECK-NEXT:   %iv.next6 = add nuw nsw i64 %iv5, 1
; CHECK-NEXT:   %cmp12.i = icmp ne i64 %iv.next6, 4
; CHECK-NEXT:   br i1 %cmp12.i, label %for.body14.i, label %for.cond.cleanup13.i

; CHECK: invertentry:                                      ; preds = %invertfor.body.i.i
; CHECK-NEXT:   tail call void @free(i8* nonnull %"call.i.i.i.i.i.i.i'mi")
; CHECK-NEXT:   tail call void @free(i8* nonnull %call.i.i.i.i.i.i.i)
; CHECK-NEXT:   ret void

; CHECK: invertfor.body.i.i:                               ; preds = %invert_ZN5Eigen8internal26call_dense_assignment_loopINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_13CwiseBinaryOpINS0_20scalar_difference_opIddEEKS3_S7_EENS0_9assign_opIddEEEEvRT_RKT0_RKT1_.exit, %incinvertfor.body.i.i
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ 15, %invert_ZN5Eigen8internal26call_dense_assignment_loopINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_13CwiseBinaryOpINS0_20scalar_difference_opIddEEKS3_S7_EENS0_9assign_opIddEEEEvRT_RKT0_RKT1_.exit ], [ %[[ivsub:.+]], %incinvertfor.body.i.i ]
; CHECK-NEXT:   %"Oi'ipg_unwrap" = getelementptr inbounds double, double* %"'ipc", i64 %"iv'ac.0"
; CHECK-NEXT:   %[[Oil:.+]] = load double, double* %"Oi'ipg_unwrap", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"Oi'ipg_unwrap", align 8
; CHECK-NEXT:   %[[neg:.+]] = {{(fsub fast double 0.000000e\+00,|fneg fast double)}} %[[Oil]]
; CHECK-NEXT:   %"arrayidx.i2.i.i.i.i'ipg_unwrap" = getelementptr inbounds double, double* %"M'", i64 %"iv'ac.0"
; CHECK-NEXT:   %[[prea:.+]] = load double, double* %"arrayidx.i2.i.i.i.i'ipg_unwrap", align 8
; CHECK-NEXT:   %[[posta:.+]] = fadd fast double %[[prea]], %[[neg]]
; CHECK-NEXT:   store double %[[posta]], double* %"arrayidx.i2.i.i.i.i'ipg_unwrap", align 8
; CHECK-NEXT:   %"arrayidx.i.i.i.i.i'ipg_unwrap" = getelementptr inbounds double, double* %"W'", i64 %"iv'ac.0"
; CHECK-NEXT:   %[[pre:.+]] = load double, double* %"arrayidx.i.i.i.i.i'ipg_unwrap", align 8
; CHECK-NEXT:   %[[post:.+]] = fadd fast double %[[pre]], %[[Oil]]
; CHECK-NEXT:   store double %[[post]], double* %"arrayidx.i.i.i.i.i'ipg_unwrap", align 8
; CHECK-NEXT:   %[[ivcmp:.+]] = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %[[ivcmp]], label %invertentry, label %incinvertfor.body.i.i

; CHECK: incinvertfor.body.i.i:                            ; preds = %invertfor.body.i.i
; CHECK-NEXT:   %[[ivsub]] = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertfor.body.i.i

; CHECK: invert_ZN5Eigen8internal26call_dense_assignment_loopINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_13CwiseBinaryOpINS0_20scalar_difference_opIddEEKS3_S7_EENS0_9assign_opIddEEEEvRT_RKT0_RKT1_.exit: ; preds = %invertfor.body.i
; CHECK-NEXT:   call void @diffesubfn(double* nonnull %[[unwrap]], double* nonnull %[[ipc8]], double* nonnull %0, double* nonnull %"'ipc")
; CHECK-NEXT:   tail call void @free(i8* nonnull %"call.i.i.i.i.i.i.i13'mi")
; CHECK-NEXT:   tail call void @free(i8* nonnull %call.i.i.i.i.i.i.i13)
; CHECK-NEXT:   br label %invertfor.body.i.i

; CHECK: invertfor.body.i:                                 ; preds = %invertfor.cond10.preheader.i, %incinvertfor.body.i
; CHECK-NEXT:   %"iv1'ac.0" = phi i64 [ %[[iv1sub:.+]], %incinvertfor.body.i ], [ 3, %invertfor.cond10.preheader.i ]
; CHECK-NEXT:   %[[iv1cmp:.+]] = icmp eq i64 %"iv1'ac.0", 0
; CHECK-NEXT:   br i1 %[[iv1cmp]], label %invert_ZN5Eigen8internal26call_dense_assignment_loopINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_13CwiseBinaryOpINS0_20scalar_difference_opIddEEKS3_S7_EENS0_9assign_opIddEEEEvRT_RKT0_RKT1_.exit, label %incinvertfor.body.i

; CHECK: incinvertfor.body.i:                              ; preds = %invertfor.body.i
; CHECK-NEXT:   %[[iv1sub]] = add nsw i64 %"iv1'ac.0", -1
; CHECK-NEXT:   br label %invertfor.body.i

; CHECK: invertfor.cond10.preheader.i:                     ; preds = %invertfor.body14.i
; CHECK-NEXT:   %[[iv3cmp:.+]] = icmp eq i64 %"iv3'ac.0", 0
; CHECK-NEXT:   %[[g15:.+]] = bitcast i64 %[[g29:.+]] to double
; CHECK-NEXT:   %[[g16:.+]] = bitcast i64 %[[sel:.+]] to double
; CHECK-NEXT:   %[[g17:.+]] = fadd fast double %[[g15]], %[[g16]]
; CHECK-NEXT:   %[[g18:.+]] = bitcast double %[[g17]] to i64
; CHECK-NEXT:   br i1 %[[iv3cmp]], label %invertfor.body.i, label %incinvertfor.cond10.preheader.i

; CHECK: incinvertfor.cond10.preheader.i:                  ; preds = %invertfor.cond10.preheader.i
; CHECK-NEXT:   %[[subiv3:.+]] = add nsw i64 %"iv3'ac.0", -1
; CHECK-NEXT:   br label %invertfor.cond.cleanup13.i

; CHECK: invertfor.cond.cleanup13.i:                       ; preds = %for.cond.cleanup13.i, %incinvertfor.cond10.preheader.i
; CHECK-NEXT:   %[[lcssade:.+]] = phi i64 [ %[[g18]], %incinvertfor.cond10.preheader.i ], [ 0, %for.cond.cleanup13.i ]
; CHECK-NEXT:   %[[addlcssa:.+]] = phi double [ 0.000000e+00, %incinvertfor.cond10.preheader.i ], [ %differeturn, %for.cond.cleanup13.i ]
; CHECK-NEXT:   %"iv3'ac.0" = phi i64 [ %[[subiv3]], %incinvertfor.cond10.preheader.i ], [ 3, %for.cond.cleanup13.i ]
; CHECK-NEXT:   br label %invertfor.body14.i

; CHECK: invertfor.body14.i:                               ; preds = %incinvertfor.body14.i, %invertfor.cond.cleanup13.i
; CHECK-NEXT:   %[[de11:.+]] = phi i64 [ %[[lcssade]], %invertfor.cond.cleanup13.i ], [ %[[bcsel:.+]], %incinvertfor.body14.i ]
; CHECK-NEXT:   %[[addiide1:.+]] = phi double [ %[[addlcssa]], %invertfor.cond.cleanup13.i ], [ 0.000000e+00, %incinvertfor.body14.i ]
; CHECK-NEXT:   %"iv5'ac.0" = phi i64 [ 3, %invertfor.cond.cleanup13.i ], [ %[[iv5inc:.+]], %incinvertfor.body14.i ]
; CHECK-NEXT:   %[[padd:.+]] = bitcast i64 %[[de11]] to double
; CHECK-NEXT:   %[[aadd:.+]] = fadd fast double %[[addiide1]], %[[padd]]
; CHECK-NEXT:   %[[abc:.+]] = bitcast double %[[aadd]] to i64
; CHECK-NEXT:   %mul.i.i_unwrap = mul nsw i64 4, %"iv3'ac.0"
; CHECK-NEXT:   %add.i4.i_unwrap = add nsw i64 %mul.i.i_unwrap, %"iv5'ac.0"
; CHECK-NEXT:   %"arrayidx.i.i'ipg_unwrap" = getelementptr inbounds double, double* %[[ipc8]], i64 %add.i4.i_unwrap
; CHECK-NEXT:   %[[bcpw:.+]] = load double, double* %"arrayidx.i.i'ipg_unwrap", align 8
; CHECK-NEXT:   %[[auw1:.+]] = fadd fast double %[[bcpw]], %[[aadd]]
; CHECK-NEXT:   store double %[[auw1]], double* %"arrayidx.i.i'ipg_unwrap", align 8
; CHECK-NEXT:   %[[iv5cmp:.+]] = icmp eq i64 %"iv5'ac.0", 0
; CHECK-NEXT:   %[[nivcmp:.+]] = xor i1 %[[iv5cmp]], true
; CHECK-NEXT:   %[[g29:.+]] = select i1 %[[nivcmp]], i64 %[[abc]], i64 0
; CHECK-NEXT:   %[[bcsel:.+]] = bitcast double %[[aadd]] to i64
; CHECK-NEXT:   %[[sel]] = select i1 %[[iv5cmp]], i64 %[[bcsel]], i64 0
; CHECK-NEXT:   br i1 %[[iv5cmp]], label %invertfor.cond10.preheader.i, label %incinvertfor.body14.i

; CHECK: incinvertfor.body14.i:                            ; preds = %invertfor.body14.i
; CHECK-NEXT:   %[[iv5inc]] = add nsw i64 %"iv5'ac.0", -1
; CHECK-NEXT:   br label %invertfor.body14.i
; CHECK-NEXT: }


; CHECK: define internal void @diffesubfn(double* %w3, double* %"w3'", double* %w9, double* %"w9'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %false = call i1 @falser()
; CHECK-NEXT:   %malloccall = tail call noalias nonnull dereferenceable(128) dereferenceable_or_null(128) i8* @malloc(i64 128)
; CHECK-NEXT:   %[[_malloccache:.+]] = bitcast i8* %malloccall to double**
; CHECK-NEXT:   br label %for.cond1.preheader

; CHECK: for.cond1.preheader:                              ; preds = %for.cond.cleanup4, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %for.cond.cleanup4 ], [ 0, %entry ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %mul.i.i.i.i.i = mul nsw i64 4, %iv
; CHECK-NEXT:   br label %for.body5

; CHECK: for.body5:                                        ; preds = %if.exit, %for.cond1.preheader
; CHECK-NEXT:   %iv1 = phi i64 [ %iv.next2, %if.exit ], [ 0, %for.cond1.preheader ]
; CHECK-NEXT:   %iv.next2 = add nuw nsw i64 %iv1, 1
; CHECK-NEXT:   %add.i.i.i = add nsw i64 %mul.i.i.i.i.i, %iv1
; CHECK-NEXT:   %arrayidx = getelementptr inbounds double, double* %w3, i64 %add.i.i.i
; CHECK-NEXT:   %"add.ptr'ipg" = getelementptr inbounds double, double* %"w9'", i64 %iv1
; CHECK-NEXT:   %add.ptr = getelementptr inbounds double, double* %w9, i64 %iv1
; CHECK-NEXT:   br i1 %false, label %if.exit, label %if.end.i.i

; CHECK: if.end.i.i:                                       ; preds = %for.body5
; CHECK-NEXT:   %call2.i.i.i_augmented = call { double*, double } @augmented_sumsq(double* %add.ptr, double* %"add.ptr'ipg")
; CHECK-NEXT:   %[[exttape:.+]] = extractvalue { double*, double } %call2.i.i.i_augmented, 0
; CHECK-NEXT:   %[[ge:.+]] = getelementptr inbounds double*, double** %[[_malloccache]], i64 %add.i.i.i
; CHECK-NEXT:   store double* %[[exttape]], double** %[[ge]]
; CHECK-NEXT:   %call2.i.i.i = extractvalue { double*, double } %call2.i.i.i_augmented, 1
; CHECK-NEXT:   br label %if.exit

; CHECK: if.exit:                                          ; preds = %if.end.i.i, %for.body5
; CHECK-NEXT:   %retval = phi double [ %call2.i.i.i, %if.end.i.i ], [ 0.000000e+00, %for.body5 ]
; CHECK-NEXT:   store double %retval, double* %arrayidx, align 8, !tbaa !2
; CHECK-NEXT:   %cmp3 = icmp ne i64 %iv.next2, 4
; CHECK-NEXT:   br i1 %cmp3, label %for.body5, label %for.cond.cleanup4

; CHECK: for.cond.cleanup4:                                ; preds = %if.exit
; CHECK-NEXT:   %cmp = icmp ne i64 %iv.next, 4
; CHECK-NEXT:   br i1 %cmp, label %for.cond1.preheader, label %invertfor.cond.cleanup4

; CHECK: invertentry:                                      ; preds = %invertfor.cond1.preheader
; CHECK-NEXT:   tail call void @free(i8* nonnull %malloccall)
; CHECK-NEXT:   ret void

; CHECK: invertfor.cond1.preheader:                        ; preds = %invertfor.body5
; CHECK-NEXT:   %[[ph1:.]] = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %[[ph1]], label %invertentry, label %incinvertfor.cond1.preheader

; CHECK: incinvertfor.cond1.preheader:                     ; preds = %invertfor.cond1.preheader
; CHECK-NEXT:   %[[ivsub:.+]] = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertfor.cond.cleanup4

; CHECK: invertfor.body5:                                  ; preds = %invertif.exit, %invertif.end.i.i
; CHECK-NEXT:   %"call2.i.i.i'de.0" = phi double [ %"call2.i.i.i'de.1", %invertif.exit ], [ 0.000000e+00, %invertif.end.i.i ]
; CHECK-NEXT:   %[[iv1cmp:.+]] = icmp eq i64 %"iv1'ac.0", 0
; CHECK-NEXT:   br i1 %[[iv1cmp]], label %invertfor.cond1.preheader, label %incinvertfor.body5

; CHECK: incinvertfor.body5:                               ; preds = %invertfor.body5
; CHECK-NEXT:   %[[iv1sub:.+]] = add nsw i64 %"iv1'ac.0", -1
; CHECK-NEXT:   br label %invertif.exit

; CHECK: invertif.end.i.i:                                 ; preds = %invertif.exit
; CHECK-NEXT:   %add.ptr_unwrap = getelementptr inbounds double, double* %w9, i64 %"iv1'ac.0"
; CHECK-NEXT:   %"add.ptr'ipg_unwrap" = getelementptr inbounds double, double* %"w9'", i64 %"iv1'ac.0"
; CHECK-NEXT:   %[[ngep:.+]] = getelementptr inbounds double*, double** %[[_malloccache]], i64 %add.i.i.i_unwrap
; CHECK-NEXT:   %[[loadtape:.+]] = load double*, double** %[[ngep]]
; CHECK-NEXT:   call void @diffesumsq(double* %add.ptr_unwrap, double* %"add.ptr'ipg_unwrap", double %[[nv:.+]], double* %[[loadtape]])
; CHECK-NEXT:   br label %invertfor.body5

; CHECK: invertif.exit:                                    ; preds = %invertfor.cond.cleanup4, %incinvertfor.body5
; CHECK-NEXT:  %"call2.i.i.i'de.1" = phi double [ %"call2.i.i.i'de.2", %invertfor.cond.cleanup4 ], [ %"call2.i.i.i'de.0", %incinvertfor.body5 ]
; CHECK-NEXT:   %"iv1'ac.0" = phi i64 [ 3, %invertfor.cond.cleanup4 ], [ %[[iv1sub]], %incinvertfor.body5 ]
; CHECK-NEXT:   %mul.i.i.i.i.i_unwrap = mul nsw i64 4, %"iv'ac.0"
; CHECK-NEXT:   %add.i.i.i_unwrap = add nsw i64 %mul.i.i.i.i.i_unwrap, %"iv1'ac.0"
; CHECK-NEXT:   %"arrayidx'ipg_unwrap" = getelementptr inbounds double, double* %"w3'", i64 %add.i.i.i_unwrap
; CHECK-NEXT:   %[[lde:.+]] = load double, double* %"arrayidx'ipg_unwrap", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"arrayidx'ipg_unwrap", align 8
; CHECK-NEXT:   %[[fad:.+]] = fadd fast double %"call2.i.i.i'de.1", %[[lde]]
; CHECK-NEXT:   %[[nv]] = select{{( fast)?}} i1 %false, double %"call2.i.i.i'de.1", double %[[fad]]
; CHECK-NEXT:   br i1 %false, label %invertfor.body5, label %invertif.end.i.i

; CHECK: invertfor.cond.cleanup4:                          ; preds = %for.cond.cleanup4, %incinvertfor.cond1.preheader
; CHECK-NEXT:   %"call2.i.i.i'de.2" = phi double [ %"call2.i.i.i'de.0", %incinvertfor.cond1.preheader ], [ 0.000000e+00, %for.cond.cleanup4 ]
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %[[ivsub]], %incinvertfor.cond1.preheader ], [ 3, %for.cond.cleanup4 ]
; CHECK-NEXT:   br label %invertif.exit
; CHECK-NEXT: }

; CHECK: define internal { double*, double } @augmented_sumsq(double* %a3, double* %"a3'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = alloca { double*, double }
; CHECK-NEXT:   %1 = getelementptr inbounds { double*, double }, { double*, double }* %0, i32 0, i32 0
; CHECK-NEXT:   %malloccall = tail call noalias nonnull dereferenceable(32) dereferenceable_or_null(32) i8* @malloc(i64 32)
; CHECK-NEXT:   %a6_malloccache = bitcast i8* %malloccall to double*
; CHECK-NEXT:   store double* %a6_malloccache, double** %1
; CHECK-NEXT:   br label %for.body

; CHECK: for.body:                                         ; preds = %for.body, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
; CHECK-NEXT:   %res.0 = phi double [ 0.000000e+00, %entry ], [ %add.i, %for.body ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %mul.i.i.i = mul nsw i64 4, %iv
; CHECK-NEXT:   %arrayidx.i.i.i = getelementptr inbounds double, double* %a3, i64 %mul.i.i.i
; CHECK-NEXT:   %a6 = load double, double* %arrayidx.i.i.i, align 8, !tbaa !2
; CHECK-NEXT:   %[[gepf:.+]] = getelementptr inbounds double, double* %a6_malloccache, i64 %iv
; CHECK-NEXT:   store double %a6, double* %[[gepf]]
; CHECK-NEXT:   %mul.i.i8 = fmul double %a6, %a6
; CHECK-NEXT:   %add.i = fadd double %res.0, %mul.i.i8
; CHECK-NEXT:   %cmp = icmp ne i64 %iv.next, 4
; CHECK-NEXT:   br i1 %cmp, label %for.body, label %for.cond.cleanup8

; CHECK: for.cond.cleanup8:                                ; preds = %for.body
; CHECK-NEXT:   %[[gepa:.+]] = getelementptr inbounds { double*, double }, { double*, double }* %0, i32 0, i32 1
; CHECK-NEXT:   store double %add.i, double* %[[gepa]]
; CHECK-NEXT:   %[[ret:.+]] = load { double*, double }, { double*, double }* %0
; CHECK-NEXT:   ret { double*, double } %[[ret]]
; CHECK-NEXT: }

; CHECK: define internal void @diffesumsq(double* %a3, double* %"a3'", double %differeturn, double* %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   br label %for.body

; CHECK: for.body:                                         ; preds = %for.body, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %cmp = icmp ne i64 %iv.next, 4
; CHECK-NEXT:   br i1 %cmp, label %for.body, label %invertfor.body

; CHECK: invertentry:                                      ; preds = %invertfor.body
; CHECK-NEXT:   %[[tofree:.+]] = bitcast double* %tapeArg to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[tofree]])
; CHECK-NEXT:   ret void

; CHECK: invertfor.body:                                   ; preds = %for.body, %incinvertfor.body
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %[[subidx:.+]], %incinvertfor.body ], [ 3, %for.body ]
; CHECK-NEXT:   %[[tapeptr:.+]] = getelementptr inbounds double, double* %tapeArg, i64 %"iv'ac.0"
; CHECK-NEXT:   %[[cached:.+]] = load double, double* %[[tapeptr]]
; CHECK-NEXT:   %[[fmul:.+]] = fmul fast double %differeturn, %[[cached]]
; CHECK-NEXT:   %[[dif:.+]] = fadd fast double %[[fmul]], %[[fmul]]
; CHECK-NEXT:   %mul.i.i.i_unwrap = mul nsw i64 4, %"iv'ac.0"
; CHECK-NEXT:   %"arrayidx.i.i.i'ipg_unwrap" = getelementptr inbounds double, double* %"a3'", i64 %mul.i.i.i_unwrap
; CHECK-NEXT:   %[[aidx:.+]] = load double, double* %"arrayidx.i.i.i'ipg_unwrap", align 8
; CHECK-NEXT:   %[[add:.+]] = fadd fast double %[[aidx]], %[[dif]]
; CHECK-NEXT:   store double %[[add]], double* %"arrayidx.i.i.i'ipg_unwrap", align 8
; CHECK-NEXT:   %[[rcmp:.+]] = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %[[rcmp]], label %invertentry, label %incinvertfor.body

; CHECK: incinvertfor.body:                                ; preds = %invertfor.body
; CHECK-NEXT:   %[[subidx]] = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertfor.body
; CHECK-NEXT: }
