; RUN: %opt < %s %loadEnzyme -enzyme -enzyme_preopt=false -mem2reg -early-cse -simplifycfg -correlated-propagation -instcombine -adce -S | FileCheck %s
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

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

declare dso_local double @__enzyme_autodiff(i8*, double*, double*, double*, double*)

define void @caller(double*, double*, double*, double*) {
entry:
  %call = call double @__enzyme_autodiff(i8* bitcast (double (double*, double*)* @_ZL6matvecPKN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEES3_ to i8*), double* %0, double* %1, double* %2, double* %3)
  ret void
}

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
  %sq = fmul double %6, %6
  %add.i.i = fadd double %sq, %7
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


; CHECK: define internal {} @diffe_ZL6matvecPKN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEES3_(double* noalias %W, double* %"W'", double* noalias %M, double* %"M'", double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %call.i.i.i.i.i.i.i = call noalias i8* @malloc(i64 128) #8
; CHECK-NEXT:   %"call.i.i.i.i.i.i.i'mi" = call noalias nonnull i8* @malloc(i64 128) #8
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull align 1 %"call.i.i.i.i.i.i.i'mi", i8 0, i64 128, i1 false)
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
; CHECK-NEXT:   %call.i.i.i.i.i.i.i13 = call noalias i8* @malloc(i64 128) #8
; CHECK-NEXT:   %"call.i.i.i.i.i.i.i13'mi" = call noalias nonnull i8* @malloc(i64 128) #8
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull align 1 %"call.i.i.i.i.i.i.i13'mi", i8 0, i64 128, i1 false)
; CHECK-NEXT:   %"'ipc8" = bitcast i8* %"call.i.i.i.i.i.i.i13'mi" to double*
; CHECK-NEXT:   %3 = bitcast i8* %call.i.i.i.i.i.i.i13 to double*
; CHECK-NEXT:   %_augmented = call { { { double* }* } } @augmented_subfn(double* %3, double* nonnull %"'ipc8", double* nonnull %0, double* nonnull %"'ipc")
; CHECK-NEXT:   br label %for.body.i

; CHECK: for.body.i:                                       ; preds = %for.body.i, %_ZN5Eigen8internal26call_dense_assignment_loopINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_13CwiseBinaryOpINS0_20scalar_difference_opIddEEKS3_S7_EENS0_9assign_opIddEEEEvRT_RKT0_RKT1_.exit
; CHECK-NEXT:   %iv1 = phi i64 [ %iv.next2, %for.body.i ], [ 0, %_ZN5Eigen8internal26call_dense_assignment_loopINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_13CwiseBinaryOpINS0_20scalar_difference_opIddEEKS3_S7_EENS0_9assign_opIddEEEEvRT_RKT0_RKT1_.exit ]
; CHECK-NEXT:   %iv.next2 = add nuw nsw i64 %iv1, 1
; CHECK-NEXT:   %cmp.i = icmp eq i64 %iv.next2, 4
; CHECK-NEXT:   br i1 %cmp.i, label %for.cond10.preheader.i, label %for.body.i

; CHECK: for.cond10.preheader.i:                           ; preds = %for.cond.cleanup13.i, %for.body.i
; CHECK-NEXT:   %iv3 = phi i64 [ %iv.next4, %for.cond.cleanup13.i ], [ 0, %for.body.i ]
; CHECK-NEXT:   %iv.next4 = add nuw nsw i64 %iv3, 1
; CHECK-NEXT:   br label %for.body14.i

; CHECK: for.cond.cleanup13.i:                             ; preds = %for.body14.i
; CHECK-NEXT:   %cmp7.i = icmp eq i64 %iv.next4, 4
; CHECK-NEXT:   br i1 %cmp7.i, label %invertfor.cond.cleanup13.i, label %for.cond10.preheader.i

; CHECK: for.body14.i:                                     ; preds = %for.body14.i, %for.cond10.preheader.i
; CHECK-NEXT:   %iv5 = phi i64 [ %iv.next6, %for.body14.i ], [ 0, %for.cond10.preheader.i ]
; CHECK-NEXT:   %iv.next6 = add nuw nsw i64 %iv5, 1
; CHECK-NEXT:   %cmp12.i = icmp eq i64 %iv.next6, 4
; CHECK-NEXT:   br i1 %cmp12.i, label %for.cond.cleanup13.i, label %for.body14.i

; CHECK: invertentry:                                      ; preds = %invertfor.body.i.i
; CHECK-NEXT:   tail call void @free(i8* nonnull %"call.i.i.i.i.i.i.i'mi")
; CHECK-NEXT:   tail call void @free(i8* %call.i.i.i.i.i.i.i)
; CHECK-NEXT:   ret {} undef

; CHECK: invertfor.body.i.i:                               ; preds = %invert_ZN5Eigen8internal26call_dense_assignment_loopINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_13CwiseBinaryOpINS0_20scalar_difference_opIddEEKS3_S7_EENS0_9assign_opIddEEEEvRT_RKT0_RKT1_.exit, %incinvertfor.body.i.i
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ 15, %invert_ZN5Eigen8internal26call_dense_assignment_loopINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_13CwiseBinaryOpINS0_20scalar_difference_opIddEEKS3_S7_EENS0_9assign_opIddEEEEvRT_RKT0_RKT1_.exit ], [ %10, %incinvertfor.body.i.i ]
; CHECK-NEXT:   %"Oi'ipg_unwrap" = getelementptr inbounds double, double* %"'ipc", i64 %"iv'ac.0"
; CHECK-NEXT:   %4 = load double, double* %"Oi'ipg_unwrap", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"Oi'ipg_unwrap", align 8
; CHECK-NEXT:   %"arrayidx.i2.i.i.i.i'ipg_unwrap" = getelementptr inbounds double, double* %"M'", i64 %"iv'ac.0"
; CHECK-NEXT:   %5 = load double, double* %"arrayidx.i2.i.i.i.i'ipg_unwrap", align 8
; CHECK-NEXT:   %6 = fsub fast double %5, %4
; CHECK-NEXT:   store double %6, double* %"arrayidx.i2.i.i.i.i'ipg_unwrap", align 8
; CHECK-NEXT:   %"arrayidx.i.i.i.i.i'ipg_unwrap" = getelementptr inbounds double, double* %"W'", i64 %"iv'ac.0"
; CHECK-NEXT:   %7 = load double, double* %"arrayidx.i.i.i.i.i'ipg_unwrap", align 8
; CHECK-NEXT:   %8 = fadd fast double %7, %4
; CHECK-NEXT:   store double %8, double* %"arrayidx.i.i.i.i.i'ipg_unwrap", align 8
; CHECK-NEXT:   %9 = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %9, label %invertentry, label %incinvertfor.body.i.i

; CHECK: incinvertfor.body.i.i:                            ; preds = %invertfor.body.i.i
; CHECK-NEXT:   %10 = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertfor.body.i.i

; CHECK: invert_ZN5Eigen8internal26call_dense_assignment_loopINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_13CwiseBinaryOpINS0_20scalar_difference_opIddEEKS3_S7_EENS0_9assign_opIddEEEEvRT_RKT0_RKT1_.exit: ; preds = %invertfor.body.i
; CHECK-NEXT:   %_unwrap11 = extractvalue { { { double* }* } } %_augmented, 0
; CHECK-NEXT:   %11 = call {} @diffesubfn(double* %3, double* nonnull %"'ipc8", double* %0, double* nonnull %"'ipc", { { double* }* } %_unwrap11)
; CHECK-NEXT:   tail call void @free(i8* nonnull %"call.i.i.i.i.i.i.i13'mi")
; CHECK-NEXT:   tail call void @free(i8* %call.i.i.i.i.i.i.i13)
; CHECK-NEXT:   br label %invertfor.body.i.i

; CHECK: invertfor.body.i:                                 ; preds = %invertfor.cond10.preheader.i, %incinvertfor.body.i
; CHECK-NEXT:   %"iv1'ac.0" = phi i64 [ %13, %incinvertfor.body.i ], [ 3, %invertfor.cond10.preheader.i ]
; CHECK-NEXT:   %12 = icmp eq i64 %"iv1'ac.0", 0
; CHECK-NEXT:   br i1 %12, label %invert_ZN5Eigen8internal26call_dense_assignment_loopINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_13CwiseBinaryOpINS0_20scalar_difference_opIddEEKS3_S7_EENS0_9assign_opIddEEEEvRT_RKT0_RKT1_.exit, label %incinvertfor.body.i

; CHECK: incinvertfor.body.i:                              ; preds = %invertfor.body.i
; CHECK-NEXT:   %13 = add nsw i64 %"iv1'ac.0", -1
; CHECK-NEXT:   br label %invertfor.body.i

; CHECK: invertfor.cond10.preheader.i:                     ; preds = %invertfor.body14.i
; CHECK-NEXT:   %14 = icmp eq i64 %"iv3'ac.0", 0
; CHECK-NEXT:   br i1 %14, label %invertfor.body.i, label %incinvertfor.cond10.preheader.i

; CHECK: incinvertfor.cond10.preheader.i:                  ; preds = %invertfor.cond10.preheader.i
; CHECK-NEXT:   %15 = add nsw i64 %"iv3'ac.0", -1
; CHECK-NEXT:   br label %invertfor.cond.cleanup13.i

; CHECK: invertfor.cond.cleanup13.i:                       ; preds = %for.cond.cleanup13.i, %incinvertfor.cond10.preheader.i
; CHECK-NEXT:   %"add.i.i.lcssa'de.0" = phi double [ 0.000000e+00, %incinvertfor.cond10.preheader.i ], [ %differeturn, %for.cond.cleanup13.i ]
; CHECK-NEXT:   %".lcssa'de.0" = phi i64 [ %22, %incinvertfor.cond10.preheader.i ], [ 0, %for.cond.cleanup13.i ]
; CHECK-NEXT:   %"iv3'ac.0" = phi i64 [ %15, %incinvertfor.cond10.preheader.i ], [ 3, %for.cond.cleanup13.i ]
; CHECK-NEXT:   br label %invertfor.body14.i

; CHECK: invertfor.body14.i:                               ; preds = %incinvertfor.body14.i, %invertfor.cond.cleanup13.i
; CHECK-NEXT:   %"add.i.i'de.1" = phi double [ %"add.i.i.lcssa'de.0", %invertfor.cond.cleanup13.i ], [ 0.000000e+00, %incinvertfor.body14.i ]
; CHECK-NEXT:   %"'de12.1" = phi i64 [ %".lcssa'de.0", %invertfor.cond.cleanup13.i ], [ %23, %incinvertfor.body14.i ]
; CHECK-NEXT:   %"iv5'ac.0" = phi i64 [ 3, %invertfor.cond.cleanup13.i ], [ %24, %incinvertfor.body14.i ]
; CHECK-NEXT:   %16 = bitcast i64 %"'de12.1" to double
; CHECK-NEXT:   %17 = fadd fast double %"add.i.i'de.1", %16
; CHECK-NEXT:   %mul.i.i_unwrap = shl nsw i64 %"iv3'ac.0", 2
; CHECK-NEXT:   %add.i4.i_unwrap = add nsw i64 %mul.i.i_unwrap, %"iv5'ac.0"
; CHECK-NEXT:   %"arrayidx.i.i'ipg_unwrap" = getelementptr inbounds double, double* %"'ipc8", i64 %add.i4.i_unwrap
; CHECK-NEXT:   %18 = load double, double* %"arrayidx.i.i'ipg_unwrap", align 8
; CHECK-NEXT:   %19 = fadd fast double %18, %17
; CHECK-NEXT:   store double %19, double* %"arrayidx.i.i'ipg_unwrap", align 8
; CHECK-NEXT:   %20 = icmp eq i64 %"iv5'ac.0", 0
; CHECK-NEXT:   %21 = bitcast double %17 to i64
; CHECK-NEXT:   %22 = select i1 %20, i64 %21, i64 0
; CHECK-NEXT:   br i1 %20, label %invertfor.cond10.preheader.i, label %incinvertfor.body14.i

; CHECK: incinvertfor.body14.i:                            ; preds = %invertfor.body14.i
; CHECK-NEXT:   %23 = bitcast double %17 to i64
; CHECK-NEXT:   %24 = add nsw i64 %"iv5'ac.0", -1
; CHECK-NEXT:   br label %invertfor.body14.i
; CHECK-NEXT: }

; CHECK: define internal { { double* }, double } @augmented_sumsq(double* %a3, double* %"a3'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = alloca { { double* }, double }, align 8
; CHECK-NEXT:   %malloccall = tail call noalias nonnull i8* @malloc(i64 32)
; CHECK-NEXT:   %a6_malloccache = bitcast i8* %malloccall to double*
; CHECK-NEXT:   %1 = bitcast { { double* }, double }* %0 to i8**
; CHECK-NEXT:   store i8* %malloccall, i8** %1, align 8
; CHECK-NEXT:   br label %for.body

; CHECK: for.body:                                         ; preds = %for.body, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
; CHECK-NEXT:   %res.0 = phi double [ %add.i, %for.body ], [ 0.000000e+00, %entry ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %mul.i.i.i = shl nsw i64 %iv, 2
; CHECK-NEXT:   %arrayidx.i.i.i = getelementptr inbounds double, double* %a3, i64 %mul.i.i.i
; CHECK-NEXT:   %a6 = load double, double* %arrayidx.i.i.i, align 8, !tbaa !2
; CHECK-NEXT:   %2 = getelementptr inbounds double, double* %a6_malloccache, i64 %iv
; CHECK-NEXT:   store double %a6, double* %2, align 8, !invariant.group !14
; CHECK-NEXT:   %mul.i.i8 = fmul double %a6, %a6
; CHECK-NEXT:   %add.i = fadd double %res.0, %mul.i.i8
; CHECK-NEXT:   %cmp = icmp eq i64 %iv.next, 4
; CHECK-NEXT:   br i1 %cmp, label %for.cond.cleanup8, label %for.body

; CHECK: for.cond.cleanup8:                                ; preds = %for.body
; CHECK-NEXT:   %3 = getelementptr inbounds { { double* }, double }, { { double* }, double }* %0, i64 0, i32 1
; CHECK-NEXT:   store double %add.i, double* %3, align 8
; CHECK-NEXT:   %4 = getelementptr inbounds { { double* }, double }, { { double* }, double }* %0, i64 0, i32 0, i32 0
; CHECK-NEXT:   %.unpack.unpack = load double*, double** %4, align 8
; CHECK-NEXT:   %.unpack3 = insertvalue { double* } undef, double* %.unpack.unpack, 0
; CHECK-NEXT:   %5 = insertvalue { { double* }, double } undef, { double* } %.unpack3, 0
; CHECK-NEXT:   %6 = insertvalue { { double* }, double } %5, double %add.i, 1
; CHECK-NEXT:   ret { { double* }, double } %6
; CHECK-NEXT: }

; CHECK: define internal { { { double* }* } } @augmented_subfn(double* %w3, double* %"w3'", double* %w9, double* %"w9'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = alloca { { { double* }* } }, align 8
; CHECK-NEXT:   %false = call i1 @falser() #8
; CHECK-NEXT:   %malloccall = tail call noalias nonnull i8* @malloc(i64 128)
; CHECK-NEXT:   %subcache_malloccache = bitcast i8* %malloccall to { double* }*
; CHECK-NEXT:   %1 = bitcast { { { double* }* } }* %0 to i8**
; CHECK-NEXT:   store i8* %malloccall, i8** %1, align 8
; CHECK-NEXT:   br label %for.cond1.preheader

; CHECK: for.cond1.preheader:                              ; preds = %for.cond.cleanup4, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %for.cond.cleanup4 ], [ 0, %entry ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %mul.i.i.i.i.i = shl nsw i64 %iv, 2
; CHECK-NEXT:   br label %for.body5

; CHECK: for.body5:                                        ; preds = %if.exit, %for.cond1.preheader
; CHECK-NEXT:   %iv1 = phi i64 [ %iv.next2, %if.exit ], [ 0, %for.cond1.preheader ]
; CHECK-NEXT:   %iv.next2 = add nuw nsw i64 %iv1, 1
; CHECK-NEXT:   %add.i.i.i = add nuw nsw i64 %mul.i.i.i.i.i, %iv1
; CHECK-NEXT:   %arrayidx = getelementptr inbounds double, double* %w3, i64 %add.i.i.i
; CHECK-NEXT:   br i1 %false, label %if.exit, label %if.end.i.i

; CHECK: if.end.i.i:                                       ; preds = %for.body5
; CHECK-NEXT:   %add.ptr = getelementptr inbounds double, double* %w9, i64 %iv1
; CHECK-NEXT:   %"add.ptr'ipg" = getelementptr inbounds double, double* %"w9'", i64 %iv1
; CHECK-NEXT:   %call2.i.i.i_augmented = call { { double* }, double } @augmented_sumsq(double* %add.ptr, double* %"add.ptr'ipg")
; CHECK-NEXT:   %subcache = extractvalue { { double* }, double } %call2.i.i.i_augmented, 0
; CHECK-NEXT:   %2 = shl nuw nsw i64 %iv1, 2
; CHECK-NEXT:   %3 = add nuw nsw i64 %iv, %2
; CHECK-NEXT:   %4 = extractvalue { double* } %subcache, 0
; CHECK-NEXT:   %5 = getelementptr inbounds { double* }, { double* }* %subcache_malloccache, i64 %3, i32 0
; CHECK-NEXT:   store double* %4, double** %5, align 8
; CHECK-NEXT:   %call2.i.i.i = extractvalue { { double* }, double } %call2.i.i.i_augmented, 1
; CHECK-NEXT:   br label %if.exit

; CHECK: if.exit:                                          ; preds = %if.end.i.i, %for.body5
; CHECK-NEXT:   %retval = phi double [ %call2.i.i.i, %if.end.i.i ], [ 0.000000e+00, %for.body5 ]
; CHECK-NEXT:   store double %retval, double* %arrayidx, align 8, !tbaa !2
; CHECK-NEXT:   %cmp3 = icmp eq i64 %iv.next2, 4
; CHECK-NEXT:   br i1 %cmp3, label %for.cond.cleanup4, label %for.body5

; CHECK: for.cond.cleanup4:                                ; preds = %if.exit
; CHECK-NEXT:   %cmp = icmp eq i64 %iv.next, 4
; CHECK-NEXT:   br i1 %cmp, label %for.cond.cleanup, label %for.cond1.preheader

; CHECK: for.cond.cleanup:                                 ; preds = %for.cond.cleanup4
; CHECK-NEXT:   %6 = getelementptr inbounds { { { double* }* } }, { { { double* }* } }* %0, i64 0, i32 0, i32 0
; CHECK-NEXT:   %.unpack.unpack = load { double* }*, { double* }** %6, align 8
; CHECK-NEXT:   %.unpack1 = insertvalue { { double* }* } undef, { double* }* %.unpack.unpack, 0
; CHECK-NEXT:   %7 = insertvalue { { { double* }* } } undef, { { double* }* } %.unpack1, 0
; CHECK-NEXT:   ret { { { double* }* } } %7
; CHECK-NEXT: }

; CHECK: define internal {} @diffesubfn(double* %w3, double* %"w3'", double* %w9, double* %"w9'", { { double* }* } %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = extractvalue { { double* }* } %tapeArg, 0
; CHECK-NEXT:   %false = call i1 @falser() #8
; CHECK-NEXT:   br label %for.cond1.preheader

; CHECK: for.cond1.preheader:                              ; preds = %for.cond.cleanup4, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %for.cond.cleanup4 ], [ 0, %entry ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   br label %for.body5

; CHECK: for.body5:                                        ; preds = %if.exit, %for.cond1.preheader
; CHECK-NEXT:   %iv1 = phi i64 [ %iv.next2, %if.exit ], [ 0, %for.cond1.preheader ]
; CHECK-NEXT:   %iv.next2 = add nuw nsw i64 %iv1, 1
; CHECK-NEXT:   br label %if.exit

; CHECK: if.end.i.i:                                       ; No predecessors!
; CHECK-NEXT:   br label %if.exit

; CHECK: if.exit:                                          ; preds = %for.body5, %if.end.i.i
; CHECK-NEXT:   %cmp3 = icmp eq i64 %iv.next2, 4
; CHECK-NEXT:   br i1 %cmp3, label %for.cond.cleanup4, label %for.body5

; CHECK: for.cond.cleanup4:                                ; preds = %if.exit
; CHECK-NEXT:   %cmp = icmp eq i64 %iv.next, 4
; CHECK-NEXT:   br i1 %cmp, label %invertfor.cond.cleanup4, label %for.cond1.preheader

; CHECK: invertentry:                                      ; preds = %invertfor.cond1.preheader
; CHECK-NEXT:   %1 = bitcast { double* }* %0 to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %1)
; CHECK-NEXT:   ret {} undef

; CHECK: invertfor.cond1.preheader:                        ; preds = %invertfor.body5
; CHECK-NEXT:   %2 = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %2, label %invertentry, label %incinvertfor.cond1.preheader

; CHECK: incinvertfor.cond1.preheader:                     ; preds = %invertfor.cond1.preheader
; CHECK-NEXT:   %3 = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertfor.cond.cleanup4

; CHECK: invertfor.body5:                                  ; preds = %invertif.exit, %invertif.end.i.i
; CHECK-NEXT:   %4 = icmp eq i64 %"iv1'ac.0", 0
; CHECK-NEXT:   br i1 %4, label %invertfor.cond1.preheader, label %incinvertfor.body5

; CHECK: incinvertfor.body5:                               ; preds = %invertfor.body5
; CHECK-NEXT:   %5 = add nsw i64 %"iv1'ac.0", -1
; CHECK-NEXT:   br label %invertif.exit

; CHECK: invertif.end.i.i:                                 ; preds = %invertif.exit
; CHECK-NEXT:   %add.ptr_unwrap = getelementptr inbounds double, double* %w9, i64 %"iv1'ac.0"
; CHECK-NEXT:   %"add.ptr'ipg_unwrap" = getelementptr inbounds double, double* %"w9'", i64 %"iv1'ac.0"
; CHECK-NEXT:   %_unwrap3 = shl nuw nsw i64 %"iv1'ac.0", 2
; CHECK-NEXT:   %_unwrap4 = add nuw nsw i64 %"iv'ac.0", %_unwrap3
; CHECK-NEXT:   %6 = getelementptr inbounds { double* }, { double* }* %0, i64 %_unwrap4, i32 0
; CHECK-NEXT:   %_unwrap6.unpack = load double*, double** %6, align 8
; CHECK-NEXT:   %_unwrap67 = insertvalue { double* } undef, double* %_unwrap6.unpack, 0
; CHECK-NEXT:   %7 = call {} @diffesumsq(double* %add.ptr_unwrap, double* %"add.ptr'ipg_unwrap", double %8, { double* } %_unwrap67)
; CHECK-NEXT:   br label %invertfor.body5

; CHECK: invertif.exit:                                    ; preds = %invertfor.cond.cleanup4, %incinvertfor.body5
; CHECK-NEXT:   %"iv1'ac.0" = phi i64 [ 3, %invertfor.cond.cleanup4 ], [ %5, %incinvertfor.body5 ]
; CHECK-NEXT:   %mul.i.i.i.i.i_unwrap = shl nsw i64 %"iv'ac.0", 2
; CHECK-NEXT:   %add.i.i.i_unwrap = add nsw i64 %mul.i.i.i.i.i_unwrap, %"iv1'ac.0"
; CHECK-NEXT:   %"arrayidx'ipg_unwrap" = getelementptr inbounds double, double* %"w3'", i64 %add.i.i.i_unwrap
; CHECK-NEXT:   %8 = load double, double* %"arrayidx'ipg_unwrap", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"arrayidx'ipg_unwrap", align 8
; CHECK-NEXT:   br i1 %false, label %invertfor.body5, label %invertif.end.i.i

; CHECK: invertfor.cond.cleanup4:                          ; preds = %for.cond.cleanup4, %incinvertfor.cond1.preheader
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %3, %incinvertfor.cond1.preheader ], [ 3, %for.cond.cleanup4 ]
; CHECK-NEXT:   br label %invertif.exit
; CHECK-NEXT: }

; CHECK: define internal {} @diffesumsq(double* %a3, double* %"a3'", double %differeturn, { double* } %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = extractvalue { double* } %tapeArg, 0
; CHECK-NEXT:   br label %for.body

; CHECK: for.body:                                         ; preds = %for.body, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %cmp = icmp eq i64 %iv.next, 4
; CHECK-NEXT:   br i1 %cmp, label %invertfor.body, label %for.body

; CHECK: invertentry:                                      ; preds = %invertfor.body
; CHECK-NEXT:   %1 = bitcast double* %0 to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %1)
; CHECK-NEXT:   ret {} undef

; CHECK: invertfor.body:                                   ; preds = %for.body, %incinvertfor.body
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %9, %incinvertfor.body ], [ 3, %for.body ]
; CHECK-NEXT:   %2 = getelementptr inbounds double, double* %0, i64 %"iv'ac.0"
; CHECK-NEXT:   %3 = load double, double* %2, align 8, !invariant.group !15, !enzyme_fromcache !16
; CHECK-NEXT:   %4 = fadd fast double %differeturn, %differeturn
; CHECK-NEXT:   %5 = fmul fast double %3, %4
; CHECK-NEXT:   %mul.i.i.i_unwrap = shl nsw i64 %"iv'ac.0", 2
; CHECK-NEXT:   %"arrayidx.i.i.i'ipg_unwrap" = getelementptr inbounds double, double* %"a3'", i64 %mul.i.i.i_unwrap
; CHECK-NEXT:   %6 = load double, double* %"arrayidx.i.i.i'ipg_unwrap", align 8
; CHECK-NEXT:   %7 = fadd fast double %6, %5
; CHECK-NEXT:   store double %7, double* %"arrayidx.i.i.i'ipg_unwrap", align 8
; CHECK-NEXT:   %8 = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %8, label %invertentry, label %incinvertfor.body

; CHECK: incinvertfor.body:                                ; preds = %invertfor.body
; CHECK-NEXT:   %9 = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertfor.body
; CHECK-NEXT: }
