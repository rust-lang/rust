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


; CHECK: define internal void @diffe_ZL6matvecPKN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEES3_(double* noalias %W, double* %"W'", double* noalias %M, double* %"M'", double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %call.i.i.i.i.i.i.i = call noalias nonnull dereferenceable(128) dereferenceable_or_null(128) i8* @malloc(i64 128)
; CHECK-NEXT:   %"call.i.i.i.i.i.i.i'mi" = call noalias nonnull dereferenceable(128) dereferenceable_or_null(128) i8* @malloc(i64 128)
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull dereferenceable(128) dereferenceable_or_null(128) %"call.i.i.i.i.i.i.i'mi", i8 0, i64 128, i1 false)
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
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull dereferenceable(128) dereferenceable_or_null(128) %"call.i.i.i.i.i.i.i13'mi", i8 0, i64 128, i1 false)
; CHECK-NEXT:   %[[ipc8:.+]] = bitcast i8* %"call.i.i.i.i.i.i.i13'mi" to double*
; CHECK-NEXT:   %3 = bitcast i8* %call.i.i.i.i.i.i.i13 to double*
; CHECK-NEXT:   %_augmented = call double** @augmented_subfn(double* nonnull %3, double* nonnull %[[ipc8]], double* nonnull %0, double* nonnull %"'ipc")
; CHECK-NEXT:   br label %for.body.i

; CHECK: for.body.i:                                       ; preds = %for.body.i, %_ZN5Eigen8internal26call_dense_assignment_loopINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_13CwiseBinaryOpINS0_20scalar_difference_opIddEEKS3_S7_EENS0_9assign_opIddEEEEvRT_RKT0_RKT1_.exit
; CHECK-NEXT:   %iv1 = phi i64 [ %iv.next2, %for.body.i ], [ 0, %_ZN5Eigen8internal26call_dense_assignment_loopINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_13CwiseBinaryOpINS0_20scalar_difference_opIddEEKS3_S7_EENS0_9assign_opIddEEEEvRT_RKT0_RKT1_.exit ]
; CHECK-NEXT:   %iv.next2 = add nuw nsw i64 %iv1, 1
; CHECK-NEXT:   %cmp.i = icmp ne i64 %iv.next2, 4
; CHECK-NEXT:   br i1 %cmp.i, label %for.body.i, label %for.cond10.preheader.i.preheader

; CHECK:  for.cond10.preheader.i.preheader:                 ; preds = %for.body.i
; CHECK-NEXT:    %malloccall = tail call noalias nonnull dereferenceable(128) dereferenceable_or_null(128) i8* @malloc(i64 128)
; CHECK-NEXT:    %[[malloccache:.+]] = bitcast i8* %malloccall to double*
; CHECK-NEXT:    br label %for.cond10.preheader.i

; CHECK:  for.cond10.preheader.i:                           ; preds = %for.cond.cleanup13.i, %for.cond10.preheader.i.preheader
; CHECK-NEXT:    %iv3 = phi i64 [ %iv.next4, %for.cond.cleanup13.i ], [ 0, %for.cond10.preheader.i.preheader ] 
; CHECK-NEXT:    %res.i.sroa.0.1 = phi i64 [ %[[res:.+]], %for.cond.cleanup13.i ], [ 0, %for.cond10.preheader.i.preheader ]
; CHECK-NEXT:    %iv.next4 = add nuw nsw i64 %iv3, 1
; CHECK-NEXT:    %mul.i.i = mul nsw i64 4, %iv3
; CHECK-NEXT:    br label %for.body14.i

; CHECK: for.cond.cleanup13.i:                             ; preds = %for.body14.i
; CHECK-NEXT:   %cmp7.i = icmp ne i64 %iv.next4, 4
; CHECK-NEXT:   br i1 %cmp7.i, label %for.cond10.preheader.i, label %invertfor.cond.cleanup13.i

; CHECK: for.body14.i:                                     ; preds = %for.body14.i, %for.cond10.preheader.i
; CHECK-NEXT:   %iv5 = phi i64 [ %iv.next6, %for.body14.i ], [ 0, %for.cond10.preheader.i ]
; CHECK-NEXT:   %res.i.sroa.0.2 = phi i64 [ %res.i.sroa.0.1, %for.cond10.preheader.i ], [ %[[res]], %for.body14.i ]
; CHECK-NEXT:   %iv.next6 = add nuw nsw i64 %iv5, 1
; CHECK-NEXT:   %[[iadd:.+]] = add nsw i64 %mul.i.i, %iv5
; CHECK-NEXT:   %arrayidx.i.i = getelementptr inbounds double, double* %3, i64 %[[iadd]]
; CHECK-NEXT:   %[[aidx:.+]] = bitcast double* %arrayidx.i.i to i64*
; CHECK-NEXT:   %[[prev:.+]] = load i64, i64* %[[aidx]], align 8, !tbaa !2
; CHECK-NEXT:   %[[bc:.+]] = bitcast i64 %res.i.sroa.0.2 to double
; CHECK-NEXT:   %[[igep:.+]] = getelementptr inbounds double, double* %[[malloccache]], i64 %[[iadd]]
; CHECK-NEXT:   store double %[[bc]], double* %[[igep]], align 8
; CHECK-NEXT:   %[[prevbc:.+]] = bitcast i64 %[[prev]] to double
; CHECK-NEXT:   %sq = fmul double %[[bc]], %[[bc]]
; CHECK-NEXT:   %add.i.i = fadd double %sq, %[[prevbc]]
; CHECK-NEXT:   %[[res]] = bitcast double %add.i.i to i64
; CHECK-NEXT:   %cmp12.i = icmp ne i64 %iv.next6, 4
; CHECK-NEXT:   br i1 %cmp12.i, label %for.body14.i, label %for.cond.cleanup13.i

; CHECK: invertentry:                                      ; preds = %invertfor.body.i.i
; CHECK-NEXT:   tail call void @free(i8* nonnull %"call.i.i.i.i.i.i.i'mi")
; CHECK-NEXT:   tail call void @free(i8* nonnull %call.i.i.i.i.i.i.i)
; CHECK-NEXT:   ret void

; CHECK: invertfor.body.i.i:                               ; preds = %invert_ZN5Eigen8internal26call_dense_assignment_loopINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_13CwiseBinaryOpINS0_20scalar_difference_opIddEEKS3_S7_EENS0_9assign_opIddEEEEvRT_RKT0_RKT1_.exit, %incinvertfor.body.i.i
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ 15, %invert_ZN5Eigen8internal26call_dense_assignment_loopINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_13CwiseBinaryOpINS0_20scalar_difference_opIddEEKS3_S7_EENS0_9assign_opIddEEEEvRT_RKT0_RKT1_.exit ], [ %[[ivsub:.+]], %incinvertfor.body.i.i ]
; CHECK-NEXT:   %"Oi'ipg_unwrap" = getelementptr inbounds double, double* %"'ipc", i64 %"iv'ac.0"
; CHECK-NEXT:   %[[pOi:.+]] = load double, double* %"Oi'ipg_unwrap", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"Oi'ipg_unwrap", align 8
; CHECK-NEXT:   %[[nOi:.+]] = {{(fsub fast double 0.000000e\+00,|fneg fast double)}} %[[pOi]]
; CHECK-NEXT:   %"arrayidx.i2.i.i.i.i'ipg_unwrap" = getelementptr inbounds double, double* %"M'", i64 %"iv'ac.0"
; CHECK-NEXT:   %[[pai:.+]] = load double, double* %"arrayidx.i2.i.i.i.i'ipg_unwrap", align 8
; CHECK-NEXT:   %[[postai:.+]] = fadd fast double %[[pai]], %[[nOi]]
; CHECK-NEXT:   store double %[[postai:.+]], double* %"arrayidx.i2.i.i.i.i'ipg_unwrap", align 8
; CHECK-NEXT:   %"arrayidx.i.i.i.i.i'ipg_unwrap" = getelementptr inbounds double, double* %"W'", i64 %"iv'ac.0"
; CHECK-NEXT:   %[[zai:.+]] = load double, double* %"arrayidx.i.i.i.i.i'ipg_unwrap", align 8
; CHECK-NEXT:   %[[rai:.+]] = fadd fast double %[[zai]], %[[pOi]]
; CHECK-NEXT:   store double %[[rai]], double* %"arrayidx.i.i.i.i.i'ipg_unwrap", align 8
; CHECK-NEXT:   %[[ivcmp:.+]] = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %[[ivcmp]], label %invertentry, label %incinvertfor.body.i.i

; CHECK: incinvertfor.body.i.i:                            ; preds = %invertfor.body.i.i
; CHECK-NEXT:   %[[ivsub]] = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertfor.body.i.i

; CHECK: invert_ZN5Eigen8internal26call_dense_assignment_loopINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_13CwiseBinaryOpINS0_20scalar_difference_opIddEEKS3_S7_EENS0_9assign_opIddEEEEvRT_RKT0_RKT1_.exit: ; preds = %invertfor.body.i
; CHECK-NEXT:   call void @diffesubfn(double* nonnull %3, double* nonnull %[[ipc8]], double* nonnull %0, double* nonnull %"'ipc", double** %_augmented)
; CHECK-NEXT:   tail call void @free(i8* nonnull %"call.i.i.i.i.i.i.i13'mi")
; CHECK-NEXT:   tail call void @free(i8* nonnull %call.i.i.i.i.i.i.i13)
; CHECK-NEXT:   br label %invertfor.body.i.i

; CHECK: invertfor.body.i:                                 ; preds = %invertfor.cond10.preheader.i.preheader, %incinvertfor.body.i
; CHECK-NEXT:   %"iv1'ac.0" = phi i64 [ 3, %invertfor.cond10.preheader.i.preheader ], [ %[[iv1sub:.+]], %incinvertfor.body.i ]
; CHECK-NEXT:   %[[iv1cmp:.+]] = icmp eq i64 %"iv1'ac.0", 0
; CHECK-NEXT:   br i1 %[[iv1cmp]], label %invert_ZN5Eigen8internal26call_dense_assignment_loopINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEENS_13CwiseBinaryOpINS0_20scalar_difference_opIddEEKS3_S7_EENS0_9assign_opIddEEEEvRT_RKT0_RKT1_.exit, label %incinvertfor.body.i

; CHECK: incinvertfor.body.i:                              ; preds = %invertfor.body.i
; CHECK-NEXT:   %[[iv1sub]] = add nsw i64 %"iv1'ac.0", -1
; CHECK-NEXT:   br label %invertfor.body.i

; CHECK: invertfor.cond10.preheader.i.preheader:           ; preds = %invertfor.cond10.preheader.i
; CHECK-NEXT:   tail call void @free(i8* nonnull %malloccall)
; CHECK-NEXT:   br label %invertfor.body.i

; CHECK: invertfor.cond10.preheader.i:                     ; preds = %invertfor.body14.i
; CHECK-NEXT:   %[[iv3cmp:.+]] = icmp eq i64 %"iv3'ac.0", 0
; CHECK-NEXT:   %[[g15:.+]] = bitcast i64 %[[g29:.+]] to double
; CHECK-NEXT:   %[[g16:.+]] = bitcast i64 %[[sel:.+]] to double
; CHECK-NEXT:   %[[g17:.+]] = fadd fast double %[[g15]], %[[g16]]
; CHECK-NEXT:   %[[g18:.+]] = bitcast double %[[g17]] to i64
; CHECK-NEXT:   br i1 %[[iv3cmp]], label %invertfor.cond10.preheader.i.preheader, label %incinvertfor.cond10.preheader.i

; CHECK: incinvertfor.cond10.preheader.i:                  ; preds = %invertfor.cond10.preheader.i
; CHECK-NEXT:   %[[iv3sub:.+]] = add nsw i64 %"iv3'ac.0", -1
; CHECK-NEXT:   br label %invertfor.cond.cleanup13.i

; CHECK: invertfor.cond.cleanup13.i:                       ; preds = %for.cond.cleanup13.i, %incinvertfor.cond10.preheader.i
; CHECK-NEXT:   %[[lcssade:.+]] = phi i64 [ %[[g18]], %incinvertfor.cond10.preheader.i ], [ 0, %for.cond.cleanup13.i ]
; CHECK-NEXT:   %[[addlcssa:.+]] = phi double [ 0.000000e+00, %incinvertfor.cond10.preheader.i ], [ %differeturn, %for.cond.cleanup13.i ]
; CHECK-NEXT:   %"iv3'ac.0" = phi i64 [ %[[iv3sub]], %incinvertfor.cond10.preheader.i ], [ 3, %for.cond.cleanup13.i ]
; CHECK-NEXT:   br label %invertfor.body14.i

; CHECK: invertfor.body14.i:                               ; preds = %incinvertfor.body14.i, %invertfor.cond.cleanup13.i
; CHECK-NEXT:   %[[de11:.+]] = phi i64 [ %[[lcssade]], %invertfor.cond.cleanup13.i ], [ %[[dessa:.+]], %incinvertfor.body14.i ]
; CHECK-NEXT:   %[[addiide1:.+]] = phi double [ %[[addlcssa]], %invertfor.cond.cleanup13.i ], [ 0.000000e+00, %incinvertfor.body14.i ]
; CHECK-NEXT:   %"iv5'ac.0" = phi i64 [ 3, %invertfor.cond.cleanup13.i ], [ %[[iv5sub:.+]], %incinvertfor.body14.i ]
; CHECK-NEXT:   %[[dedd:.+]] = bitcast i64 %[[de11:.+]] to double
; CHECK-NEXT:   %[[fad:.+]] = fadd fast double %"add.i.i'de.1", %[[dedd]]
; CHECK-NEXT:   %[[iv54:.+]] = mul {{(nuw )?}}nsw i64 %"iv3'ac.0", 4
; CHECK-NEXT:   %[[iv35a:.+]] = add {{(nuw )?}}nsw i64 %"iv5'ac.0", %[[iv54]]
; CHECK-NEXT:   %[[bcq:.+]] = getelementptr inbounds double, double* %[[malloccache]], i64 %[[iv35a]]
; CHECK-NEXT:   %[[unwrap13:.+]] = load double, double* %[[bcq]], align 8, !invariant.group !
; CHECK-NEXT:   %m0diffe = fmul fast double %[[fad]], %[[unwrap13]]
; CHECK-NEXT:   %[[m2a:.+]] = fadd fast double %m0diffe, %m0diffe
; CHECK-NEXT:   %[[dessa:.+]] = bitcast double %[[m2a]] to i64
; CHECK-NEXT:   %"arrayidx.i.i'ipg_unwrap" = getelementptr inbounds double, double* %[[ipc8]], i64 %[[iv35a]]
; CHECK-NEXT:   %[[ddpc:.+]] = load double, double* %"arrayidx.i.i'ipg_unwrap", align 8
; CHECK-NEXT:   %[[ddpost:.+]] = fadd fast double %[[ddpc]], %[[fad]]
; CHECK-NEXT:   store double %[[ddpost]], double* %"arrayidx.i.i'ipg_unwrap", align 8
; CHECK-NEXT:   %[[iv5cmp:.+]] = icmp eq i64 %"iv5'ac.0", 0
; CHECK-NEXT:   %[[nivcmp:.+]] = xor i1 %[[iv5cmp]], true
; CHECK-NEXT:   %[[g29:.+]] = select i1 %[[nivcmp]], i64 %[[dessa]], i64 0
; CHECK-NEXT:   %[[selo2:.+]] = bitcast double %[[m2a]] to i64
; CHECK-NEXT:   %[[sel]] = select{{( fast)?}} i1 %[[iv5cmp]], i64 %[[selo2]], i64 0
; CHECK-NEXT:   br i1 %[[iv5cmp]], label %invertfor.cond10.preheader.i, label %incinvertfor.body14.i

; CHECK: incinvertfor.body14.i:                            ; preds = %invertfor.body14.i
; CHECK-NEXT:   %[[iv5sub]] = add nsw i64 %"iv5'ac.0", -1
; CHECK-NEXT:   br label %invertfor.body14.i
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
; CHECK-NEXT:   %[[a6gep:.+]] = getelementptr inbounds double, double* %a6_malloccache, i64 %iv
; CHECK-NEXT:   store double %a6, double* %[[a6gep]], align 8, !tbaa !2, !invariant.group ![[iga6:[0-9]+]]
; CHECK-NEXT:   %mul.i.i8 = fmul double %a6, %a6
; CHECK-NEXT:   %add.i = fadd double %res.0, %mul.i.i8
; CHECK-NEXT:   %cmp = icmp ne i64 %iv.next, 4
; CHECK-NEXT:   br i1 %cmp, label %for.body, label %for.cond.cleanup8

; CHECK: for.cond.cleanup8:                                ; preds = %for.body
; CHECK-NEXT:   %[[resp:.+]] = getelementptr inbounds { double*, double }, { double*, double }* %0, i32 0, i32 1
; CHECK-NEXT:   store double %add.i, double* %[[resp]]
; CHECK-NEXT:   %[[out:.+]] = load { double*, double }, { double*, double }* %0
; CHECK-NEXT:   ret { double*, double } %[[out]]
; CHECK-NEXT: }

; CHECK: define internal double** @augmented_subfn(double* %w3, double* %"w3'", double* %w9, double* %"w9'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %false = call i1 @falser()
; CHECK-NEXT:   %malloccall = tail call noalias nonnull dereferenceable(128) dereferenceable_or_null(128) i8* @malloc(i64 128)
; CHECK-NEXT:   %subcache_malloccache = bitcast i8* %malloccall to double**
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
; CHECK-NEXT:   %subcache = extractvalue { double*, double } %call2.i.i.i_augmented, 0
; CHECK-NEXT:   %[[loc:.+]] = getelementptr inbounds double*, double** %subcache_malloccache, i64 %add.i.i.i
; CHECK-NEXT:   store double* %subcache, double** %[[loc]], align 8
; CHECK-NEXT:   %call2.i.i.i = extractvalue { double*, double } %call2.i.i.i_augmented, 1
; CHECK-NEXT:   br label %if.exit

; CHECK: if.exit:                                          ; preds = %if.end.i.i, %for.body5
; CHECK-NEXT:   %retval = phi double [ %call2.i.i.i, %if.end.i.i ], [ 0.000000e+00, %for.body5 ]
; CHECK-NEXT:   store double %retval, double* %arrayidx, align 8, !tbaa !2
; CHECK-NEXT:   %cmp3 = icmp ne i64 %iv.next2, 4
; CHECK-NEXT:   br i1 %cmp3, label %for.body5, label %for.cond.cleanup4

; CHECK: for.cond.cleanup4:                                ; preds = %if.exit
; CHECK-NEXT:   %cmp = icmp ne i64 %iv.next, 4
; CHECK-NEXT:   br i1 %cmp, label %for.cond1.preheader, label %for.cond.cleanup

; CHECK: for.cond.cleanup:                                 ; preds = %for.cond.cleanup4
; CHECK-NEXT:   ret double** %subcache_malloccache
; CHECK-NEXT: }

; CHECK: define internal void @diffesubfn(double* %w3, double* %"w3'", double* %w9, double* %"w9'", double** %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %false = call i1 @falser()
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
; CHECK-NEXT:   %cmp3 = icmp ne i64 %iv.next2, 4
; CHECK-NEXT:   br i1 %cmp3, label %for.body5, label %for.cond.cleanup4

; CHECK: for.cond.cleanup4:                                ; preds = %if.exit
; CHECK-NEXT:   %cmp = icmp ne i64 %iv.next, 4
; CHECK-NEXT:   br i1 %cmp, label %for.cond1.preheader, label %invertfor.cond.cleanup4

; CHECK: invertentry:                                      ; preds = %invertfor.cond1.preheader
; CHECK-NEXT:   %[[tofree:.+]] = bitcast double** %tapeArg to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[tofree]])
; CHECK-NEXT:   ret void

; CHECK: invertfor.cond1.preheader:                        ; preds = %invertfor.body5
; CHECK-NEXT:   %[[cmp1:.+]] = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %[[cmp1]], label %invertentry, label %incinvertfor.cond1.preheader

; CHECK: incinvertfor.cond1.preheader:                     ; preds = %invertfor.cond1.preheader
; CHECK-NEXT:   %[[ivsub:.+]] = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertfor.cond.cleanup4

; CHECK: invertfor.body5:                                  ; preds = %invertif.exit, %invertif.end.i.i
; CHECK-NEXT:   %"call2.i.i.i'de.0" = phi double [ %"call2.i.i.i'de.1", %invertif.exit ], [ 0.000000e+00, %invertif.end.i.i
; CHECK-NEXT:   %[[iv1cmp:.+]] = icmp eq i64 %"iv1'ac.0", 0
; CHECK-NEXT:   br i1 %[[iv1cmp]], label %invertfor.cond1.preheader, label %incinvertfor.body5

; CHECK: incinvertfor.body5:                               ; preds = %invertfor.body5
; CHECK-NEXT:   %[[iv1sub:.+]] = add nsw i64 %"iv1'ac.0", -1
; CHECK-NEXT:   br label %invertif.exit

; CHECK: invertif.end.i.i:                                 ; preds = %invertif.exit
; CHECK-NEXT:   %add.ptr_unwrap = getelementptr inbounds double, double* %w9, i64 %"iv1'ac.0"
; CHECK-NEXT:   %"add.ptr'ipg_unwrap" = getelementptr inbounds double, double* %"w9'", i64 %"iv1'ac.0"
; CHECK-NEXT:   %[[unwrap5:.+]] = getelementptr inbounds double*, double** %tapeArg, i64 %add.i.i.i_unwrap
; CHECK-NEXT:   %[[unwrap6:.+]] = load double*, double** %[[unwrap5]]
; CHECK-NEXT:   call void @diffesumsq(double* %add.ptr_unwrap, double* %"add.ptr'ipg_unwrap", double %[[ddret:.+]], double* %[[unwrap6:.+]])
; CHECK-NEXT:   br label %invertfor.body5

; CHECK: invertif.exit:                                    ; preds = %invertfor.cond.cleanup4, %incinvertfor.body5
; CHECK-NEXT:   %"call2.i.i.i'de.1" = phi double [ %"call2.i.i.i'de.2", %invertfor.cond.cleanup4 ], [ %"call2.i.i.i'de.0", %incinvertfor.body5 ]
; CHECK-NEXT:   %"iv1'ac.0" = phi i64 [ 3, %invertfor.cond.cleanup4 ], [ %[[iv1sub]], %incinvertfor.body5 ]
; CHECK-NEXT:   %mul.i.i.i.i.i_unwrap = mul nsw i64 4, %"iv'ac.0"
; CHECK-NEXT:   %add.i.i.i_unwrap = add nsw i64 %mul.i.i.i.i.i_unwrap, %"iv1'ac.0"
; CHECK-NEXT:   %"arrayidx'ipg_unwrap" = getelementptr inbounds double, double* %"w3'", i64 %add.i.i.i_unwrap
; CHECK-NEXT:   %[[dret:.+]] = load double, double* %"arrayidx'ipg_unwrap", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"arrayidx'ipg_unwrap", align 8
; CHECK-NEXT:   %[[pdret:.+]] = fadd fast double %"call2.i.i.i'de.1", %[[dret]]
; CHECK-NEXT:   %[[ddret]] = select{{( fast)?}} i1 %false, double %"call2.i.i.i'de.1", double %[[pdret]]
; CHECK-NEXT:   br i1 %false, label %invertfor.body5, label %invertif.end.i.i

; CHECK: invertfor.cond.cleanup4:                          ; preds = %for.cond.cleanup4, %incinvertfor.cond1.preheader
; CHECK-NEXT:   %"call2.i.i.i'de.2" = phi double [ %"call2.i.i.i'de.0", %incinvertfor.cond1.preheader ], [ 0.000000e+00, %for.cond.cleanup4 ]
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %[[ivsub]], %incinvertfor.cond1.preheader ], [ 3, %for.cond.cleanup4 ]
; CHECK-NEXT:   br label %invertif.exit
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
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %[[inciv:.+]], %incinvertfor.body ], [ 3, %for.body ]
; CHECK-NEXT:   %[[ge1:.+]] = getelementptr inbounds double, double* %tapeArg, i64 %"iv'ac.0"
; TODO make this use iga6
; CHECK-NEXT:   %[[loc:.+]] = load double, double* %[[ge1]], align 8, !tbaa !2, !invariant.group ![[iga6_other:[0-9]+]]
; CHECK-NEXT:   %m0diffea6 = fmul fast double %differeturn, %[[loc]]
; CHECK-NEXT:   %[[adx:.+]] = fadd fast double %m0diffea6, %m0diffea6
; CHECK-NEXT:   %mul.i.i.i_unwrap = mul nsw i64 4, %"iv'ac.0"
; CHECK-NEXT:   %"arrayidx.i.i.i'ipg_unwrap" = getelementptr inbounds double, double* %"a3'", i64 %mul.i.i.i_unwrap
; CHECK-NEXT:   %[[paidx:.+]] = load double, double* %"arrayidx.i.i.i'ipg_unwrap", align 8
; CHECK-NEXT:   %[[ssidx:.+]] = fadd fast double %[[paidx]], %[[adx]]
; CHECK-NEXT:   store double %[[ssidx]], double* %"arrayidx.i.i.i'ipg_unwrap", align 8
; CHECK-NEXT:   %[[cmpiv:.+]] = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %[[cmpiv]], label %invertentry, label %incinvertfor.body

; CHECK: incinvertfor.body:                                ; preds = %invertfor.body
; CHECK-NEXT:   %[[inciv]] = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertfor.body
; CHECK-NEXT: }
