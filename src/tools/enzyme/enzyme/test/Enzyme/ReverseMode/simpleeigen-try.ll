; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -inline -mem2reg -instsimplify -adce -loop-deletion -correlated-propagation -simplifycfg -S | FileCheck %s

%"class.Eigen::Matrix" = type { %"class.Eigen::PlainObjectBase" }
%"class.Eigen::PlainObjectBase" = type { %"class.Eigen::DenseStorage" }
%"class.Eigen::DenseStorage" = type { double*, i64, i64 }
%"class.Eigen::Matrix.2" = type { %"class.Eigen::PlainObjectBase.3" }
%"class.Eigen::PlainObjectBase.3" = type { %"class.Eigen::DenseStorage.10" }
%"class.Eigen::DenseStorage.10" = type { double*, i64 }

@.str = private unnamed_addr constant [29 x i8] c"index >= 0 && index < size()\00", align 1
@.str.1 = private unnamed_addr constant [59 x i8] c"/usr/local/include/eigen3/Eigen/src/Core/DenseCoeffsBase.h\00", align 1
@__PRETTY_FUNCTION__._ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEixEl = private unnamed_addr constant [209 x i8] c"Eigen::DenseCoeffsBase<type-parameter-0-0, 1>::Scalar &Eigen::DenseCoeffsBase<Eigen::Matrix<double, -1, 1, 0, -1, 1>, 1>::operator[](Eigen::Index) [Derived = Eigen::Matrix<double, -1, 1, 0, -1, 1>, Level = 1]\00", align 1
@.str.4 = private unnamed_addr constant [53 x i8] c"row >= 0 && row < rows() && col >= 0 && col < cols()\00", align 1
@__PRETTY_FUNCTION__._ZNK5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi0EEclEll = private unnamed_addr constant [241 x i8] c"Eigen::DenseCoeffsBase<type-parameter-0-0, 0>::CoeffReturnType Eigen::DenseCoeffsBase<Eigen::Matrix<double, -1, -1, 0, -1, -1>, 0>::operator()(Eigen::Index, Eigen::Index) const [Derived = Eigen::Matrix<double, -1, -1, 0, -1, -1>, Level = 0]\00", align 1
@__PRETTY_FUNCTION__._ZNK5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi0EEclEl = private unnamed_addr constant [223 x i8] c"Eigen::DenseCoeffsBase<type-parameter-0-0, 0>::CoeffReturnType Eigen::DenseCoeffsBase<Eigen::Matrix<double, -1, 1, 0, -1, 1>, 0>::operator()(Eigen::Index) const [Derived = Eigen::Matrix<double, -1, 1, 0, -1, 1>, Level = 0]\00", align 1

; Function Attrs: nounwind uwtable
define dso_local void @_Z7dmatvecRKN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEERS1_RKNS0_IdLin1ELi1ELi0ELin1ELi1EEERS5_S7_S8_(%"class.Eigen::Matrix"* noalias dereferenceable(24) %W, %"class.Eigen::Matrix"* noalias dereferenceable(24) %Wp, %"class.Eigen::Matrix.2"* noalias dereferenceable(16) %b, %"class.Eigen::Matrix.2"* noalias dereferenceable(16) %bp, %"class.Eigen::Matrix.2"* noalias dereferenceable(16) %output, %"class.Eigen::Matrix.2"* noalias dereferenceable(16) %outputp) local_unnamed_addr #0 {
entry:
  %0 = tail call double (void (%"class.Eigen::Matrix"*, %"class.Eigen::Matrix.2"*, %"class.Eigen::Matrix.2"*)*, ...) @__enzyme_autodiff(void (%"class.Eigen::Matrix"*, %"class.Eigen::Matrix.2"*, %"class.Eigen::Matrix.2"*)* nonnull @_ZL6matvecRKN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEERKNS0_IdLin1ELi1ELi0ELin1ELi1EEERS4_, %"class.Eigen::Matrix"* nonnull %W, %"class.Eigen::Matrix"* nonnull %Wp, %"class.Eigen::Matrix.2"* nonnull %b, %"class.Eigen::Matrix.2"* nonnull %bp, %"class.Eigen::Matrix.2"* nonnull %output, %"class.Eigen::Matrix.2"* nonnull %outputp)
  ret void
}

; Function Attrs: noinline nounwind uwtable
define internal void @_ZL6matvecRKN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEERKNS0_IdLin1ELi1ELi0ELin1ELi1EEERS4_(%"class.Eigen::Matrix"* noalias nocapture readonly dereferenceable(24) %W, %"class.Eigen::Matrix.2"* noalias nocapture readonly dereferenceable(16) %b, %"class.Eigen::Matrix.2"* noalias nocapture readonly dereferenceable(16) %output) #1 {
entry:
  %m_rows.i.i = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %W, i64 0, i32 0, i32 0, i32 1
  %0 = load i64, i64* %m_rows.i.i, align 8, !tbaa !2
  %cmp45 = icmp sgt i64 %0, 0
  br i1 %cmp45, label %for.body.lr.ph, label %for.cond.cleanup

for.body.lr.ph:                                   ; preds = %entry
  %m_rows.i.i.i.i6.i31 = getelementptr inbounds %"class.Eigen::Matrix.2", %"class.Eigen::Matrix.2"* %output, i64 0, i32 0, i32 0, i32 1
  %1 = load i64, i64* %m_rows.i.i.i.i6.i31, align 8, !tbaa !8
  %2 = getelementptr inbounds %"class.Eigen::Matrix.2", %"class.Eigen::Matrix.2"* %output, i64 0, i32 0, i32 0, i32 0
  %3 = load double*, double** %2, align 8
  %m_cols.i.i = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %W, i64 0, i32 0, i32 0, i32 2
  %4 = load i64, i64* %m_cols.i.i, align 8
  %cmp643 = icmp sgt i64 %4, 0
  %m_rows.i.i.i.i6.i36 = getelementptr inbounds %"class.Eigen::Matrix.2", %"class.Eigen::Matrix.2"* %b, i64 0, i32 0, i32 0, i32 1
  %5 = load i64, i64* %m_rows.i.i.i.i6.i36, align 8
  %6 = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %W, i64 0, i32 0, i32 0, i32 0
  %7 = load double*, double** %6, align 8
  %8 = getelementptr inbounds %"class.Eigen::Matrix.2", %"class.Eigen::Matrix.2"* %b, i64 0, i32 0, i32 0, i32 0
  %9 = load double*, double** %8, align 8
  br i1 %cmp643, label %for.body.us, label %for.body

for.body.us:                                      ; preds = %for.body.lr.ph, %for.cond3.for.cond.cleanup7_crit_edge.us
  %indvars.iv50 = phi i64 [ %indvars.iv.next51, %for.cond3.for.cond.cleanup7_crit_edge.us ], [ 0, %for.body.lr.ph ]
  %cmp2.i32.us = icmp sgt i64 %1, %indvars.iv50
  br i1 %cmp2.i32.us, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEixEl.exit35.us, label %cond.false.i33

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEixEl.exit35.us: ; preds = %for.body.us
  %arrayidx.i.i.i34.us = getelementptr inbounds double, double* %3, i64 %indvars.iv50
  store double 0.000000e+00, double* %arrayidx.i.i.i34.us, align 8, !tbaa !10
  %cmp2.i40.us = icmp sgt i64 %0, %indvars.iv50
  br i1 %cmp2.i40.us, label %for.body8.us, label %cond.false.i41.split

for.body8.us:                                     ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEixEl.exit35.us, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEixEl.exit.us
  %10 = phi double [ %add.us, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEixEl.exit.us ], [ 0.000000e+00, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEixEl.exit35.us ]
  %indvars.iv = phi i64 [ %indvars.iv.next, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEixEl.exit.us ], [ 0, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEixEl.exit35.us ]
  %cmp7.i.us = icmp sgt i64 %4, %indvars.iv
  br i1 %cmp7.i.us, label %_ZNK5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi0EEclEll.exit.us, label %cond.false.i41.split

_ZNK5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi0EEclEll.exit.us: ; preds = %for.body8.us
  %cmp2.i37.us = icmp sgt i64 %5, %indvars.iv
  br i1 %cmp2.i37.us, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEixEl.exit.us, label %cond.false.i38

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEixEl.exit.us: ; preds = %_ZNK5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi0EEclEll.exit.us
  %mul.i.i.i.us = mul nsw i64 %0, %indvars.iv
  %add.i.i.i.us = add nsw i64 %mul.i.i.i.us, %indvars.iv50
  %arrayidx.i.i.i42.us = getelementptr inbounds double, double* %7, i64 %add.i.i.i.us
  %11 = load double, double* %arrayidx.i.i.i42.us, align 8, !tbaa !10
  %arrayidx.i.i.i39.us = getelementptr inbounds double, double* %9, i64 %indvars.iv
  %12 = load double, double* %arrayidx.i.i.i39.us, align 8, !tbaa !10
  %mul.us = fmul fast double %12, %11
  %add.us = fadd fast double %10, %mul.us
  store double %add.us, double* %arrayidx.i.i.i34.us, align 8, !tbaa !10
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %cmp6.us = icmp sgt i64 %4, %indvars.iv.next
  br i1 %cmp6.us, label %for.body8.us, label %for.cond3.for.cond.cleanup7_crit_edge.us

for.cond3.for.cond.cleanup7_crit_edge.us:         ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEixEl.exit.us
  %indvars.iv.next51 = add nuw nsw i64 %indvars.iv50, 1
  %cmp.us = icmp sgt i64 %0, %indvars.iv.next51
  br i1 %cmp.us, label %for.body.us, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEixEl.exit35, %for.cond3.for.cond.cleanup7_crit_edge.us, %entry
  ret void

for.body:                                         ; preds = %for.body.lr.ph, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEixEl.exit35
  %indvars.iv52 = phi i64 [ %indvars.iv.next53, %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEixEl.exit35 ], [ 0, %for.body.lr.ph ]
  %cmp2.i32 = icmp sgt i64 %1, %indvars.iv52
  br i1 %cmp2.i32, label %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEixEl.exit35, label %cond.false.i33

cond.false.i33:                                   ; preds = %for.body, %for.body.us
  tail call void @__assert_fail(i8* getelementptr inbounds ([29 x i8], [29 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([59 x i8], [59 x i8]* @.str.1, i64 0, i64 0), i32 408, i8* getelementptr inbounds ([209 x i8], [209 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEixEl, i64 0, i64 0)) #4
  unreachable

_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEixEl.exit35: ; preds = %for.body
  %arrayidx.i.i.i34 = getelementptr inbounds double, double* %3, i64 %indvars.iv52
  store double 0.000000e+00, double* %arrayidx.i.i.i34, align 8, !tbaa !10
  %indvars.iv.next53 = add nuw nsw i64 %indvars.iv52, 1
  %cmp = icmp sgt i64 %0, %indvars.iv.next53
  br i1 %cmp, label %for.body, label %for.cond.cleanup

cond.false.i41.split:                             ; preds = %_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi1EEixEl.exit35.us, %for.body8.us
  tail call void @__assert_fail(i8* getelementptr inbounds ([53 x i8], [53 x i8]* @.str.4, i64 0, i64 0), i8* getelementptr inbounds ([59 x i8], [59 x i8]* @.str.1, i64 0, i64 0), i32 118, i8* getelementptr inbounds ([241 x i8], [241 x i8]* @__PRETTY_FUNCTION__._ZNK5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi0EEclEll, i64 0, i64 0)) #4
  unreachable

cond.false.i38:                                   ; preds = %_ZNK5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEELi0EEclEll.exit.us
  tail call void @__assert_fail(i8* getelementptr inbounds ([29 x i8], [29 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([59 x i8], [59 x i8]* @.str.1, i64 0, i64 0), i32 180, i8* getelementptr inbounds ([223 x i8], [223 x i8]* @__PRETTY_FUNCTION__._ZNK5Eigen15DenseCoeffsBaseINS_6MatrixIdLin1ELi1ELi0ELin1ELi1EEELi0EEclEl, i64 0, i64 0)) #4
  unreachable
}

; Function Attrs: nounwind
declare double @__enzyme_autodiff(void (%"class.Eigen::Matrix"*, %"class.Eigen::Matrix.2"*, %"class.Eigen::Matrix.2"*)*, ...) #2

; Function Attrs: noreturn nounwind
declare dso_local void @__assert_fail(i8*, i8*, i32, i8*) local_unnamed_addr #3

attributes #0 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #2 = { nounwind }
attributes #3 = { noreturn nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #4 = { noreturn nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !7, i64 8}
!3 = !{!"_ZTSN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EEE", !4, i64 0, !7, i64 8, !7, i64 16}
!4 = !{!"any pointer", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}
!7 = !{!"long", !5, i64 0}
!8 = !{!9, !7, i64 8}
!9 = !{!"_ZTSN5Eigen12DenseStorageIdLin1ELin1ELi1ELi0EEE", !4, i64 0, !7, i64 8}
!10 = !{!11, !11, i64 0}
!11 = !{!"double", !5, i64 0}

; CHECK: define internal void @diffe_ZL6matvecRKN5Eigen6Matrix
