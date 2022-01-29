; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -simplifycfg -instcombine -adce -S | FileCheck %s

%"class.Eigen::Matrix" = type { [4 x double] }

%"class.Eigen::internal::generic_dense_assignment_kernel" = type { double** }
%"struct.Eigen::internal::binary_evaluator" = type { double*, double* }
%"struct.Eigen::internal::evaluator.12" = type { double* }

; Function Attrs: alwaysinline norecurse nounwind uwtable
define dso_local i32 @caller(i32 %argc, i8** nocapture readnone %argv) local_unnamed_addr #0 {
entry:
  %W = alloca double, align 16
  %O = alloca double, align 16
  %Wp = alloca double, align 16
  %Op = alloca double, align 16
  %call = call double (...) @__enzyme_autodiff(i8* bitcast (void (double*, double*)* @_ZL6matvecPKN5Eigen6MatrixIdLi2ELi2ELi0ELi2ELi2EEES3_PS1_ to i8*), double* nonnull %W, double* nonnull %Wp, double* nonnull %O, double* nonnull %Op) #2
  ret i32 0
}

declare dso_local double @__enzyme_autodiff(...)

; Function Attrs: alwaysinline
define internal void @_ZL6matvecPKN5Eigen6MatrixIdLi2ELi2ELi0ELi2ELi2EEES3_PS1_(double* %ad, double* noalias %arraydecay) #1 {
entry:
  %dstEvaluator = alloca double*, align 8
  %kernel = alloca double**, align 8
  store double* %arraydecay, double** %dstEvaluator, align 8, !tbaa !2
  store double** %dstEvaluator, double*** %kernel, align 8, !tbaa !8
  %a1 = bitcast double* %ad to <2 x double>*
  %a2 = load <2 x double>, <2 x double>* %a1, align 16, !tbaa !9
  %add = fadd <2 x double> %a2, %a2
  %a3 = bitcast double*** %kernel to %"struct.Eigen::internal::evaluator.12"**
  %a4 = load %"struct.Eigen::internal::evaluator.12"*, %"struct.Eigen::internal::evaluator.12"** %a3, align 8, !tbaa !10
  %m_data = getelementptr inbounds %"struct.Eigen::internal::evaluator.12", %"struct.Eigen::internal::evaluator.12"* %a4, i64 0, i32 0
  %a5 = load double*, double** %m_data, align 8, !tbaa !2
  %arrayidx = getelementptr inbounds double, double* %a5, i64 2
  %a8 = bitcast double* %arrayidx to <2 x double>*
  store <2 x double> %add, <2 x double>* %a8, align 16, !tbaa !9
  ret void
}

attributes #0 = { alwaysinline norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { alwaysinline }
attributes #2 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !4, i64 0}
!3 = !{!"_ZTSN5Eigen8internal9evaluatorINS_15PlainObjectBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEEEE", !4, i64 0, !7, i64 8}
!4 = !{!"any pointer", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}
!7 = !{!"_ZTSN5Eigen8internal19variable_if_dynamicIlLi2EEE"}
!8 = !{!4, !4, i64 0}
!9 = !{!5, !5, i64 0}
!10 = !{!11, !4, i64 0}
!11 = !{!"_ZTSN5Eigen8internal31generic_dense_assignment_kernelINS0_9evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEENS2_INS_13CwiseBinaryOpINS0_13scalar_sum_opIddEEKS4_S9_EEEENS0_9assign_opIddEELi0EEE", !4, i64 0, !4, i64 8, !4, i64 16, !4, i64 24}

; CHECK: define internal void @diffe_ZL6matvecPKN5Eigen6MatrixIdLi2ELi2ELi0ELi2ELi2EEES3_PS1_(double* %ad, double* %"ad'", double* noalias %arraydecay, double* %"arraydecay'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[a1ipc:.+]] = bitcast double* %"ad'" to <2 x double>*
; CHECK-NEXT:   %a1 = bitcast double* %ad to <2 x double>*
; CHECK-NEXT:   %a2 = load <2 x double>, <2 x double>* %a1, align 16, !tbaa !2
; CHECK-NEXT:   %add = fadd <2 x double> %a2, %a2
; CHECK-NEXT:   %[[arrayidxipge:.+]] = getelementptr inbounds double, double* %"arraydecay'", i64 2
; CHECK-NEXT:   %arrayidx = getelementptr inbounds double, double* %arraydecay, i64 2
; CHECK-NEXT:   %[[a8ipc:.+]] = bitcast double* %[[arrayidxipge]] to <2 x double>*
; CHECK-NEXT:   %a8 = bitcast double* %arrayidx to <2 x double>*
; CHECK-NEXT:   store <2 x double> %add, <2 x double>* %a8, align 16, !tbaa !2
; CHECK-NEXT:   %0 = load <2 x double>, <2 x double>* %[[a8ipc]], align 16
; CHECK-NEXT:   store <2 x double> zeroinitializer, <2 x double>* %[[a8ipc]], align 16
; CHECK-NEXT:   %1 = fadd fast <2 x double> %0, %0
; CHECK-NEXT:   %2 = load <2 x double>, <2 x double>* %[[a1ipc]], align 16
; CHECK-NEXT:   %3 = fadd fast <2 x double> %2, %1
; CHECK-NEXT:   store <2 x double> %3, <2 x double>* %[[a1ipc]], align 16
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
