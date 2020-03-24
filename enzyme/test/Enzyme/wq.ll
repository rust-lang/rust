; RUN: %opt < %s %loadEnzyme -enzyme -enzyme_preopt=false -mem2reg -sroa -simplifycfg -instcombine -adce -S | FileCheck %s

; the modref_map is wrong and overly conservative, upon fixing that this test should pass
; XFAIL: *

; ModuleID = '/home/wmoses/Enzyme/enzyme/test/Integration/simpleeigenstatic-sumsq.cpp'
source_filename = "/home/wmoses/Enzyme/enzyme/test/Integration/simpleeigenstatic-sumsq.cpp"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }
%"class.Eigen::Matrix" = type { %"class.Eigen::PlainObjectBase" }
%"class.Eigen::PlainObjectBase" = type { %"class.Eigen::DenseStorage" }
%"class.Eigen::DenseStorage" = type { %"struct.Eigen::internal::plain_array" }
%"struct.Eigen::internal::plain_array" = type { [4 x double] }
%"class.Eigen::CwiseNullaryOp" = type { %"class.Eigen::internal::variable_if_dynamic", %"class.Eigen::internal::variable_if_dynamic", %"struct.Eigen::internal::scalar_constant_op" }
%"class.Eigen::internal::variable_if_dynamic" = type { i8 }
%"struct.Eigen::internal::scalar_constant_op" = type { double }
%"struct.Eigen::EigenBase.5" = type { i8 }
%"class.Eigen::DenseCoeffsBase.0" = type { i8 }
%"class.Eigen::DenseBase.3" = type { i8 }
%"class.Eigen::DenseBase" = type { i8 }
%"struct.Eigen::EigenBase" = type { i8 }
%"struct.Eigen::internal::scalar_sum_op" = type { i8 }
%"class.Eigen::internal::redux_evaluator" = type { %"struct.Eigen::internal::evaluator", %"class.Eigen::Matrix"* }
%"struct.Eigen::internal::evaluator" = type { %"struct.Eigen::internal::evaluator.base", [7 x i8] }
%"struct.Eigen::internal::evaluator.base" = type <{ double*, %"class.Eigen::internal::variable_if_dynamic" }>
%"class.Eigen::internal::noncopyable" = type { i8 }
%"struct.Eigen::internal::evaluator.6" = type <{ double*, %"class.Eigen::internal::variable_if_dynamic", [7 x i8] }>
%"struct.Eigen::internal::evaluator_base" = type { i8 }
%"class.Eigen::DenseCoeffsBase" = type { i8 }
%"class.Eigen::MatrixBase.2" = type { i8 }
%"class.Eigen::MatrixBase" = type { i8 }
%"struct.Eigen::internal::assign_op" = type { i8 }
%"struct.Eigen::internal::evaluator.8" = type <{ %"struct.Eigen::internal::scalar_constant_op", %"struct.Eigen::internal::nullary_wrapper", [7 x i8] }>
%"struct.Eigen::internal::nullary_wrapper" = type { i8 }
%"class.Eigen::internal::generic_dense_assignment_kernel" = type { %"struct.Eigen::internal::evaluator"*, %"struct.Eigen::internal::evaluator.8"*, %"struct.Eigen::internal::assign_op"*, %"class.Eigen::Matrix"* }
%"struct.Eigen::internal::evaluator_base.9" = type { i8 }

$_ZN5Eigen9DenseBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE8ConstantEllRKd = comdat any

$_ZN5Eigen6MatrixIdLi2ELi2ELi0ELi2ELi2EEC2INS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES1_EEEERKNS_9EigenBaseIT_EE = comdat any

$_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEELi1EEclEll = comdat any

$_ZNK5Eigen9DenseBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE3sumEv = comdat any

$_ZNK5Eigen9EigenBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE7derivedEv = comdat any

$_ZNK5Eigen9DenseBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE5reduxINS_8internal13scalar_sum_opIddEEEEdRKT_ = comdat any

$_ZN5Eigen8internal13scalar_sum_opIddEC2Ev = comdat any

$_ZNK5Eigen9EigenBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE4rowsEv = comdat any

$_ZNK5Eigen9EigenBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE4colsEv = comdat any

$_ZN5Eigen8internal15redux_evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEC2ERKS3_ = comdat any

$_ZN5Eigen8internal10redux_implINS0_13scalar_sum_opIddEENS0_15redux_evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEELi3ELi2EE3runERKS7_RKS3_ = comdat any

$_ZN5Eigen8internal15redux_evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEED2Ev = comdat any

$_ZNK5Eigen15PlainObjectBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE4rowsEv = comdat any

$_ZN5Eigen12DenseStorageIdLi4ELi2ELi2ELi0EE4rowsEv = comdat any

$_ZNK5Eigen15PlainObjectBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE4colsEv = comdat any

$_ZN5Eigen12DenseStorageIdLi4ELi2ELi2ELi0EE4colsEv = comdat any

$_ZN5Eigen8internal9evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEC2ERKS3_ = comdat any

$_ZN5Eigen8internal9evaluatorINS_15PlainObjectBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEEEC2ERKS5_ = comdat any

$_ZN5Eigen8internal14evaluator_baseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEC2Ev = comdat any

$_ZNK5Eigen15PlainObjectBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE4dataEv = comdat any

$_ZNK5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEELi3EE11outerStrideEv = comdat any

$_ZN5Eigen8internal19variable_if_dynamicIlLi2EEC2El = comdat any

$_ZN5Eigen8internal11noncopyableC2Ev = comdat any

$_ZNK5Eigen12DenseStorageIdLi4ELi2ELi2ELi0EE4dataEv = comdat any

$_ZNK5Eigen6MatrixIdLi2ELi2ELi0ELi2ELi2EE11outerStrideEv = comdat any

$_ZNK5Eigen9DenseBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE9innerSizeEv = comdat any

$_ZNK5Eigen8internal15redux_evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE4rowsEv = comdat any

$_ZNK5Eigen8internal15redux_evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE4colsEv = comdat any

$_ZNK5Eigen8internal13scalar_sum_opIddE6preduxIDv2_dEEKdRKT_ = comdat any

$_ZN5Eigen8internal18redux_vec_unrollerINS0_13scalar_sum_opIddEENS0_15redux_evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEELi0ELi2EE3runERKS7_RKS3_ = comdat any

$_ZN5Eigen8internal6preduxIDv2_dEENS0_15unpacket_traitsIT_E4typeERKS4_ = comdat any

$_ZN5Eigen8internal6pfirstIDv2_dEENS0_15unpacket_traitsIT_E4typeERKS4_ = comdat any

$_ZNK5Eigen8internal13scalar_sum_opIddE8packetOpIDv2_dEEKT_RS6_S7_ = comdat any

$_ZN5Eigen8internal18redux_vec_unrollerINS0_13scalar_sum_opIddEENS0_15redux_evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEELi0ELi1EE3runERKS7_RKS3_ = comdat any

$_ZN5Eigen8internal18redux_vec_unrollerINS0_13scalar_sum_opIddEENS0_15redux_evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEELi1ELi1EE3runERKS7_RKS3_ = comdat any

$_ZN5Eigen8internal4paddIDv2_dEET_RKS3_S5_ = comdat any

$_ZNK5Eigen8internal15redux_evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE18packetByOuterInnerILi16EDv2_dEET0_ll = comdat any

$_ZNK5Eigen8internal9evaluatorINS_15PlainObjectBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEEE6packetILi16EDv2_dEET0_ll = comdat any

$_ZN5Eigen8internal19variable_if_dynamicIlLi2EE5valueEv = comdat any

$_ZN5Eigen8internal5ploadIDv2_dEET_PKNS0_15unpacket_traitsIS3_E4typeE = comdat any

$_ZN5Eigen8internal11noncopyableD2Ev = comdat any

$_ZN5Eigen9DenseBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE11NullaryExprINS_8internal18scalar_constant_opIdEEEEKNS_14CwiseNullaryOpIT_S2_EEllRKS9_ = comdat any

$_ZN5Eigen8internal18scalar_constant_opIdEC2ERKd = comdat any

$_ZN5Eigen14CwiseNullaryOpINS_8internal18scalar_constant_opIdEENS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEC2EllRKS3_ = comdat any

$_ZN5Eigen10MatrixBaseINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEENS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEEEC2Ev = comdat any

$_ZN5Eigen8internal18scalar_constant_opIdEC2ERKS2_ = comdat any

$_ZN5Eigen9DenseBaseINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEENS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEEEC2Ev = comdat any

$_ZNK5Eigen9EigenBaseINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEENS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEEE7derivedEv = comdat any

$_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEC2INS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEERKNS_9DenseBaseIT_EE = comdat any

$_ZN5Eigen10MatrixBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEC2Ev = comdat any

$_ZN5Eigen12DenseStorageIdLi4ELi2ELi2ELi0EEC2Ev = comdat any

$_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE22_check_template_paramsEv = comdat any

$_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE10resizeLikeINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEEvRKNS_9EigenBaseIT_EE = comdat any

$_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE12_set_noaliasINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEERS2_RKNS_9DenseBaseIT_EE = comdat any

$_ZN5Eigen9DenseBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEC2Ev = comdat any

$_ZN5Eigen8internal11plain_arrayIdLi4ELi0ELi16EEC2Ev = comdat any

$_ZN5Eigen8internal28check_static_allocation_sizeIdLi4EEEvv = comdat any

$_ZNK5Eigen14CwiseNullaryOpINS_8internal18scalar_constant_opIdEENS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE4rowsEv = comdat any

$_ZNK5Eigen14CwiseNullaryOpINS_8internal18scalar_constant_opIdEENS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE4colsEv = comdat any

$_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE6resizeEll = comdat any

$_ZN5Eigen12DenseStorageIdLi4ELi2ELi2ELi0EE6resizeElll = comdat any

$_ZN5Eigen8internal24call_assignment_no_aliasINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EENS0_9assign_opIddEEEEvRT_RKT0_RKT1_ = comdat any

$_ZN5Eigen9EigenBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE7derivedEv = comdat any

$_ZN5Eigen8internal9assign_opIddEC2Ev = comdat any

$_ZN5Eigen8internal10AssignmentINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EENS0_9assign_opIddEENS0_11Dense2DenseEvE3runERS3_RKS7_RKS9_ = comdat any

$_ZN5Eigen8internal18check_for_aliasingINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EEEEvRKT_RKT0_ = comdat any

$_ZN5Eigen8internal26call_dense_assignment_loopINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EENS0_9assign_opIddEEEEvRT_RKT0_RKT1_ = comdat any

$_ZN5Eigen8internal27checkTransposeAliasing_implINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EELb0EE3runERKS3_RKS7_ = comdat any

$_ZN5Eigen8internal9evaluatorINS_14CwiseNullaryOpINS0_18scalar_constant_opIdEENS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEEEC2ERKS7_ = comdat any

$_ZN5Eigen8internal17resize_if_allowedINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EEddEEvRT_RKT0_RKNS0_9assign_opIT1_T2_EE = comdat any

$_ZNK5Eigen9EigenBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE18const_cast_derivedEv = comdat any

$_ZN5Eigen8internal31generic_dense_assignment_kernelINS0_9evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEENS2_INS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES4_EEEENS0_9assign_opIddEELi0EEC2ERS5_RKSA_RKSC_RS4_ = comdat any

$_ZN5Eigen8internal21dense_assignment_loopINS0_31generic_dense_assignment_kernelINS0_9evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEENS3_INS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES5_EEEENS0_9assign_opIddEELi0EEELi2ELi2EE3runERSE_ = comdat any

$_ZN5Eigen8internal14evaluator_baseINS_14CwiseNullaryOpINS0_18scalar_constant_opIdEENS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEEEC2Ev = comdat any

$_ZNK5Eigen14CwiseNullaryOpINS_8internal18scalar_constant_opIdEENS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE7functorEv = comdat any

$_ZN5Eigen8internal47copy_using_evaluator_innervec_CompleteUnrollingINS0_31generic_dense_assignment_kernelINS0_9evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEENS3_INS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES5_EEEENS0_9assign_opIddEELi0EEELi0ELi4EE3runERSE_ = comdat any

$_ZN5Eigen8internal31generic_dense_assignment_kernelINS0_9evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEENS2_INS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES4_EEEENS0_9assign_opIddEELi0EE24assignPacketByOuterInnerILi16ELi16EDv2_dEEvll = comdat any

$_ZN5Eigen8internal47copy_using_evaluator_innervec_CompleteUnrollingINS0_31generic_dense_assignment_kernelINS0_9evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEENS3_INS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES5_EEEENS0_9assign_opIddEELi0EEELi2ELi4EE3runERSE_ = comdat any

$_ZN5Eigen8internal31generic_dense_assignment_kernelINS0_9evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEENS2_INS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES4_EEEENS0_9assign_opIddEELi0EE20rowIndexByOuterInnerEll = comdat any

$_ZN5Eigen8internal31generic_dense_assignment_kernelINS0_9evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEENS2_INS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES4_EEEENS0_9assign_opIddEELi0EE20colIndexByOuterInnerEll = comdat any

$_ZN5Eigen8internal31generic_dense_assignment_kernelINS0_9evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEENS2_INS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES4_EEEENS0_9assign_opIddEELi0EE12assignPacketILi16ELi16EDv2_dEEvll = comdat any

$_ZNK5Eigen8internal9assign_opIddE12assignPacketILi16EDv2_dEEvPdRKT0_ = comdat any

$_ZN5Eigen8internal9evaluatorINS_15PlainObjectBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEEE8coeffRefEll = comdat any

$_ZNK5Eigen8internal9evaluatorINS_14CwiseNullaryOpINS0_18scalar_constant_opIdEENS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEEE6packetILi16EDv2_dlEET0_T1_SC_ = comdat any

$_ZN5Eigen8internal6pstoreIdDv2_dEEvPT_RKT0_ = comdat any

$_ZNK5Eigen8internal15nullary_wrapperIdNS0_18scalar_constant_opIdEELb1ELb0ELb0EE8packetOpIDv2_dlEET_RKS3_T0_SA_ = comdat any

$_ZNK5Eigen8internal18scalar_constant_opIdE8packetOpIDv2_dEEKT_v = comdat any

$_ZN5Eigen8internal5pset1IDv2_dEET_RKNS0_15unpacket_traitsIS3_E4typeE = comdat any

$_ZN5Eigen8internal47copy_using_evaluator_innervec_CompleteUnrollingINS0_31generic_dense_assignment_kernelINS0_9evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEENS3_INS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES5_EEEENS0_9assign_opIddEELi0EEELi4ELi4EE3runERSE_ = comdat any

$_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEELi1EE8coeffRefEll = comdat any

@stderr = external dso_local local_unnamed_addr global %struct._IO_FILE*, align 8
@.str = private unnamed_addr constant [18 x i8] c"W(o=%d, i=%d)=%f\0A\00", align 1
@.str.1 = private unnamed_addr constant [68 x i8] c"Assertion Failed: fabs( [%s = %g] - [%s = %g] ) > %g at %s:%d (%s)\0A\00", align 1
@.str.2 = private unnamed_addr constant [9 x i8] c"Wp(i, o)\00", align 1
@.str.3 = private unnamed_addr constant [4 x i8] c"1.0\00", align 1
@.str.4 = private unnamed_addr constant [72 x i8] c"/home/wmoses/Enzyme/enzyme/test/Integration/simpleeigenstatic-sumsq.cpp\00", align 1
@__PRETTY_FUNCTION__.main = private unnamed_addr constant [23 x i8] c"int main(int, char **)\00", align 1
@.str.5 = private unnamed_addr constant [19 x i8] c"Wp(o=%d, i=%d)=%f\0A\00", align 1
@.str.6 = private unnamed_addr constant [68 x i8] c"this->rows()>0 && this->cols()>0 && \22you are using an empty matrix\22\00", align 1
@.str.7 = private unnamed_addr constant [49 x i8] c"/usr/local/include/eigen3/Eigen/src/Core/Redux.h\00", align 1
@__PRETTY_FUNCTION__._ZNK5Eigen9DenseBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE5reduxINS_8internal13scalar_sum_opIddEEEEdRKT_ = private unnamed_addr constant [234 x i8] c"typename internal::traits<Derived>::Scalar Eigen::DenseBase<Eigen::Matrix<double, 2, 2, 0, 2, 2> >::redux(const Func &) const [Derived = Eigen::Matrix<double, 2, 2, 0, 2, 2>, BinaryOp = Eigen::internal::scalar_sum_op<double, double>]\00", align 1
@.str.8 = private unnamed_addr constant [14 x i8] c"v == T(Value)\00", align 1
@.str.9 = private unnamed_addr constant [58 x i8] c"/usr/local/include/eigen3/Eigen/src/Core/util/XprHelper.h\00", align 1
@__PRETTY_FUNCTION__._ZN5Eigen8internal19variable_if_dynamicIlLi2EEC2El = private unnamed_addr constant [92 x i8] c"Eigen::internal::variable_if_dynamic<long, 2>::variable_if_dynamic(T) [T = long, Value = 2]\00", align 1
@.str.10 = private unnamed_addr constant [64 x i8] c"mat.rows()>0 && mat.cols()>0 && \22you are using an empty matrix\22\00", align 1
@__PRETTY_FUNCTION__._ZN5Eigen8internal10redux_implINS0_13scalar_sum_opIddEENS0_15redux_evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEELi3ELi2EE3runERKS7_RKS3_ = private unnamed_addr constant [449 x i8] c"static Eigen::internal::redux_impl<type-parameter-0-0, type-parameter-0-1, 3, 2>::Scalar Eigen::internal::redux_impl<Eigen::internal::scalar_sum_op<double, double>, Eigen::internal::redux_evaluator<Eigen::Matrix<double, 2, 2, 0, 2, 2> >, 3, 2>::run(const Derived &, const Func &) [Func = Eigen::internal::scalar_sum_op<double, double>, Derived = Eigen::internal::redux_evaluator<Eigen::Matrix<double, 2, 2, 0, 2, 2> >, Traversal = 3, Unrolling = 2]\00", align 1
@.str.11 = private unnamed_addr constant [149 x i8] c"rows >= 0 && (RowsAtCompileTime == Dynamic || RowsAtCompileTime == rows) && cols >= 0 && (ColsAtCompileTime == Dynamic || ColsAtCompileTime == cols)\00", align 1
@.str.12 = private unnamed_addr constant [58 x i8] c"/usr/local/include/eigen3/Eigen/src/Core/CwiseNullaryOp.h\00", align 1
@__PRETTY_FUNCTION__._ZN5Eigen14CwiseNullaryOpINS_8internal18scalar_constant_opIdEENS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEC2EllRKS3_ = private unnamed_addr constant [278 x i8] c"Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<double>, Eigen::Matrix<double, 2, 2, 0, 2, 2> >::CwiseNullaryOp(Eigen::Index, Eigen::Index, const NullaryOp &) [NullaryOp = Eigen::internal::scalar_constant_op<double>, MatrixType = Eigen::Matrix<double, 2, 2, 0, 2, 2>]\00", align 1
@.str.13 = private unnamed_addr constant [192 x i8] c"(internal::UIntPtr(array) & (15)) == 0 && \22this assertion is explained here: \22 \22http://eigen.tuxfamily.org/dox-devel/group__TopicUnalignedArrayAssert.html\22 \22 **** READ THIS WEB PAGE !!! ****\22\00", align 1
@.str.14 = private unnamed_addr constant [56 x i8] c"/usr/local/include/eigen3/Eigen/src/Core/DenseStorage.h\00", align 1
@__PRETTY_FUNCTION__._ZN5Eigen8internal11plain_arrayIdLi4ELi0ELi16EEC2Ev = private unnamed_addr constant [127 x i8] c"Eigen::internal::plain_array<double, 4, 0, 16>::plain_array() [T = double, Size = 4, MatrixOrArrayOptions = 0, Alignment = 16]\00", align 1
@.str.15 = private unnamed_addr constant [399 x i8] c"(!(RowsAtCompileTime!=Dynamic) || (rows==RowsAtCompileTime)) && (!(ColsAtCompileTime!=Dynamic) || (cols==ColsAtCompileTime)) && (!(RowsAtCompileTime==Dynamic && MaxRowsAtCompileTime!=Dynamic) || (rows<=MaxRowsAtCompileTime)) && (!(ColsAtCompileTime==Dynamic && MaxColsAtCompileTime!=Dynamic) || (cols<=MaxColsAtCompileTime)) && rows>=0 && cols>=0 && \22Invalid sizes when resizing a matrix or array.\22\00", align 1
@.str.16 = private unnamed_addr constant [59 x i8] c"/usr/local/include/eigen3/Eigen/src/Core/PlainObjectBase.h\00", align 1
@__PRETTY_FUNCTION__._ZN5Eigen15PlainObjectBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE6resizeEll = private unnamed_addr constant [152 x i8] c"void Eigen::PlainObjectBase<Eigen::Matrix<double, 2, 2, 0, 2, 2> >::resize(Eigen::Index, Eigen::Index) [Derived = Eigen::Matrix<double, 2, 2, 0, 2, 2>]\00", align 1
@.str.17 = private unnamed_addr constant [47 x i8] c"dst.rows() == dstRows && dst.cols() == dstCols\00", align 1
@.str.18 = private unnamed_addr constant [59 x i8] c"/usr/local/include/eigen3/Eigen/src/Core/AssignEvaluator.h\00", align 1
@__PRETTY_FUNCTION__._ZN5Eigen8internal17resize_if_allowedINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EEddEEvRT_RKT0_RKNS0_9assign_opIT1_T2_EE = private unnamed_addr constant [309 x i8] c"void Eigen::internal::resize_if_allowed(DstXprType &, const SrcXprType &, const internal::assign_op<T1, T2> &) [DstXprType = Eigen::Matrix<double, 2, 2, 0, 2, 2>, SrcXprType = Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<double>, Eigen::Matrix<double, 2, 2, 0, 2, 2> >, T1 = double, T2 = double]\00", align 1
@.str.19 = private unnamed_addr constant [53 x i8] c"row >= 0 && row < rows() && col >= 0 && col < cols()\00", align 1
@.str.20 = private unnamed_addr constant [59 x i8] c"/usr/local/include/eigen3/Eigen/src/Core/DenseCoeffsBase.h\00", align 1
@__PRETTY_FUNCTION__._ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEELi1EEclEll = private unnamed_addr constant [219 x i8] c"Eigen::DenseCoeffsBase<type-parameter-0-0, 1>::Scalar &Eigen::DenseCoeffsBase<Eigen::Matrix<double, 2, 2, 0, 2, 2>, 1>::operator()(Eigen::Index, Eigen::Index) [Derived = Eigen::Matrix<double, 2, 2, 0, 2, 2>, Level = 1]\00", align 1

; Function Attrs: norecurse nounwind uwtable
define dso_local i32 @main(i32 %argc, i8** nocapture readnone %argv) local_unnamed_addr #0 {
entry:
  %W = alloca %"class.Eigen::Matrix", align 16
  %ref.tmp = alloca %"class.Eigen::CwiseNullaryOp", align 8
  %ref.tmp1 = alloca double, align 8
  %Wp = alloca %"class.Eigen::Matrix", align 16
  %ref.tmp2 = alloca %"class.Eigen::CwiseNullaryOp", align 8
  %ref.tmp3 = alloca double, align 8
  %0 = bitcast %"class.Eigen::Matrix"* %W to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %0) #12
  %1 = getelementptr inbounds %"class.Eigen::CwiseNullaryOp", %"class.Eigen::CwiseNullaryOp"* %ref.tmp, i64 0, i32 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %1) #12
  %2 = bitcast double* %ref.tmp1 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2) #12
  store double 3.000000e+00, double* %ref.tmp1, align 8, !tbaa !2
  call void @_ZN5Eigen9DenseBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE8ConstantEllRKd(%"class.Eigen::CwiseNullaryOp"* nonnull sret %ref.tmp, i64 2, i64 2, double* nonnull dereferenceable(8) %ref.tmp1)
  %3 = bitcast %"class.Eigen::CwiseNullaryOp"* %ref.tmp to %"struct.Eigen::EigenBase.5"*
  call void @_ZN5Eigen6MatrixIdLi2ELi2ELi0ELi2ELi2EEC2INS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES1_EEEERKNS_9EigenBaseIT_EE(%"class.Eigen::Matrix"* nonnull %W, %"struct.Eigen::EigenBase.5"* nonnull dereferenceable(1) %3)
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2) #12
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %1) #12
  %4 = bitcast %"class.Eigen::Matrix"* %Wp to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %4) #12
  %5 = getelementptr inbounds %"class.Eigen::CwiseNullaryOp", %"class.Eigen::CwiseNullaryOp"* %ref.tmp2, i64 0, i32 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %5) #12
  %6 = bitcast double* %ref.tmp3 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6) #12
  store double 0.000000e+00, double* %ref.tmp3, align 8, !tbaa !2
  call void @_ZN5Eigen9DenseBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE8ConstantEllRKd(%"class.Eigen::CwiseNullaryOp"* nonnull sret %ref.tmp2, i64 2, i64 2, double* nonnull dereferenceable(8) %ref.tmp3)
  %7 = bitcast %"class.Eigen::CwiseNullaryOp"* %ref.tmp2 to %"struct.Eigen::EigenBase.5"*
  call void @_ZN5Eigen6MatrixIdLi2ELi2ELi0ELi2ELi2EEC2INS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES1_EEEERKNS_9EigenBaseIT_EE(%"class.Eigen::Matrix"* nonnull %Wp, %"struct.Eigen::EigenBase.5"* nonnull dereferenceable(1) %7)
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6) #12
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %5) #12
  %call = call double @__enzyme_autodiff(i8* bitcast (double (%"class.Eigen::Matrix"*)* @_ZL6matvecPKN5Eigen6MatrixIdLi2ELi2ELi0ELi2ELi2EEE to i8*), i8* nonnull %0, i8* nonnull %4) #12
  %8 = bitcast %"class.Eigen::Matrix"* %W to %"class.Eigen::DenseCoeffsBase.0"*
  br label %for.cond4.preheader

for.cond4.preheader:                              ; preds = %for.cond.cleanup7, %entry
  %indvars.iv78 = phi i64 [ 0, %entry ], [ %indvars.iv.next79, %for.cond.cleanup7 ]
  %9 = trunc i64 %indvars.iv78 to i32
  br label %for.body8

for.cond17.preheader:                             ; preds = %for.cond.cleanup7
  %10 = bitcast %"class.Eigen::Matrix"* %Wp to %"class.Eigen::DenseCoeffsBase.0"*
  br label %for.cond23.preheader

for.cond.cleanup7:                                ; preds = %for.body8
  %indvars.iv.next79 = add nuw nsw i64 %indvars.iv78, 1
  %exitcond80 = icmp eq i64 %indvars.iv.next79, 2
  br i1 %exitcond80, label %for.cond17.preheader, label %for.cond4.preheader

for.body8:                                        ; preds = %for.body8, %for.cond4.preheader
  %indvars.iv76 = phi i64 [ 0, %for.cond4.preheader ], [ %indvars.iv.next77, %for.body8 ]
  %11 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !6
  %call11 = call dereferenceable(8) double* @_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEELi1EEclEll(%"class.Eigen::DenseCoeffsBase.0"* nonnull %8, i64 %indvars.iv76, i64 %indvars.iv78)
  %12 = load double, double* %call11, align 8, !tbaa !2
  %13 = trunc i64 %indvars.iv76 to i32
  %call12 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %11, i8* getelementptr inbounds ([18 x i8], [18 x i8]* @.str, i64 0, i64 0), i32 %13, i32 %9, double %12) #13
  %indvars.iv.next77 = add nuw nsw i64 %indvars.iv76, 1
  %exitcond = icmp eq i64 %indvars.iv.next77, 2
  br i1 %exitcond, label %for.cond.cleanup7, label %for.body8

for.cond23.preheader:                             ; preds = %for.cond17.preheader, %for.cond.cleanup26
  %indvars.iv74 = phi i64 [ 0, %for.cond17.preheader ], [ %indvars.iv.next75, %for.cond.cleanup26 ]
  %14 = trunc i64 %indvars.iv74 to i32
  br label %for.body27

for.cond.cleanup20:                               ; preds = %for.cond.cleanup26
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %4) #12
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %0) #12
  ret i32 0

for.cond.cleanup26:                               ; preds = %if.end
  %indvars.iv.next75 = add nuw nsw i64 %indvars.iv74, 1
  %cmp19 = icmp ult i64 %indvars.iv.next75, 2
  br i1 %cmp19, label %for.cond23.preheader, label %for.cond.cleanup20

for.body27:                                       ; preds = %for.cond23.preheader, %if.end
  %indvars.iv = phi i64 [ 0, %for.cond23.preheader ], [ %indvars.iv.next, %if.end ]
  %call30 = call dereferenceable(8) double* @_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEELi1EEclEll(%"class.Eigen::DenseCoeffsBase.0"* nonnull %10, i64 %indvars.iv, i64 %indvars.iv74)
  %15 = load double, double* %call30, align 8, !tbaa !2
  %sub = fadd double %15, -1.000000e+00
  %16 = call double @llvm.fabs.f64(double %sub)
  %cmp31 = fcmp ogt double %16, 1.000000e-10
  %17 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !6
  %call34 = call dereferenceable(8) double* @_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEELi1EEclEll(%"class.Eigen::DenseCoeffsBase.0"* nonnull %10, i64 %indvars.iv, i64 %indvars.iv74)
  %18 = load double, double* %call34, align 8, !tbaa !2
  br i1 %cmp31, label %if.then, label %if.end

if.then:                                          ; preds = %for.body27
  %call35 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %17, i8* getelementptr inbounds ([68 x i8], [68 x i8]* @.str.1, i64 0, i64 0), i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str.2, i64 0, i64 0), double %18, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.3, i64 0, i64 0), double 1.000000e+00, double 1.000000e-10, i8* getelementptr inbounds ([72 x i8], [72 x i8]* @.str.4, i64 0, i64 0), i32 64, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @__PRETTY_FUNCTION__.main, i64 0, i64 0)) #13
  call void @abort() #14
  unreachable

if.end:                                           ; preds = %for.body27
  %19 = trunc i64 %indvars.iv to i32
  %call39 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %17, i8* getelementptr inbounds ([19 x i8], [19 x i8]* @.str.5, i64 0, i64 0), i32 %19, i32 %14, double %18) #13
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %cmp25 = icmp ult i64 %indvars.iv.next, 2
  br i1 %cmp25, label %for.body27, label %for.cond.cleanup26
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZN5Eigen9DenseBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE8ConstantEllRKd(%"class.Eigen::CwiseNullaryOp"* noalias sret %agg.result, i64 %rows, i64 %cols, double* dereferenceable(8) %value) local_unnamed_addr #2 comdat align 2 {
entry:
  %ref.tmp = alloca %"struct.Eigen::internal::scalar_constant_op", align 8
  %0 = bitcast %"struct.Eigen::internal::scalar_constant_op"* %ref.tmp to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %0) #12
  call void @_ZN5Eigen8internal18scalar_constant_opIdEC2ERKd(%"struct.Eigen::internal::scalar_constant_op"* nonnull %ref.tmp, double* nonnull dereferenceable(8) %value)
  call void @_ZN5Eigen9DenseBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE11NullaryExprINS_8internal18scalar_constant_opIdEEEEKNS_14CwiseNullaryOpIT_S2_EEllRKS9_(%"class.Eigen::CwiseNullaryOp"* sret %agg.result, i64 %rows, i64 %cols, %"struct.Eigen::internal::scalar_constant_op"* nonnull dereferenceable(8) %ref.tmp)
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %0) #12
  ret void
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local void @_ZN5Eigen6MatrixIdLi2ELi2ELi0ELi2ELi2EEC2INS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES1_EEEERKNS_9EigenBaseIT_EE(%"class.Eigen::Matrix"* %this, %"struct.Eigen::EigenBase.5"* dereferenceable(1) %other) unnamed_addr #3 comdat align 2 {
entry:
  %0 = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %this, i64 0, i32 0
  %call = tail call dereferenceable(16) %"class.Eigen::CwiseNullaryOp"* @_ZNK5Eigen9EigenBaseINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEENS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEEE7derivedEv(%"struct.Eigen::EigenBase.5"* nonnull %other)
  %1 = bitcast %"class.Eigen::CwiseNullaryOp"* %call to %"class.Eigen::DenseBase.3"*
  tail call void @_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEC2INS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEERKNS_9DenseBaseIT_EE(%"class.Eigen::PlainObjectBase"* %0, %"class.Eigen::DenseBase.3"* nonnull dereferenceable(1) %1)
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

declare dso_local double @__enzyme_autodiff(i8*, i8*, i8*) local_unnamed_addr #4

; Function Attrs: noinline nounwind uwtable
define internal double @_ZL6matvecPKN5Eigen6MatrixIdLi2ELi2ELi0ELi2ELi2EEE(%"class.Eigen::Matrix"* noalias %W) #5 {
entry:
  %func = alloca %"struct.Eigen::internal::scalar_sum_op", align 1
  %this = bitcast %"class.Eigen::Matrix"* %W to %"class.Eigen::DenseBase"* 
  %thisEval = alloca %"class.Eigen::internal::redux_evaluator", align 8
  %b0 = bitcast %"class.Eigen::DenseBase"* %this to %"struct.Eigen::EigenBase"*
  %call4 = bitcast %"struct.Eigen::EigenBase"* %b0 to %"class.Eigen::Matrix"*
  call void @_ZN5Eigen8internal15redux_evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEC2ERKS3_(%"class.Eigen::internal::redux_evaluator"* nonnull %thisEval, %"class.Eigen::Matrix"* nonnull dereferenceable(32) %call4)
  %call5 = call double @_ZN5Eigen8internal10redux_implINS0_13scalar_sum_opIddEENS0_15redux_evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEELi3ELi2EE3runERKS7_RKS3_(%"class.Eigen::internal::redux_evaluator"* nonnull dereferenceable(24) %thisEval, %"struct.Eigen::internal::scalar_sum_op"* nonnull dereferenceable(1) %func)
  call void @nothing(%"class.Eigen::internal::redux_evaluator"* nonnull %thisEval) #12
  ret double %call5
}

; Function Attrs: nounwind
declare dso_local i32 @fprintf(%struct._IO_FILE* nocapture, i8* nocapture readonly, ...) local_unnamed_addr #6

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local dereferenceable(8) double* @_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEELi1EEclEll(%"class.Eigen::DenseCoeffsBase.0"* %this, i64 %row, i64 %col) local_unnamed_addr #3 comdat align 2 {
entry:
  %cmp = icmp sgt i64 %row, -1
  br i1 %cmp, label %land.lhs.true, label %cond.false

land.lhs.true:                                    ; preds = %entry
  %0 = bitcast %"class.Eigen::DenseCoeffsBase.0"* %this to %"struct.Eigen::EigenBase"*
  %call = tail call i64 @_ZNK5Eigen9EigenBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE4rowsEv(%"struct.Eigen::EigenBase"* %0)
  %cmp2 = icmp sgt i64 %call, %row
  %cmp4 = icmp sgt i64 %col, -1
  %or.cond = and i1 %cmp4, %cmp2
  br i1 %or.cond, label %land.lhs.true5, label %cond.false

land.lhs.true5:                                   ; preds = %land.lhs.true
  %call6 = tail call i64 @_ZNK5Eigen9EigenBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE4colsEv(%"struct.Eigen::EigenBase"* %0)
  %cmp7 = icmp sgt i64 %call6, %col
  br i1 %cmp7, label %cond.end, label %cond.false

cond.false:                                       ; preds = %land.lhs.true5, %land.lhs.true, %entry
  tail call void @__assert_fail(i8* getelementptr inbounds ([53 x i8], [53 x i8]* @.str.19, i64 0, i64 0), i8* getelementptr inbounds ([59 x i8], [59 x i8]* @.str.20, i64 0, i64 0), i32 365, i8* getelementptr inbounds ([219 x i8], [219 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEELi1EEclEll, i64 0, i64 0)) #14
  unreachable

cond.end:                                         ; preds = %land.lhs.true5
  %call8 = tail call dereferenceable(8) double* @_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEELi1EE8coeffRefEll(%"class.Eigen::DenseCoeffsBase.0"* %this, i64 %row, i64 %col)
  ret double* %call8
}

; Function Attrs: nounwind readnone speculatable
declare double @llvm.fabs.f64(double) #7

; Function Attrs: noreturn nounwind
declare dso_local void @abort() local_unnamed_addr #8

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local double @_ZNK5Eigen9DenseBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE3sumEv(%"class.Eigen::DenseBase"* %this) local_unnamed_addr #2 comdat align 2 {
entry:
  %ref.tmp = alloca %"struct.Eigen::internal::scalar_sum_op", align 1
  %0 = bitcast %"class.Eigen::DenseBase"* %this to %"struct.Eigen::EigenBase"*
  %call = tail call dereferenceable(32) %"class.Eigen::Matrix"* @_ZNK5Eigen9EigenBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE7derivedEv(%"struct.Eigen::EigenBase"* %0)
  %1 = bitcast %"class.Eigen::Matrix"* %call to %"class.Eigen::DenseBase"*
  %2 = getelementptr inbounds %"struct.Eigen::internal::scalar_sum_op", %"struct.Eigen::internal::scalar_sum_op"* %ref.tmp, i64 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %2) #12
  call void @_ZN5Eigen8internal13scalar_sum_opIddEC2Ev(%"struct.Eigen::internal::scalar_sum_op"* nonnull %ref.tmp)
  %call2 = call double @_ZNK5Eigen9DenseBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE5reduxINS_8internal13scalar_sum_opIddEEEEdRKT_(%"class.Eigen::DenseBase"* nonnull %1, %"struct.Eigen::internal::scalar_sum_op"* nonnull dereferenceable(1) %ref.tmp)
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %2) #12
  ret double %call2
}

; Function Attrs: norecurse nounwind uwtable
define linkonce_odr dso_local dereferenceable(32) %"class.Eigen::Matrix"* @_ZNK5Eigen9EigenBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE7derivedEv(%"struct.Eigen::EigenBase"* %this) local_unnamed_addr #0 comdat align 2 {
entry:
  %0 = bitcast %"struct.Eigen::EigenBase"* %this to %"class.Eigen::Matrix"*
  ret %"class.Eigen::Matrix"* %0
}
; herenow
; Function Attrs: nounwind uwtable
define linkonce_odr dso_local double @_ZNK5Eigen9DenseBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE5reduxINS_8internal13scalar_sum_opIddEEEEdRKT_(%"class.Eigen::DenseBase"* %this, %"struct.Eigen::internal::scalar_sum_op"* dereferenceable(1) %func) local_unnamed_addr #2 comdat align 2 {
entry:
  %thisEval = alloca %"class.Eigen::internal::redux_evaluator", align 8
  %b0 = bitcast %"class.Eigen::DenseBase"* %this to %"struct.Eigen::EigenBase"*
  %call = tail call i64 @_ZNK5Eigen9EigenBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE4rowsEv(%"struct.Eigen::EigenBase"* %b0)
  %call4 = tail call dereferenceable(32) %"class.Eigen::Matrix"* @_ZNK5Eigen9EigenBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE7derivedEv(%"struct.Eigen::EigenBase"* %b0)
  call void @_ZN5Eigen8internal15redux_evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEC2ERKS3_(%"class.Eigen::internal::redux_evaluator"* nonnull %thisEval, %"class.Eigen::Matrix"* nonnull dereferenceable(32) %call4)
  %call5 = call double @_ZN5Eigen8internal10redux_implINS0_13scalar_sum_opIddEENS0_15redux_evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEELi3ELi2EE3runERKS7_RKS3_(%"class.Eigen::internal::redux_evaluator"* nonnull dereferenceable(24) %thisEval, %"struct.Eigen::internal::scalar_sum_op"* nonnull dereferenceable(1) %func)
  call void @nothing(%"class.Eigen::internal::redux_evaluator"* nonnull %thisEval) #12
  ret double %call5
}

define linkonce_odr dso_local void @nothing(%"class.Eigen::internal::redux_evaluator"* %this) {
entry:
  ret void
}


; Function Attrs: inlinehint norecurse nounwind uwtable
define linkonce_odr dso_local void @_ZN5Eigen8internal13scalar_sum_opIddEC2Ev(%"struct.Eigen::internal::scalar_sum_op"* %this) unnamed_addr #9 comdat align 2 {
entry:
  ret void
}

; Function Attrs: inlinehint norecurse nounwind uwtable
define linkonce_odr dso_local i64 @_ZNK5Eigen9EigenBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE4rowsEv(%"struct.Eigen::EigenBase"* %this) local_unnamed_addr #9 comdat align 2 {
entry:
  %call = tail call dereferenceable(32) %"class.Eigen::Matrix"* @_ZNK5Eigen9EigenBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE7derivedEv(%"struct.Eigen::EigenBase"* %this)
  %0 = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %call, i64 0, i32 0
  %call2 = tail call i64 @_ZNK5Eigen15PlainObjectBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE4rowsEv(%"class.Eigen::PlainObjectBase"* nonnull %0)
  ret i64 %call2
}

; Function Attrs: inlinehint norecurse nounwind uwtable
define linkonce_odr dso_local i64 @_ZNK5Eigen9EigenBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE4colsEv(%"struct.Eigen::EigenBase"* %this) local_unnamed_addr #9 comdat align 2 {
entry:
  %call = tail call dereferenceable(32) %"class.Eigen::Matrix"* @_ZNK5Eigen9EigenBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE7derivedEv(%"struct.Eigen::EigenBase"* %this)
  %0 = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %call, i64 0, i32 0
  %call2 = tail call i64 @_ZNK5Eigen15PlainObjectBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE4colsEv(%"class.Eigen::PlainObjectBase"* nonnull %0)
  ret i64 %call2
}

; Function Attrs: noreturn nounwind
declare dso_local void @__assert_fail(i8*, i8*, i32, i8*) local_unnamed_addr #8

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZN5Eigen8internal15redux_evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEC2ERKS3_(%"class.Eigen::internal::redux_evaluator"* %this, %"class.Eigen::Matrix"* dereferenceable(32) %xpr) unnamed_addr #2 comdat align 2 {
entry:
  %m_evaluator = getelementptr inbounds %"class.Eigen::internal::redux_evaluator", %"class.Eigen::internal::redux_evaluator"* %this, i64 0, i32 0
  tail call void @_ZN5Eigen8internal9evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEC2ERKS3_(%"struct.Eigen::internal::evaluator"* %m_evaluator, %"class.Eigen::Matrix"* nonnull dereferenceable(32) %xpr)
  %m_xpr = getelementptr inbounds %"class.Eigen::internal::redux_evaluator", %"class.Eigen::internal::redux_evaluator"* %this, i64 0, i32 1
  store %"class.Eigen::Matrix"* %xpr, %"class.Eigen::Matrix"** %m_xpr, align 8, !tbaa !6
  ret void
}
; there
; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local double @_ZN5Eigen8internal10redux_implINS0_13scalar_sum_opIddEENS0_15redux_evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEELi3ELi2EE3runERKS7_RKS3_(%"class.Eigen::internal::redux_evaluator"* dereferenceable(24) %mat, %"struct.Eigen::internal::scalar_sum_op"* dereferenceable(1) %func) local_unnamed_addr #3 comdat align 2 {
entry:
  %m_data = bitcast %"class.Eigen::internal::redux_evaluator"* %mat to <2 x double>**
  %from = load <2 x double>*, <2 x double>** %m_data, align 8, !tbaa !9
  %call3 = load <2 x double>, <2 x double>* %from, align 16, !tbaa !8
  %vecext.i = extractelement <2 x double> %call3, i32 1
  %vecext1.i = extractelement <2 x double> %call3, i32 0
  %add.i = fadd double %vecext1.i, %vecext.i
  ret double %add.i
}

; Function Attrs: inlinehint norecurse nounwind uwtable
define linkonce_odr dso_local void @_ZN5Eigen8internal15redux_evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEED2Ev(%"class.Eigen::internal::redux_evaluator"* %this) unnamed_addr #9 comdat align 2 {
entry:
  %0 = bitcast %"class.Eigen::internal::redux_evaluator"* %this to %"class.Eigen::internal::noncopyable"*
  ; tail call void @_ZN5Eigen8internal11noncopyableD2Ev(%"class.Eigen::internal::noncopyable"* %0) #12
  ret void
}

; Function Attrs: inlinehint norecurse nounwind uwtable
define linkonce_odr dso_local i64 @_ZNK5Eigen15PlainObjectBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE4rowsEv(%"class.Eigen::PlainObjectBase"* %this) local_unnamed_addr #9 comdat align 2 {
entry:
  %call = tail call i64 @_ZN5Eigen12DenseStorageIdLi4ELi2ELi2ELi0EE4rowsEv()
  ret i64 %call
}

; Function Attrs: norecurse nounwind uwtable
define linkonce_odr dso_local i64 @_ZN5Eigen12DenseStorageIdLi4ELi2ELi2ELi0EE4rowsEv() local_unnamed_addr #0 comdat align 2 {
entry:
  ret i64 2
}

; Function Attrs: inlinehint norecurse nounwind uwtable
define linkonce_odr dso_local i64 @_ZNK5Eigen15PlainObjectBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE4colsEv(%"class.Eigen::PlainObjectBase"* %this) local_unnamed_addr #9 comdat align 2 {
entry:
  %call = tail call i64 @_ZN5Eigen12DenseStorageIdLi4ELi2ELi2ELi0EE4colsEv()
  ret i64 %call
}

; Function Attrs: norecurse nounwind uwtable
define linkonce_odr dso_local i64 @_ZN5Eigen12DenseStorageIdLi4ELi2ELi2ELi0EE4colsEv() local_unnamed_addr #0 comdat align 2 {
entry:
  ret i64 2
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZN5Eigen8internal9evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEC2ERKS3_(%"struct.Eigen::internal::evaluator"* %this, %"class.Eigen::Matrix"* dereferenceable(32) %m) unnamed_addr #2 comdat align 2 {
entry:
  %0 = bitcast %"struct.Eigen::internal::evaluator"* %this to %"struct.Eigen::internal::evaluator.6"*
  %1 = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %m, i64 0, i32 0
  tail call void @_ZN5Eigen8internal9evaluatorINS_15PlainObjectBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEEEC2ERKS5_(%"struct.Eigen::internal::evaluator.6"* %0, %"class.Eigen::PlainObjectBase"* nonnull dereferenceable(32) %1)
  ret void
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZN5Eigen8internal9evaluatorINS_15PlainObjectBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEEEC2ERKS5_(%"struct.Eigen::internal::evaluator.6"* %this, %"class.Eigen::PlainObjectBase"* dereferenceable(32) %m) unnamed_addr #2 comdat align 2 {
entry:
  %0 = bitcast %"struct.Eigen::internal::evaluator.6"* %this to %"struct.Eigen::internal::evaluator_base"*
  tail call void @_ZN5Eigen8internal14evaluator_baseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEC2Ev(%"struct.Eigen::internal::evaluator_base"* %0)
  %m_data = getelementptr inbounds %"struct.Eigen::internal::evaluator.6", %"struct.Eigen::internal::evaluator.6"* %this, i64 0, i32 0
  %call = tail call double* @_ZNK5Eigen15PlainObjectBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE4dataEv(%"class.Eigen::PlainObjectBase"* nonnull %m)
  store double* %call, double** %m_data, align 8, !tbaa !9
  %m_outerStride = getelementptr inbounds %"struct.Eigen::internal::evaluator.6", %"struct.Eigen::internal::evaluator.6"* %this, i64 0, i32 1
  %1 = bitcast %"class.Eigen::PlainObjectBase"* %m to %"class.Eigen::DenseCoeffsBase"*
  %call2 = tail call i64 @_ZNK5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEELi3EE11outerStrideEv(%"class.Eigen::DenseCoeffsBase"* nonnull %1)
  tail call void @_ZN5Eigen8internal19variable_if_dynamicIlLi2EEC2El(%"class.Eigen::internal::variable_if_dynamic"* nonnull %m_outerStride, i64 %call2)
  ret void
}

; Function Attrs: inlinehint norecurse nounwind uwtable
define linkonce_odr dso_local void @_ZN5Eigen8internal14evaluator_baseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEC2Ev(%"struct.Eigen::internal::evaluator_base"* %this) unnamed_addr #9 comdat align 2 {
entry:
  %0 = bitcast %"struct.Eigen::internal::evaluator_base"* %this to %"class.Eigen::internal::noncopyable"*
  tail call void @_ZN5Eigen8internal11noncopyableC2Ev(%"class.Eigen::internal::noncopyable"* %0)
  ret void
}

; Function Attrs: inlinehint norecurse nounwind uwtable
define linkonce_odr dso_local double* @_ZNK5Eigen15PlainObjectBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE4dataEv(%"class.Eigen::PlainObjectBase"* %this) local_unnamed_addr #9 comdat align 2 {
entry:
  %m_storage = getelementptr inbounds %"class.Eigen::PlainObjectBase", %"class.Eigen::PlainObjectBase"* %this, i64 0, i32 0
  %call = tail call double* @_ZNK5Eigen12DenseStorageIdLi4ELi2ELi2ELi0EE4dataEv(%"class.Eigen::DenseStorage"* %m_storage)
  ret double* %call
}

; Function Attrs: inlinehint norecurse nounwind uwtable
define linkonce_odr dso_local i64 @_ZNK5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEELi3EE11outerStrideEv(%"class.Eigen::DenseCoeffsBase"* %this) local_unnamed_addr #9 comdat align 2 {
entry:
  %0 = bitcast %"class.Eigen::DenseCoeffsBase"* %this to %"struct.Eigen::EigenBase"*
  %call = tail call dereferenceable(32) %"class.Eigen::Matrix"* @_ZNK5Eigen9EigenBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE7derivedEv(%"struct.Eigen::EigenBase"* %0)
  %call2 = tail call i64 @_ZNK5Eigen6MatrixIdLi2ELi2ELi0ELi2ELi2EE11outerStrideEv(%"class.Eigen::Matrix"* nonnull %call)
  ret i64 %call2
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local void @_ZN5Eigen8internal19variable_if_dynamicIlLi2EEC2El(%"class.Eigen::internal::variable_if_dynamic"* %this, i64 %v) unnamed_addr #3 comdat align 2 {
entry:
  %cmp = icmp eq i64 %v, 2
  br i1 %cmp, label %cond.end, label %cond.false

cond.false:                                       ; preds = %entry
  tail call void @__assert_fail(i8* getelementptr inbounds ([14 x i8], [14 x i8]* @.str.8, i64 0, i64 0), i8* getelementptr inbounds ([58 x i8], [58 x i8]* @.str.9, i64 0, i64 0), i32 110, i8* getelementptr inbounds ([92 x i8], [92 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen8internal19variable_if_dynamicIlLi2EEC2El, i64 0, i64 0)) #14
  unreachable

cond.end:                                         ; preds = %entry
  ret void
}

; Function Attrs: norecurse nounwind uwtable
define linkonce_odr dso_local void @_ZN5Eigen8internal11noncopyableC2Ev(%"class.Eigen::internal::noncopyable"* %this) unnamed_addr #0 comdat align 2 {
entry:
  ret void
}

; Function Attrs: norecurse nounwind uwtable
define linkonce_odr dso_local double* @_ZNK5Eigen12DenseStorageIdLi4ELi2ELi2ELi0EE4dataEv(%"class.Eigen::DenseStorage"* %this) local_unnamed_addr #0 comdat align 2 {
entry:
  %arraydecay = getelementptr inbounds %"class.Eigen::DenseStorage", %"class.Eigen::DenseStorage"* %this, i64 0, i32 0, i32 0, i64 0
  ret double* %arraydecay
}

; Function Attrs: inlinehint norecurse nounwind uwtable
define linkonce_odr dso_local i64 @_ZNK5Eigen6MatrixIdLi2ELi2ELi0ELi2ELi2EE11outerStrideEv(%"class.Eigen::Matrix"* %this) local_unnamed_addr #9 comdat align 2 {
entry:
  %0 = bitcast %"class.Eigen::Matrix"* %this to %"class.Eigen::DenseBase"*
  %call = tail call i64 @_ZNK5Eigen9DenseBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE9innerSizeEv(%"class.Eigen::DenseBase"* %0)
  ret i64 %call
}

; Function Attrs: norecurse nounwind uwtable
define linkonce_odr dso_local i64 @_ZNK5Eigen9DenseBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE9innerSizeEv(%"class.Eigen::DenseBase"* %this) local_unnamed_addr #0 comdat align 2 {
entry:
  %0 = bitcast %"class.Eigen::DenseBase"* %this to %"struct.Eigen::EigenBase"*
  %call = tail call i64 @_ZNK5Eigen9EigenBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE4rowsEv(%"struct.Eigen::EigenBase"* %0)
  ret i64 %call
}

; Function Attrs: norecurse nounwind uwtable
define linkonce_odr dso_local i64 @_ZNK5Eigen8internal15redux_evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE4rowsEv(%"class.Eigen::internal::redux_evaluator"* %this) local_unnamed_addr #0 comdat align 2 {
entry:
  %m_xpr = getelementptr inbounds %"class.Eigen::internal::redux_evaluator", %"class.Eigen::internal::redux_evaluator"* %this, i64 0, i32 1
  %0 = bitcast %"class.Eigen::Matrix"** %m_xpr to %"class.Eigen::PlainObjectBase"**
  %1 = load %"class.Eigen::PlainObjectBase"*, %"class.Eigen::PlainObjectBase"** %0, align 8, !tbaa !12
  %call = tail call i64 @_ZNK5Eigen15PlainObjectBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE4rowsEv(%"class.Eigen::PlainObjectBase"* %1)
  ret i64 %call
}

; Function Attrs: norecurse nounwind uwtable
define linkonce_odr dso_local i64 @_ZNK5Eigen8internal15redux_evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE4colsEv(%"class.Eigen::internal::redux_evaluator"* %this) local_unnamed_addr #0 comdat align 2 {
entry:
  %m_xpr = getelementptr inbounds %"class.Eigen::internal::redux_evaluator", %"class.Eigen::internal::redux_evaluator"* %this, i64 0, i32 1
  %0 = bitcast %"class.Eigen::Matrix"** %m_xpr to %"class.Eigen::PlainObjectBase"**
  %1 = load %"class.Eigen::PlainObjectBase"*, %"class.Eigen::PlainObjectBase"** %0, align 8, !tbaa !12
  %call = tail call i64 @_ZNK5Eigen15PlainObjectBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE4colsEv(%"class.Eigen::PlainObjectBase"* %1)
  ret i64 %call
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local double @_ZNK5Eigen8internal13scalar_sum_opIddE6preduxIDv2_dEEKdRKT_(%"struct.Eigen::internal::scalar_sum_op"* %this, <2 x double>* dereferenceable(16) %a) local_unnamed_addr #3 comdat align 2 {
entry:
  %call = tail call double @_ZN5Eigen8internal6preduxIDv2_dEENS0_15unpacket_traitsIT_E4typeERKS4_(<2 x double>* nonnull dereferenceable(16) %a)
  ret double %call
}
; zzzz
; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local <2 x double> @_ZN5Eigen8internal18redux_vec_unrollerINS0_13scalar_sum_opIddEENS0_15redux_evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEELi0ELi2EE3runERKS7_RKS3_(%"class.Eigen::internal::redux_evaluator"* dereferenceable(24) %mat, %"struct.Eigen::internal::scalar_sum_op"* dereferenceable(1) %func) local_unnamed_addr #3 comdat align 2 {
entry:
  %e0 = bitcast %"class.Eigen::internal::redux_evaluator"* %mat to %"struct.Eigen::internal::evaluator.6"*
  %m_data = getelementptr inbounds %"struct.Eigen::internal::evaluator.6", %"struct.Eigen::internal::evaluator.6"* %e0, i64 0, i32 0
  %from = load double*, double** %m_data, align 8, !tbaa !9
  %0 = bitcast double* %from to <2 x double>*
  %1 = load <2 x double>, <2 x double>* %0, align 16, !tbaa !8
  ret <2 x double> %1
}

; continue

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local double @_ZN5Eigen8internal6preduxIDv2_dEENS0_15unpacket_traitsIT_E4typeERKS4_(<2 x double>* dereferenceable(16) %a) local_unnamed_addr #10 comdat {
entry:
  %z1 = load <2 x double>, <2 x double>* %a, align 16, !tbaa !8
  %vecext.i = extractelement <2 x double> %z1, i32 1
  %vecext1.i = extractelement <2 x double> %z1, i32 0
  %add.i = fadd double %vecext1.i, %vecext.i
  ret double %add.i
}

; Function Attrs: inlinehint norecurse nounwind uwtable
define linkonce_odr dso_local double @_ZN5Eigen8internal6pfirstIDv2_dEENS0_15unpacket_traitsIT_E4typeERKS4_(<2 x double>* dereferenceable(16) %a) local_unnamed_addr #11 comdat {
entry:
  %0 = load <2 x double>, <2 x double>* %a, align 16, !tbaa !8
  %vecext.i = extractelement <2 x double> %0, i32 0
  ret double %vecext.i
}

; Function Attrs: inlinehint norecurse nounwind uwtable
define linkonce_odr dso_local <2 x double> @_ZNK5Eigen8internal13scalar_sum_opIddE8packetOpIDv2_dEEKT_RS6_S7_(%"struct.Eigen::internal::scalar_sum_op"* %this, <2 x double>* dereferenceable(16) %a, <2 x double>* dereferenceable(16) %b) local_unnamed_addr #9 comdat align 2 {
entry:
  %call = tail call <2 x double> @_ZN5Eigen8internal4paddIDv2_dEET_RKS3_S5_(<2 x double>* nonnull dereferenceable(16) %a, <2 x double>* nonnull dereferenceable(16) %b)
  ret <2 x double> %call
}

; Function Attrs: inlinehint norecurse nounwind uwtable
define linkonce_odr dso_local <2 x double> @_ZN5Eigen8internal18redux_vec_unrollerINS0_13scalar_sum_opIddEENS0_15redux_evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEELi0ELi1EE3runERKS7_RKS3_(%"class.Eigen::internal::redux_evaluator"* dereferenceable(24) %mat, %"struct.Eigen::internal::scalar_sum_op"* dereferenceable(1)) local_unnamed_addr #9 comdat align 2 {
entry:
  %call = tail call <2 x double> @_ZNK5Eigen8internal15redux_evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE18packetByOuterInnerILi16EDv2_dEET0_ll(%"class.Eigen::internal::redux_evaluator"* nonnull %mat, i64 0, i64 0)
  ret <2 x double> %call
}

; Function Attrs: inlinehint norecurse nounwind uwtable
define linkonce_odr dso_local <2 x double> @_ZN5Eigen8internal18redux_vec_unrollerINS0_13scalar_sum_opIddEENS0_15redux_evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEELi1ELi1EE3runERKS7_RKS3_(%"class.Eigen::internal::redux_evaluator"* dereferenceable(24) %mat, %"struct.Eigen::internal::scalar_sum_op"* dereferenceable(1)) local_unnamed_addr #9 comdat align 2 {
entry:
  %call = tail call <2 x double> @_ZNK5Eigen8internal15redux_evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE18packetByOuterInnerILi16EDv2_dEET0_ll(%"class.Eigen::internal::redux_evaluator"* nonnull %mat, i64 1, i64 0)
  ret <2 x double> %call
}

; Function Attrs: inlinehint norecurse nounwind uwtable
define linkonce_odr dso_local <2 x double> @_ZN5Eigen8internal4paddIDv2_dEET_RKS3_S5_(<2 x double>* dereferenceable(16) %a, <2 x double>* dereferenceable(16) %b) local_unnamed_addr #11 comdat {
entry:
  %0 = load <2 x double>, <2 x double>* %a, align 16, !tbaa !8
  %1 = load <2 x double>, <2 x double>* %b, align 16, !tbaa !8
  %add.i = fadd <2 x double> %0, %1
  ret <2 x double> %add.i
}

; Function Attrs: norecurse nounwind uwtable
define linkonce_odr dso_local <2 x double> @_ZNK5Eigen8internal15redux_evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE18packetByOuterInnerILi16EDv2_dEET0_ll(%"class.Eigen::internal::redux_evaluator"* %this, i64 %outer, i64 %inner) local_unnamed_addr #0 comdat align 2 {
entry:
  %r = bitcast %"class.Eigen::internal::redux_evaluator"* %this to %"struct.Eigen::internal::evaluator.6"*
  %m_data = getelementptr inbounds %"struct.Eigen::internal::evaluator.6", %"struct.Eigen::internal::evaluator.6"* %r, i64 0, i32 0
  %q0 = load double*, double** %m_data, align 8, !tbaa !9
  %0 = bitcast double* %q0 to <2 x double>*
  %1 = load <2 x double>, <2 x double>* %0, align 16, !tbaa !8
  ret <2 x double> %1
}

; Function Attrs: inlinehint norecurse nounwind uwtable
define linkonce_odr dso_local <2 x double> @_ZNK5Eigen8internal9evaluatorINS_15PlainObjectBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEEE6packetILi16EDv2_dEET0_ll(%"struct.Eigen::internal::evaluator.6"* %this, i64 %row, i64 %col) local_unnamed_addr #9 comdat align 2 {
entry:
  %m_data = getelementptr inbounds %"struct.Eigen::internal::evaluator.6", %"struct.Eigen::internal::evaluator.6"* %this, i64 0, i32 0
  %0 = load double*, double** %m_data, align 8, !tbaa !9
  %add.ptr = getelementptr inbounds double, double* %0, i64 %row
  %call = tail call i64 @_ZN5Eigen8internal19variable_if_dynamicIlLi2EE5valueEv()
  %mul = mul nsw i64 %call, %col
  %add.ptr2 = getelementptr inbounds double, double* %add.ptr, i64 %mul
  %call.i = tail call <2 x double> @_ZN5Eigen8internal5ploadIDv2_dEET_PKNS0_15unpacket_traitsIS3_E4typeE(double* %add.ptr2) #12
  ret <2 x double> %call.i
}

; Function Attrs: inlinehint norecurse nounwind uwtable
define linkonce_odr dso_local i64 @_ZN5Eigen8internal19variable_if_dynamicIlLi2EE5valueEv() local_unnamed_addr #9 comdat align 2 {
entry:
  ret i64 2
}

; Function Attrs: inlinehint norecurse nounwind uwtable
define linkonce_odr dso_local <2 x double> @_ZN5Eigen8internal5ploadIDv2_dEET_PKNS0_15unpacket_traitsIS3_E4typeE(double* %from) local_unnamed_addr #11 comdat {
entry:
  %0 = bitcast double* %from to <2 x double>*
  %1 = load <2 x double>, <2 x double>* %0, align 16, !tbaa !8
  ret <2 x double> %1
}

; Function Attrs: norecurse nounwind uwtable
define linkonce_odr dso_local void @_ZN5Eigen8internal11noncopyableD2Ev(%"class.Eigen::internal::noncopyable"* %this) unnamed_addr #0 comdat align 2 {
entry:
  ret void
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZN5Eigen9DenseBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE11NullaryExprINS_8internal18scalar_constant_opIdEEEEKNS_14CwiseNullaryOpIT_S2_EEllRKS9_(%"class.Eigen::CwiseNullaryOp"* noalias sret %agg.result, i64 %rows, i64 %cols, %"struct.Eigen::internal::scalar_constant_op"* dereferenceable(8) %func) local_unnamed_addr #2 comdat align 2 {
entry:
  tail call void @_ZN5Eigen14CwiseNullaryOpINS_8internal18scalar_constant_opIdEENS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEC2EllRKS3_(%"class.Eigen::CwiseNullaryOp"* %agg.result, i64 %rows, i64 %cols, %"struct.Eigen::internal::scalar_constant_op"* nonnull dereferenceable(8) %func)
  ret void
}

; Function Attrs: inlinehint norecurse nounwind uwtable
define linkonce_odr dso_local void @_ZN5Eigen8internal18scalar_constant_opIdEC2ERKd(%"struct.Eigen::internal::scalar_constant_op"* %this, double* dereferenceable(8) %other) unnamed_addr #9 comdat align 2 {
entry:
  %0 = bitcast double* %other to i64*
  %1 = load i64, i64* %0, align 8, !tbaa !2
  %2 = bitcast %"struct.Eigen::internal::scalar_constant_op"* %this to i64*
  store i64 %1, i64* %2, align 8, !tbaa !15
  ret void
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZN5Eigen14CwiseNullaryOpINS_8internal18scalar_constant_opIdEENS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEC2EllRKS3_(%"class.Eigen::CwiseNullaryOp"* %this, i64 %rows, i64 %cols, %"struct.Eigen::internal::scalar_constant_op"* dereferenceable(8) %func) unnamed_addr #2 comdat align 2 {
entry:
  %0 = bitcast %"class.Eigen::CwiseNullaryOp"* %this to %"class.Eigen::MatrixBase.2"*
  tail call void @_ZN5Eigen10MatrixBaseINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEENS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEEEC2Ev(%"class.Eigen::MatrixBase.2"* %0)
  %m_rows = getelementptr inbounds %"class.Eigen::CwiseNullaryOp", %"class.Eigen::CwiseNullaryOp"* %this, i64 0, i32 0
  tail call void @_ZN5Eigen8internal19variable_if_dynamicIlLi2EEC2El(%"class.Eigen::internal::variable_if_dynamic"* %m_rows, i64 %rows)
  %m_cols = getelementptr inbounds %"class.Eigen::CwiseNullaryOp", %"class.Eigen::CwiseNullaryOp"* %this, i64 0, i32 1
  tail call void @_ZN5Eigen8internal19variable_if_dynamicIlLi2EEC2El(%"class.Eigen::internal::variable_if_dynamic"* nonnull %m_cols, i64 %cols)
  %m_functor = getelementptr inbounds %"class.Eigen::CwiseNullaryOp", %"class.Eigen::CwiseNullaryOp"* %this, i64 0, i32 2
  tail call void @_ZN5Eigen8internal18scalar_constant_opIdEC2ERKS2_(%"struct.Eigen::internal::scalar_constant_op"* nonnull %m_functor, %"struct.Eigen::internal::scalar_constant_op"* nonnull dereferenceable(8) %func)
  %cmp2 = icmp eq i64 %rows, 2
  %cmp6 = icmp eq i64 %cols, 2
  %or.cond8 = and i1 %cmp2, %cmp6
  br i1 %or.cond8, label %cond.end, label %cond.false

cond.false:                                       ; preds = %entry
  tail call void @__assert_fail(i8* getelementptr inbounds ([149 x i8], [149 x i8]* @.str.11, i64 0, i64 0), i8* getelementptr inbounds ([58 x i8], [58 x i8]* @.str.12, i64 0, i64 0), i32 74, i8* getelementptr inbounds ([278 x i8], [278 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen14CwiseNullaryOpINS_8internal18scalar_constant_opIdEENS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEC2EllRKS3_, i64 0, i64 0)) #14
  unreachable

cond.end:                                         ; preds = %entry
  ret void
}

; Function Attrs: norecurse nounwind uwtable
define linkonce_odr dso_local void @_ZN5Eigen10MatrixBaseINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEENS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEEEC2Ev(%"class.Eigen::MatrixBase.2"* %this) unnamed_addr #0 comdat align 2 {
entry:
  %0 = bitcast %"class.Eigen::MatrixBase.2"* %this to %"class.Eigen::DenseBase.3"*
  tail call void @_ZN5Eigen9DenseBaseINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEENS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEEEC2Ev(%"class.Eigen::DenseBase.3"* %0)
  ret void
}

; Function Attrs: inlinehint norecurse nounwind uwtable
define linkonce_odr dso_local void @_ZN5Eigen8internal18scalar_constant_opIdEC2ERKS2_(%"struct.Eigen::internal::scalar_constant_op"* %this, %"struct.Eigen::internal::scalar_constant_op"* dereferenceable(8) %other) unnamed_addr #9 comdat align 2 {
entry:
  %0 = bitcast %"struct.Eigen::internal::scalar_constant_op"* %other to i64*
  %1 = load i64, i64* %0, align 8, !tbaa !15
  %2 = bitcast %"struct.Eigen::internal::scalar_constant_op"* %this to i64*
  store i64 %1, i64* %2, align 8, !tbaa !15
  ret void
}

; Function Attrs: norecurse nounwind uwtable
define linkonce_odr dso_local void @_ZN5Eigen9DenseBaseINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEENS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEEEC2Ev(%"class.Eigen::DenseBase.3"* %this) unnamed_addr #0 comdat align 2 {
entry:
  ret void
}

; Function Attrs: norecurse nounwind uwtable
define linkonce_odr dso_local dereferenceable(16) %"class.Eigen::CwiseNullaryOp"* @_ZNK5Eigen9EigenBaseINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEENS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEEE7derivedEv(%"struct.Eigen::EigenBase.5"* %this) local_unnamed_addr #0 comdat align 2 {
entry:
  %0 = bitcast %"struct.Eigen::EigenBase.5"* %this to %"class.Eigen::CwiseNullaryOp"*
  ret %"class.Eigen::CwiseNullaryOp"* %0
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local void @_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEC2INS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEERKNS_9DenseBaseIT_EE(%"class.Eigen::PlainObjectBase"* %this, %"class.Eigen::DenseBase.3"* dereferenceable(1) %other) unnamed_addr #3 comdat align 2 {
entry:
  %0 = bitcast %"class.Eigen::PlainObjectBase"* %this to %"class.Eigen::MatrixBase"*
  tail call void @_ZN5Eigen10MatrixBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEC2Ev(%"class.Eigen::MatrixBase"* %0)
  %m_storage = getelementptr inbounds %"class.Eigen::PlainObjectBase", %"class.Eigen::PlainObjectBase"* %this, i64 0, i32 0
  tail call void @_ZN5Eigen12DenseStorageIdLi4ELi2ELi2ELi0EEC2Ev(%"class.Eigen::DenseStorage"* %m_storage)
  tail call void @_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE22_check_template_paramsEv()
  %1 = bitcast %"class.Eigen::DenseBase.3"* %other to %"struct.Eigen::EigenBase.5"*
  tail call void @_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE10resizeLikeINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEEvRKNS_9EigenBaseIT_EE(%"class.Eigen::PlainObjectBase"* %this, %"struct.Eigen::EigenBase.5"* nonnull dereferenceable(1) %1)
  %call = tail call dereferenceable(32) %"class.Eigen::Matrix"* @_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE12_set_noaliasINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEERS2_RKNS_9DenseBaseIT_EE(%"class.Eigen::PlainObjectBase"* %this, %"class.Eigen::DenseBase.3"* nonnull dereferenceable(1) %other)
  ret void
}

; Function Attrs: norecurse nounwind uwtable
define linkonce_odr dso_local void @_ZN5Eigen10MatrixBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEC2Ev(%"class.Eigen::MatrixBase"* %this) unnamed_addr #0 comdat align 2 {
entry:
  %0 = bitcast %"class.Eigen::MatrixBase"* %this to %"class.Eigen::DenseBase"*
  tail call void @_ZN5Eigen9DenseBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEC2Ev(%"class.Eigen::DenseBase"* %0)
  ret void
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZN5Eigen12DenseStorageIdLi4ELi2ELi2ELi0EEC2Ev(%"class.Eigen::DenseStorage"* %this) unnamed_addr #2 comdat align 2 {
entry:
  %m_data = getelementptr inbounds %"class.Eigen::DenseStorage", %"class.Eigen::DenseStorage"* %this, i64 0, i32 0
  tail call void @_ZN5Eigen8internal11plain_arrayIdLi4ELi0ELi16EEC2Ev(%"struct.Eigen::internal::plain_array"* %m_data)
  ret void
}

; Function Attrs: inlinehint norecurse nounwind uwtable
define linkonce_odr dso_local void @_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE22_check_template_paramsEv() local_unnamed_addr #9 comdat align 2 {
entry:
  ret void
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local void @_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE10resizeLikeINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEEvRKNS_9EigenBaseIT_EE(%"class.Eigen::PlainObjectBase"* %this, %"struct.Eigen::EigenBase.5"* dereferenceable(1) %_other) local_unnamed_addr #3 comdat align 2 {
entry:
  %call = tail call dereferenceable(16) %"class.Eigen::CwiseNullaryOp"* @_ZNK5Eigen9EigenBaseINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEENS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEEE7derivedEv(%"struct.Eigen::EigenBase.5"* nonnull %_other)
  %call2 = tail call i64 @_ZNK5Eigen14CwiseNullaryOpINS_8internal18scalar_constant_opIdEENS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE4rowsEv(%"class.Eigen::CwiseNullaryOp"* nonnull %call)
  %call3 = tail call i64 @_ZNK5Eigen14CwiseNullaryOpINS_8internal18scalar_constant_opIdEENS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE4colsEv(%"class.Eigen::CwiseNullaryOp"* nonnull %call)
  %call4 = tail call i64 @_ZNK5Eigen14CwiseNullaryOpINS_8internal18scalar_constant_opIdEENS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE4rowsEv(%"class.Eigen::CwiseNullaryOp"* nonnull %call)
  %call5 = tail call i64 @_ZNK5Eigen14CwiseNullaryOpINS_8internal18scalar_constant_opIdEENS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE4colsEv(%"class.Eigen::CwiseNullaryOp"* nonnull %call)
  %call6 = tail call i64 @_ZNK5Eigen14CwiseNullaryOpINS_8internal18scalar_constant_opIdEENS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE4rowsEv(%"class.Eigen::CwiseNullaryOp"* nonnull %call)
  %call7 = tail call i64 @_ZNK5Eigen14CwiseNullaryOpINS_8internal18scalar_constant_opIdEENS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE4colsEv(%"class.Eigen::CwiseNullaryOp"* nonnull %call)
  tail call void @_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE6resizeEll(%"class.Eigen::PlainObjectBase"* %this, i64 %call6, i64 %call7)
  ret void
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local dereferenceable(32) %"class.Eigen::Matrix"* @_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE12_set_noaliasINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEES2_EEEERS2_RKNS_9DenseBaseIT_EE(%"class.Eigen::PlainObjectBase"* %this, %"class.Eigen::DenseBase.3"* dereferenceable(1) %other) local_unnamed_addr #3 comdat align 2 {
entry:
  %ref.tmp = alloca %"struct.Eigen::internal::assign_op", align 1
  %0 = bitcast %"class.Eigen::PlainObjectBase"* %this to %"struct.Eigen::EigenBase"*
  %call = tail call dereferenceable(32) %"class.Eigen::Matrix"* @_ZN5Eigen9EigenBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE7derivedEv(%"struct.Eigen::EigenBase"* %0)
  %1 = bitcast %"class.Eigen::DenseBase.3"* %other to %"struct.Eigen::EigenBase.5"*
  %call2 = tail call dereferenceable(16) %"class.Eigen::CwiseNullaryOp"* @_ZNK5Eigen9EigenBaseINS_14CwiseNullaryOpINS_8internal18scalar_constant_opIdEENS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEEE7derivedEv(%"struct.Eigen::EigenBase.5"* nonnull %1)
  %2 = getelementptr inbounds %"struct.Eigen::internal::assign_op", %"struct.Eigen::internal::assign_op"* %ref.tmp, i64 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %2) #12
  call void @_ZN5Eigen8internal9assign_opIddEC2Ev(%"struct.Eigen::internal::assign_op"* nonnull %ref.tmp)
  call void @_ZN5Eigen8internal24call_assignment_no_aliasINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EENS0_9assign_opIddEEEEvRT_RKT0_RKT1_(%"class.Eigen::Matrix"* nonnull dereferenceable(32) %call, %"class.Eigen::CwiseNullaryOp"* nonnull dereferenceable(16) %call2, %"struct.Eigen::internal::assign_op"* nonnull dereferenceable(1) %ref.tmp)
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %2) #12
  %call3 = call dereferenceable(32) %"class.Eigen::Matrix"* @_ZN5Eigen9EigenBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE7derivedEv(%"struct.Eigen::EigenBase"* %0)
  ret %"class.Eigen::Matrix"* %call3
}

; Function Attrs: norecurse nounwind uwtable
define linkonce_odr dso_local void @_ZN5Eigen9DenseBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEC2Ev(%"class.Eigen::DenseBase"* %this) unnamed_addr #0 comdat align 2 {
entry:
  ret void
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZN5Eigen8internal11plain_arrayIdLi4ELi0ELi16EEC2Ev(%"struct.Eigen::internal::plain_array"* %this) unnamed_addr #2 comdat align 2 {
entry:
  %0 = ptrtoint %"struct.Eigen::internal::plain_array"* %this to i64
  %and = and i64 %0, 15
  %cmp = icmp eq i64 %and, 0
  br i1 %cmp, label %cond.end, label %cond.false

cond.false:                                       ; preds = %entry
  tail call void @__assert_fail(i8* getelementptr inbounds ([192 x i8], [192 x i8]* @.str.13, i64 0, i64 0), i8* getelementptr inbounds ([56 x i8], [56 x i8]* @.str.14, i64 0, i64 0), i32 109, i8* getelementptr inbounds ([127 x i8], [127 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen8internal11plain_arrayIdLi4ELi0ELi16EEC2Ev, i64 0, i64 0)) #14
  unreachable

cond.end:                                         ; preds = %entry
  tail call void @_ZN5Eigen8internal28check_static_allocation_sizeIdLi4EEEvv()
  ret void
}

; Function Attrs: norecurse nounwind uwtable
define linkonce_odr dso_local void @_ZN5Eigen8internal28check_static_allocation_sizeIdLi4EEEvv() local_unnamed_addr #0 comdat {
entry:
  ret void
}

; Function Attrs: inlinehint norecurse nounwind uwtable
define linkonce_odr dso_local i64 @_ZNK5Eigen14CwiseNullaryOpINS_8internal18scalar_constant_opIdEENS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE4rowsEv(%"class.Eigen::CwiseNullaryOp"* %this) local_unnamed_addr #9 comdat align 2 {
entry:
  %call = tail call i64 @_ZN5Eigen8internal19variable_if_dynamicIlLi2EE5valueEv()
  ret i64 %call
}

; Function Attrs: inlinehint norecurse nounwind uwtable
define linkonce_odr dso_local i64 @_ZNK5Eigen14CwiseNullaryOpINS_8internal18scalar_constant_opIdEENS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE4colsEv(%"class.Eigen::CwiseNullaryOp"* %this) local_unnamed_addr #9 comdat align 2 {
entry:
  %call = tail call i64 @_ZN5Eigen8internal19variable_if_dynamicIlLi2EE5valueEv()
  ret i64 %call
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local void @_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE6resizeEll(%"class.Eigen::PlainObjectBase"* %this, i64 %rows, i64 %cols) local_unnamed_addr #3 comdat align 2 {
entry:
  %cmp = icmp eq i64 %rows, 2
  %cmp2 = icmp eq i64 %cols, 2
  %or.cond = and i1 %cmp, %cmp2
  br i1 %or.cond, label %cond.end, label %cond.false

cond.false:                                       ; preds = %entry
  tail call void @__assert_fail(i8* getelementptr inbounds ([399 x i8], [399 x i8]* @.str.15, i64 0, i64 0), i8* getelementptr inbounds ([59 x i8], [59 x i8]* @.str.16, i64 0, i64 0), i32 285, i8* getelementptr inbounds ([152 x i8], [152 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen15PlainObjectBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE6resizeEll, i64 0, i64 0)) #14
  unreachable

cond.end:                                         ; preds = %entry
  %m_storage = getelementptr inbounds %"class.Eigen::PlainObjectBase", %"class.Eigen::PlainObjectBase"* %this, i64 0, i32 0
  %mul = mul nsw i64 %cols, %rows
  tail call void @_ZN5Eigen12DenseStorageIdLi4ELi2ELi2ELi0EE6resizeElll(%"class.Eigen::DenseStorage"* %m_storage, i64 %mul, i64 %rows, i64 %cols)
  ret void
}

; Function Attrs: norecurse nounwind uwtable
define linkonce_odr dso_local void @_ZN5Eigen12DenseStorageIdLi4ELi2ELi2ELi0EE6resizeElll(%"class.Eigen::DenseStorage"* %this, i64, i64, i64) local_unnamed_addr #0 comdat align 2 {
entry:
  ret void
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local void @_ZN5Eigen8internal24call_assignment_no_aliasINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EENS0_9assign_opIddEEEEvRT_RKT0_RKT1_(%"class.Eigen::Matrix"* dereferenceable(32) %dst, %"class.Eigen::CwiseNullaryOp"* dereferenceable(16) %src, %"struct.Eigen::internal::assign_op"* dereferenceable(1) %func) local_unnamed_addr #3 comdat {
entry:
  tail call void @_ZN5Eigen8internal10AssignmentINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EENS0_9assign_opIddEENS0_11Dense2DenseEvE3runERS3_RKS7_RKS9_(%"class.Eigen::Matrix"* nonnull dereferenceable(32) %dst, %"class.Eigen::CwiseNullaryOp"* nonnull dereferenceable(16) %src, %"struct.Eigen::internal::assign_op"* nonnull dereferenceable(1) %func)
  ret void
}

; Function Attrs: norecurse nounwind uwtable
define linkonce_odr dso_local dereferenceable(32) %"class.Eigen::Matrix"* @_ZN5Eigen9EigenBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE7derivedEv(%"struct.Eigen::EigenBase"* %this) local_unnamed_addr #0 comdat align 2 {
entry:
  %0 = bitcast %"struct.Eigen::EigenBase"* %this to %"class.Eigen::Matrix"*
  ret %"class.Eigen::Matrix"* %0
}

; Function Attrs: inlinehint norecurse nounwind uwtable
define linkonce_odr dso_local void @_ZN5Eigen8internal9assign_opIddEC2Ev(%"struct.Eigen::internal::assign_op"* %this) unnamed_addr #9 comdat align 2 {
entry:
  ret void
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local void @_ZN5Eigen8internal10AssignmentINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EENS0_9assign_opIddEENS0_11Dense2DenseEvE3runERS3_RKS7_RKS9_(%"class.Eigen::Matrix"* dereferenceable(32) %dst, %"class.Eigen::CwiseNullaryOp"* dereferenceable(16) %src, %"struct.Eigen::internal::assign_op"* dereferenceable(1) %func) local_unnamed_addr #3 comdat align 2 {
entry:
  tail call void @_ZN5Eigen8internal18check_for_aliasingINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EEEEvRKT_RKT0_(%"class.Eigen::Matrix"* nonnull dereferenceable(32) %dst, %"class.Eigen::CwiseNullaryOp"* nonnull dereferenceable(16) %src)
  tail call void @_ZN5Eigen8internal26call_dense_assignment_loopINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EENS0_9assign_opIddEEEEvRT_RKT0_RKT1_(%"class.Eigen::Matrix"* nonnull dereferenceable(32) %dst, %"class.Eigen::CwiseNullaryOp"* nonnull dereferenceable(16) %src, %"struct.Eigen::internal::assign_op"* nonnull dereferenceable(1) %func)
  ret void
}

; Function Attrs: norecurse nounwind uwtable
define linkonce_odr dso_local void @_ZN5Eigen8internal18check_for_aliasingINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EEEEvRKT_RKT0_(%"class.Eigen::Matrix"* dereferenceable(32) %dst, %"class.Eigen::CwiseNullaryOp"* dereferenceable(16) %src) local_unnamed_addr #0 comdat {
entry:
  tail call void @_ZN5Eigen8internal27checkTransposeAliasing_implINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EELb0EE3runERKS3_RKS7_(%"class.Eigen::Matrix"* nonnull dereferenceable(32) %dst, %"class.Eigen::CwiseNullaryOp"* nonnull dereferenceable(16) %src)
  ret void
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local void @_ZN5Eigen8internal26call_dense_assignment_loopINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EENS0_9assign_opIddEEEEvRT_RKT0_RKT1_(%"class.Eigen::Matrix"* dereferenceable(32) %dst, %"class.Eigen::CwiseNullaryOp"* dereferenceable(16) %src, %"struct.Eigen::internal::assign_op"* dereferenceable(1) %func) local_unnamed_addr #3 comdat {
entry:
  %srcEvaluator = alloca %"struct.Eigen::internal::evaluator.8", align 8
  %dstEvaluator = alloca %"struct.Eigen::internal::evaluator", align 8
  %kernel = alloca %"class.Eigen::internal::generic_dense_assignment_kernel", align 8
  %0 = bitcast %"struct.Eigen::internal::evaluator.8"* %srcEvaluator to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %0) #12
  call void @_ZN5Eigen8internal9evaluatorINS_14CwiseNullaryOpINS0_18scalar_constant_opIdEENS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEEEC2ERKS7_(%"struct.Eigen::internal::evaluator.8"* nonnull %srcEvaluator, %"class.Eigen::CwiseNullaryOp"* nonnull dereferenceable(16) %src)
  call void @_ZN5Eigen8internal17resize_if_allowedINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EEddEEvRT_RKT0_RKNS0_9assign_opIT1_T2_EE(%"class.Eigen::Matrix"* nonnull dereferenceable(32) %dst, %"class.Eigen::CwiseNullaryOp"* nonnull dereferenceable(16) %src, %"struct.Eigen::internal::assign_op"* nonnull dereferenceable(1) %func)
  %1 = bitcast %"struct.Eigen::internal::evaluator"* %dstEvaluator to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %1) #12
  call void @_ZN5Eigen8internal9evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEC2ERKS3_(%"struct.Eigen::internal::evaluator"* nonnull %dstEvaluator, %"class.Eigen::Matrix"* nonnull dereferenceable(32) %dst)
  %2 = bitcast %"class.Eigen::internal::generic_dense_assignment_kernel"* %kernel to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %2) #12
  %3 = bitcast %"class.Eigen::Matrix"* %dst to %"struct.Eigen::EigenBase"*
  %call = call dereferenceable(32) %"class.Eigen::Matrix"* @_ZNK5Eigen9EigenBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE18const_cast_derivedEv(%"struct.Eigen::EigenBase"* nonnull %3)
  call void @_ZN5Eigen8internal31generic_dense_assignment_kernelINS0_9evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEENS2_INS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES4_EEEENS0_9assign_opIddEELi0EEC2ERS5_RKSA_RKSC_RS4_(%"class.Eigen::internal::generic_dense_assignment_kernel"* nonnull %kernel, %"struct.Eigen::internal::evaluator"* nonnull dereferenceable(16) %dstEvaluator, %"struct.Eigen::internal::evaluator.8"* nonnull dereferenceable(16) %srcEvaluator, %"struct.Eigen::internal::assign_op"* nonnull dereferenceable(1) %func, %"class.Eigen::Matrix"* nonnull dereferenceable(32) %call)
  call void @_ZN5Eigen8internal21dense_assignment_loopINS0_31generic_dense_assignment_kernelINS0_9evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEENS3_INS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES5_EEEENS0_9assign_opIddEELi0EEELi2ELi2EE3runERSE_(%"class.Eigen::internal::generic_dense_assignment_kernel"* nonnull dereferenceable(32) %kernel)
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %2) #12
  %4 = bitcast %"struct.Eigen::internal::evaluator"* %dstEvaluator to %"class.Eigen::internal::noncopyable"*
  call void @_ZN5Eigen8internal11noncopyableD2Ev(%"class.Eigen::internal::noncopyable"* nonnull %4) #12
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %1) #12
  %5 = bitcast %"struct.Eigen::internal::evaluator.8"* %srcEvaluator to %"class.Eigen::internal::noncopyable"*
  call void @_ZN5Eigen8internal11noncopyableD2Ev(%"class.Eigen::internal::noncopyable"* nonnull %5) #12
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %0) #12
  ret void
}

; Function Attrs: norecurse nounwind uwtable
define linkonce_odr dso_local void @_ZN5Eigen8internal27checkTransposeAliasing_implINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EELb0EE3runERKS3_RKS7_(%"class.Eigen::Matrix"* dereferenceable(32), %"class.Eigen::CwiseNullaryOp"* dereferenceable(16)) local_unnamed_addr #0 comdat align 2 {
entry:
  ret void
}

; Function Attrs: norecurse nounwind uwtable
define linkonce_odr dso_local void @_ZN5Eigen8internal9evaluatorINS_14CwiseNullaryOpINS0_18scalar_constant_opIdEENS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEEEC2ERKS7_(%"struct.Eigen::internal::evaluator.8"* %this, %"class.Eigen::CwiseNullaryOp"* dereferenceable(16) %n) unnamed_addr #0 comdat align 2 {
entry:
  %0 = bitcast %"struct.Eigen::internal::evaluator.8"* %this to %"struct.Eigen::internal::evaluator_base.9"*
  tail call void @_ZN5Eigen8internal14evaluator_baseINS_14CwiseNullaryOpINS0_18scalar_constant_opIdEENS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEEEC2Ev(%"struct.Eigen::internal::evaluator_base.9"* %0)
  %m_functor = getelementptr inbounds %"struct.Eigen::internal::evaluator.8", %"struct.Eigen::internal::evaluator.8"* %this, i64 0, i32 0
  %call = tail call dereferenceable(8) %"struct.Eigen::internal::scalar_constant_op"* @_ZNK5Eigen14CwiseNullaryOpINS_8internal18scalar_constant_opIdEENS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE7functorEv(%"class.Eigen::CwiseNullaryOp"* nonnull %n)
  tail call void @_ZN5Eigen8internal18scalar_constant_opIdEC2ERKS2_(%"struct.Eigen::internal::scalar_constant_op"* %m_functor, %"struct.Eigen::internal::scalar_constant_op"* nonnull dereferenceable(8) %call)
  ret void
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local void @_ZN5Eigen8internal17resize_if_allowedINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EEddEEvRT_RKT0_RKNS0_9assign_opIT1_T2_EE(%"class.Eigen::Matrix"* dereferenceable(32) %dst, %"class.Eigen::CwiseNullaryOp"* dereferenceable(16) %src, %"struct.Eigen::internal::assign_op"* dereferenceable(1)) local_unnamed_addr #3 comdat {
entry:
  %call = tail call i64 @_ZNK5Eigen14CwiseNullaryOpINS_8internal18scalar_constant_opIdEENS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE4rowsEv(%"class.Eigen::CwiseNullaryOp"* nonnull %src)
  %call1 = tail call i64 @_ZNK5Eigen14CwiseNullaryOpINS_8internal18scalar_constant_opIdEENS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE4colsEv(%"class.Eigen::CwiseNullaryOp"* nonnull %src)
  %1 = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %dst, i64 0, i32 0
  %call2 = tail call i64 @_ZNK5Eigen15PlainObjectBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE4rowsEv(%"class.Eigen::PlainObjectBase"* nonnull %1)
  %cmp = icmp eq i64 %call2, %call
  br i1 %cmp, label %lor.lhs.false, label %if.then

lor.lhs.false:                                    ; preds = %entry
  %call3 = tail call i64 @_ZNK5Eigen15PlainObjectBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE4colsEv(%"class.Eigen::PlainObjectBase"* nonnull %1)
  %cmp4 = icmp eq i64 %call3, %call1
  br i1 %cmp4, label %if.end, label %if.then

if.then:                                          ; preds = %lor.lhs.false, %entry
  tail call void @_ZN5Eigen15PlainObjectBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE6resizeEll(%"class.Eigen::PlainObjectBase"* nonnull %1, i64 %call, i64 %call1)
  br label %if.end

if.end:                                           ; preds = %lor.lhs.false, %if.then
  %call5 = tail call i64 @_ZNK5Eigen15PlainObjectBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE4rowsEv(%"class.Eigen::PlainObjectBase"* nonnull %1)
  %cmp6 = icmp eq i64 %call5, %call
  br i1 %cmp6, label %land.lhs.true, label %cond.false

land.lhs.true:                                    ; preds = %if.end
  %call7 = tail call i64 @_ZNK5Eigen15PlainObjectBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE4colsEv(%"class.Eigen::PlainObjectBase"* nonnull %1)
  %cmp8 = icmp eq i64 %call7, %call1
  br i1 %cmp8, label %cond.end, label %cond.false

cond.false:                                       ; preds = %land.lhs.true, %if.end
  tail call void @__assert_fail(i8* getelementptr inbounds ([47 x i8], [47 x i8]* @.str.17, i64 0, i64 0), i8* getelementptr inbounds ([59 x i8], [59 x i8]* @.str.18, i64 0, i64 0), i32 721, i8* getelementptr inbounds ([309 x i8], [309 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen8internal17resize_if_allowedINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEENS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES3_EEddEEvRT_RKT0_RKNS0_9assign_opIT1_T2_EE, i64 0, i64 0)) #14
  unreachable

cond.end:                                         ; preds = %land.lhs.true
  ret void
}

; Function Attrs: inlinehint norecurse nounwind uwtable
define linkonce_odr dso_local dereferenceable(32) %"class.Eigen::Matrix"* @_ZNK5Eigen9EigenBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE18const_cast_derivedEv(%"struct.Eigen::EigenBase"* %this) local_unnamed_addr #9 comdat align 2 {
entry:
  %0 = bitcast %"struct.Eigen::EigenBase"* %this to %"class.Eigen::Matrix"*
  ret %"class.Eigen::Matrix"* %0
}

; Function Attrs: norecurse nounwind uwtable
define linkonce_odr dso_local void @_ZN5Eigen8internal31generic_dense_assignment_kernelINS0_9evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEENS2_INS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES4_EEEENS0_9assign_opIddEELi0EEC2ERS5_RKSA_RKSC_RS4_(%"class.Eigen::internal::generic_dense_assignment_kernel"* %this, %"struct.Eigen::internal::evaluator"* dereferenceable(16) %dst, %"struct.Eigen::internal::evaluator.8"* dereferenceable(16) %src, %"struct.Eigen::internal::assign_op"* dereferenceable(1) %func, %"class.Eigen::Matrix"* dereferenceable(32) %dstExpr) unnamed_addr #0 comdat align 2 {
entry:
  %m_dst = getelementptr inbounds %"class.Eigen::internal::generic_dense_assignment_kernel", %"class.Eigen::internal::generic_dense_assignment_kernel"* %this, i64 0, i32 0
  store %"struct.Eigen::internal::evaluator"* %dst, %"struct.Eigen::internal::evaluator"** %m_dst, align 8, !tbaa !6
  %m_src = getelementptr inbounds %"class.Eigen::internal::generic_dense_assignment_kernel", %"class.Eigen::internal::generic_dense_assignment_kernel"* %this, i64 0, i32 1
  store %"struct.Eigen::internal::evaluator.8"* %src, %"struct.Eigen::internal::evaluator.8"** %m_src, align 8, !tbaa !6
  %m_functor = getelementptr inbounds %"class.Eigen::internal::generic_dense_assignment_kernel", %"class.Eigen::internal::generic_dense_assignment_kernel"* %this, i64 0, i32 2
  store %"struct.Eigen::internal::assign_op"* %func, %"struct.Eigen::internal::assign_op"** %m_functor, align 8, !tbaa !6
  %m_dstExpr = getelementptr inbounds %"class.Eigen::internal::generic_dense_assignment_kernel", %"class.Eigen::internal::generic_dense_assignment_kernel"* %this, i64 0, i32 3
  store %"class.Eigen::Matrix"* %dstExpr, %"class.Eigen::Matrix"** %m_dstExpr, align 8, !tbaa !6
  ret void
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local void @_ZN5Eigen8internal21dense_assignment_loopINS0_31generic_dense_assignment_kernelINS0_9evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEENS3_INS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES5_EEEENS0_9assign_opIddEELi0EEELi2ELi2EE3runERSE_(%"class.Eigen::internal::generic_dense_assignment_kernel"* dereferenceable(32) %kernel) local_unnamed_addr #3 comdat align 2 {
entry:
  tail call void @_ZN5Eigen8internal47copy_using_evaluator_innervec_CompleteUnrollingINS0_31generic_dense_assignment_kernelINS0_9evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEENS3_INS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES5_EEEENS0_9assign_opIddEELi0EEELi0ELi4EE3runERSE_(%"class.Eigen::internal::generic_dense_assignment_kernel"* nonnull dereferenceable(32) %kernel)
  ret void
}

; Function Attrs: inlinehint norecurse nounwind uwtable
define linkonce_odr dso_local void @_ZN5Eigen8internal14evaluator_baseINS_14CwiseNullaryOpINS0_18scalar_constant_opIdEENS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEEEC2Ev(%"struct.Eigen::internal::evaluator_base.9"* %this) unnamed_addr #9 comdat align 2 {
entry:
  %0 = bitcast %"struct.Eigen::internal::evaluator_base.9"* %this to %"class.Eigen::internal::noncopyable"*
  tail call void @_ZN5Eigen8internal11noncopyableC2Ev(%"class.Eigen::internal::noncopyable"* %0)
  ret void
}

; Function Attrs: norecurse nounwind uwtable
define linkonce_odr dso_local dereferenceable(8) %"struct.Eigen::internal::scalar_constant_op"* @_ZNK5Eigen14CwiseNullaryOpINS_8internal18scalar_constant_opIdEENS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE7functorEv(%"class.Eigen::CwiseNullaryOp"* %this) local_unnamed_addr #0 comdat align 2 {
entry:
  %m_functor = getelementptr inbounds %"class.Eigen::CwiseNullaryOp", %"class.Eigen::CwiseNullaryOp"* %this, i64 0, i32 2
  ret %"struct.Eigen::internal::scalar_constant_op"* %m_functor
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local void @_ZN5Eigen8internal47copy_using_evaluator_innervec_CompleteUnrollingINS0_31generic_dense_assignment_kernelINS0_9evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEENS3_INS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES5_EEEENS0_9assign_opIddEELi0EEELi0ELi4EE3runERSE_(%"class.Eigen::internal::generic_dense_assignment_kernel"* dereferenceable(32) %kernel) local_unnamed_addr #3 comdat align 2 {
entry:
  tail call void @_ZN5Eigen8internal31generic_dense_assignment_kernelINS0_9evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEENS2_INS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES4_EEEENS0_9assign_opIddEELi0EE24assignPacketByOuterInnerILi16ELi16EDv2_dEEvll(%"class.Eigen::internal::generic_dense_assignment_kernel"* nonnull %kernel, i64 0, i64 0)
  tail call void @_ZN5Eigen8internal47copy_using_evaluator_innervec_CompleteUnrollingINS0_31generic_dense_assignment_kernelINS0_9evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEENS3_INS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES5_EEEENS0_9assign_opIddEELi0EEELi2ELi4EE3runERSE_(%"class.Eigen::internal::generic_dense_assignment_kernel"* nonnull dereferenceable(32) %kernel)
  ret void
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local void @_ZN5Eigen8internal31generic_dense_assignment_kernelINS0_9evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEENS2_INS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES4_EEEENS0_9assign_opIddEELi0EE24assignPacketByOuterInnerILi16ELi16EDv2_dEEvll(%"class.Eigen::internal::generic_dense_assignment_kernel"* %this, i64 %outer, i64 %inner) local_unnamed_addr #3 comdat align 2 {
entry:
  %call = tail call i64 @_ZN5Eigen8internal31generic_dense_assignment_kernelINS0_9evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEENS2_INS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES4_EEEENS0_9assign_opIddEELi0EE20rowIndexByOuterInnerEll(i64 %outer, i64 %inner)
  %call2 = tail call i64 @_ZN5Eigen8internal31generic_dense_assignment_kernelINS0_9evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEENS2_INS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES4_EEEENS0_9assign_opIddEELi0EE20colIndexByOuterInnerEll(i64 %outer, i64 %inner)
  tail call void @_ZN5Eigen8internal31generic_dense_assignment_kernelINS0_9evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEENS2_INS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES4_EEEENS0_9assign_opIddEELi0EE12assignPacketILi16ELi16EDv2_dEEvll(%"class.Eigen::internal::generic_dense_assignment_kernel"* %this, i64 %call, i64 %call2)
  ret void
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local void @_ZN5Eigen8internal47copy_using_evaluator_innervec_CompleteUnrollingINS0_31generic_dense_assignment_kernelINS0_9evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEENS3_INS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES5_EEEENS0_9assign_opIddEELi0EEELi2ELi4EE3runERSE_(%"class.Eigen::internal::generic_dense_assignment_kernel"* dereferenceable(32) %kernel) local_unnamed_addr #3 comdat align 2 {
entry:
  tail call void @_ZN5Eigen8internal31generic_dense_assignment_kernelINS0_9evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEENS2_INS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES4_EEEENS0_9assign_opIddEELi0EE24assignPacketByOuterInnerILi16ELi16EDv2_dEEvll(%"class.Eigen::internal::generic_dense_assignment_kernel"* nonnull %kernel, i64 1, i64 0)
  tail call void @_ZN5Eigen8internal47copy_using_evaluator_innervec_CompleteUnrollingINS0_31generic_dense_assignment_kernelINS0_9evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEENS3_INS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES5_EEEENS0_9assign_opIddEELi0EEELi4ELi4EE3runERSE_(%"class.Eigen::internal::generic_dense_assignment_kernel"* nonnull dereferenceable(32) %kernel)
  ret void
}

; Function Attrs: inlinehint norecurse nounwind uwtable
define linkonce_odr dso_local i64 @_ZN5Eigen8internal31generic_dense_assignment_kernelINS0_9evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEENS2_INS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES4_EEEENS0_9assign_opIddEELi0EE20rowIndexByOuterInnerEll(i64 %outer, i64 %inner) local_unnamed_addr #9 comdat align 2 {
entry:
  ret i64 %inner
}

; Function Attrs: inlinehint norecurse nounwind uwtable
define linkonce_odr dso_local i64 @_ZN5Eigen8internal31generic_dense_assignment_kernelINS0_9evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEENS2_INS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES4_EEEENS0_9assign_opIddEELi0EE20colIndexByOuterInnerEll(i64 %outer, i64 %inner) local_unnamed_addr #9 comdat align 2 {
entry:
  ret i64 %outer
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local void @_ZN5Eigen8internal31generic_dense_assignment_kernelINS0_9evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEENS2_INS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES4_EEEENS0_9assign_opIddEELi0EE12assignPacketILi16ELi16EDv2_dEEvll(%"class.Eigen::internal::generic_dense_assignment_kernel"* %this, i64 %row, i64 %col) local_unnamed_addr #3 comdat align 2 {
entry:
  %ref.tmp = alloca <2 x double>, align 16
  %m_functor = getelementptr inbounds %"class.Eigen::internal::generic_dense_assignment_kernel", %"class.Eigen::internal::generic_dense_assignment_kernel"* %this, i64 0, i32 2
  %0 = load %"struct.Eigen::internal::assign_op"*, %"struct.Eigen::internal::assign_op"** %m_functor, align 8, !tbaa !17
  %1 = bitcast %"class.Eigen::internal::generic_dense_assignment_kernel"* %this to %"struct.Eigen::internal::evaluator.6"**
  %2 = load %"struct.Eigen::internal::evaluator.6"*, %"struct.Eigen::internal::evaluator.6"** %1, align 8, !tbaa !19
  %call = tail call dereferenceable(8) double* @_ZN5Eigen8internal9evaluatorINS_15PlainObjectBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEEE8coeffRefEll(%"struct.Eigen::internal::evaluator.6"* %2, i64 %row, i64 %col)
  %3 = bitcast <2 x double>* %ref.tmp to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %3) #12
  %m_src = getelementptr inbounds %"class.Eigen::internal::generic_dense_assignment_kernel", %"class.Eigen::internal::generic_dense_assignment_kernel"* %this, i64 0, i32 1
  %4 = load %"struct.Eigen::internal::evaluator.8"*, %"struct.Eigen::internal::evaluator.8"** %m_src, align 8, !tbaa !20
  %call2 = tail call <2 x double> @_ZNK5Eigen8internal9evaluatorINS_14CwiseNullaryOpINS0_18scalar_constant_opIdEENS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEEE6packetILi16EDv2_dlEET0_T1_SC_(%"struct.Eigen::internal::evaluator.8"* %4, i64 %row, i64 %col)
  store <2 x double> %call2, <2 x double>* %ref.tmp, align 16, !tbaa !8
  call void @_ZNK5Eigen8internal9assign_opIddE12assignPacketILi16EDv2_dEEvPdRKT0_(%"struct.Eigen::internal::assign_op"* %0, double* nonnull %call, <2 x double>* nonnull dereferenceable(16) %ref.tmp)
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %3) #12
  ret void
}

; Function Attrs: inlinehint norecurse nounwind uwtable
define linkonce_odr dso_local void @_ZNK5Eigen8internal9assign_opIddE12assignPacketILi16EDv2_dEEvPdRKT0_(%"struct.Eigen::internal::assign_op"* %this, double* %a, <2 x double>* dereferenceable(16) %b) local_unnamed_addr #9 comdat align 2 {
entry:
  tail call void @_ZN5Eigen8internal6pstoreIdDv2_dEEvPT_RKT0_(double* %a, <2 x double>* nonnull dereferenceable(16) %b) #12
  ret void
}

; Function Attrs: inlinehint norecurse nounwind uwtable
define linkonce_odr dso_local dereferenceable(8) double* @_ZN5Eigen8internal9evaluatorINS_15PlainObjectBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEEE8coeffRefEll(%"struct.Eigen::internal::evaluator.6"* %this, i64 %row, i64 %col) local_unnamed_addr #9 comdat align 2 {
entry:
  %m_data = getelementptr inbounds %"struct.Eigen::internal::evaluator.6", %"struct.Eigen::internal::evaluator.6"* %this, i64 0, i32 0
  %0 = load double*, double** %m_data, align 8, !tbaa !9
  %call = tail call i64 @_ZN5Eigen8internal19variable_if_dynamicIlLi2EE5valueEv()
  %mul = mul nsw i64 %call, %col
  %add = add nsw i64 %mul, %row
  %arrayidx = getelementptr inbounds double, double* %0, i64 %add
  ret double* %arrayidx
}

; Function Attrs: inlinehint norecurse nounwind uwtable
define linkonce_odr dso_local <2 x double> @_ZNK5Eigen8internal9evaluatorINS_14CwiseNullaryOpINS0_18scalar_constant_opIdEENS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEEE6packetILi16EDv2_dlEET0_T1_SC_(%"struct.Eigen::internal::evaluator.8"* %this, i64 %row, i64 %col) local_unnamed_addr #9 comdat align 2 {
entry:
  %m_wrapper = getelementptr inbounds %"struct.Eigen::internal::evaluator.8", %"struct.Eigen::internal::evaluator.8"* %this, i64 0, i32 1
  %m_functor = getelementptr inbounds %"struct.Eigen::internal::evaluator.8", %"struct.Eigen::internal::evaluator.8"* %this, i64 0, i32 0
  %call = tail call <2 x double> @_ZNK5Eigen8internal15nullary_wrapperIdNS0_18scalar_constant_opIdEELb1ELb0ELb0EE8packetOpIDv2_dlEET_RKS3_T0_SA_(%"struct.Eigen::internal::nullary_wrapper"* nonnull %m_wrapper, %"struct.Eigen::internal::scalar_constant_op"* dereferenceable(8) %m_functor, i64 %row, i64 %col)
  ret <2 x double> %call
}

; Function Attrs: inlinehint norecurse nounwind uwtable
define linkonce_odr dso_local void @_ZN5Eigen8internal6pstoreIdDv2_dEEvPT_RKT0_(double* %to, <2 x double>* dereferenceable(16) %from) local_unnamed_addr #11 comdat {
entry:
  %0 = load <2 x double>, <2 x double>* %from, align 16, !tbaa !8
  %1 = bitcast double* %to to <2 x double>*
  store <2 x double> %0, <2 x double>* %1, align 16, !tbaa !8
  ret void
}

; Function Attrs: inlinehint norecurse nounwind uwtable
define linkonce_odr dso_local <2 x double> @_ZNK5Eigen8internal15nullary_wrapperIdNS0_18scalar_constant_opIdEELb1ELb0ELb0EE8packetOpIDv2_dlEET_RKS3_T0_SA_(%"struct.Eigen::internal::nullary_wrapper"* %this, %"struct.Eigen::internal::scalar_constant_op"* dereferenceable(8) %op, i64, i64) local_unnamed_addr #9 comdat align 2 {
entry:
  %call = tail call <2 x double> @_ZNK5Eigen8internal18scalar_constant_opIdE8packetOpIDv2_dEEKT_v(%"struct.Eigen::internal::scalar_constant_op"* nonnull %op)
  ret <2 x double> %call
}

; Function Attrs: inlinehint norecurse nounwind uwtable
define linkonce_odr dso_local <2 x double> @_ZNK5Eigen8internal18scalar_constant_opIdE8packetOpIDv2_dEEKT_v(%"struct.Eigen::internal::scalar_constant_op"* %this) local_unnamed_addr #9 comdat align 2 {
entry:
  %m_other = getelementptr inbounds %"struct.Eigen::internal::scalar_constant_op", %"struct.Eigen::internal::scalar_constant_op"* %this, i64 0, i32 0
  %call = tail call <2 x double> @_ZN5Eigen8internal5pset1IDv2_dEET_RKNS0_15unpacket_traitsIS3_E4typeE(double* dereferenceable(8) %m_other)
  ret <2 x double> %call
}

; Function Attrs: inlinehint norecurse nounwind uwtable
define linkonce_odr dso_local <2 x double> @_ZN5Eigen8internal5pset1IDv2_dEET_RKNS0_15unpacket_traitsIS3_E4typeE(double* dereferenceable(8) %from) local_unnamed_addr #11 comdat {
entry:
  %0 = load double, double* %from, align 8, !tbaa !2
  %vecinit.i = insertelement <2 x double> undef, double %0, i32 0
  %vecinit1.i = shufflevector <2 x double> %vecinit.i, <2 x double> undef, <2 x i32> zeroinitializer
  ret <2 x double> %vecinit1.i
}

; Function Attrs: inlinehint norecurse nounwind uwtable
define linkonce_odr dso_local void @_ZN5Eigen8internal47copy_using_evaluator_innervec_CompleteUnrollingINS0_31generic_dense_assignment_kernelINS0_9evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEENS3_INS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES5_EEEENS0_9assign_opIddEELi0EEELi4ELi4EE3runERSE_(%"class.Eigen::internal::generic_dense_assignment_kernel"* dereferenceable(32)) local_unnamed_addr #9 comdat align 2 {
entry:
  ret void
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local dereferenceable(8) double* @_ZN5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEELi1EE8coeffRefEll(%"class.Eigen::DenseCoeffsBase.0"* %this, i64 %row, i64 %col) local_unnamed_addr #3 comdat align 2 {
entry:
  %ref.tmp = alloca %"struct.Eigen::internal::evaluator", align 8
  %0 = bitcast %"struct.Eigen::internal::evaluator"* %ref.tmp to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %0) #12
  %1 = bitcast %"class.Eigen::DenseCoeffsBase.0"* %this to %"struct.Eigen::EigenBase"*
  %call = tail call dereferenceable(32) %"class.Eigen::Matrix"* @_ZN5Eigen9EigenBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE7derivedEv(%"struct.Eigen::EigenBase"* %1)
  call void @_ZN5Eigen8internal9evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEC2ERKS3_(%"struct.Eigen::internal::evaluator"* nonnull %ref.tmp, %"class.Eigen::Matrix"* nonnull dereferenceable(32) %call)
  %2 = bitcast %"struct.Eigen::internal::evaluator"* %ref.tmp to %"struct.Eigen::internal::evaluator.6"*
  %call2 = call dereferenceable(8) double* @_ZN5Eigen8internal9evaluatorINS_15PlainObjectBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEEE8coeffRefEll(%"struct.Eigen::internal::evaluator.6"* nonnull %2, i64 %row, i64 %col)
  %3 = bitcast %"struct.Eigen::internal::evaluator"* %ref.tmp to %"class.Eigen::internal::noncopyable"*
  call void @_ZN5Eigen8internal11noncopyableD2Ev(%"class.Eigen::internal::noncopyable"* nonnull %3) #12
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %0) #12
  ret double* %call2
}

attributes #0 = { norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { inlinehint nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #7 = { nounwind readnone speculatable }
attributes #8 = { noreturn nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #9 = { inlinehint norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #10 = { inlinehint nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="128" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #11 = { inlinehint norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="128" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #12 = { nounwind }
attributes #13 = { cold }
attributes #14 = { noreturn nounwind }

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
!8 = !{!4, !4, i64 0}
!9 = !{!10, !7, i64 0}
!10 = !{!"_ZTSN5Eigen8internal9evaluatorINS_15PlainObjectBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEEEE", !7, i64 0, !11, i64 8}
!11 = !{!"_ZTSN5Eigen8internal19variable_if_dynamicIlLi2EEE"}
!12 = !{!13, !7, i64 16}
!13 = !{!"_ZTSN5Eigen8internal15redux_evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEE", !14, i64 0, !7, i64 16}
!14 = !{!"_ZTSN5Eigen8internal9evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEE"}
!15 = !{!16, !3, i64 0}
!16 = !{!"_ZTSN5Eigen8internal18scalar_constant_opIdEE", !3, i64 0}
!17 = !{!18, !7, i64 16}
!18 = !{!"_ZTSN5Eigen8internal31generic_dense_assignment_kernelINS0_9evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEENS2_INS_14CwiseNullaryOpINS0_18scalar_constant_opIdEES4_EEEENS0_9assign_opIddEELi0EEE", !7, i64 0, !7, i64 8, !7, i64 16, !7, i64 24}
!19 = !{!18, !7, i64 0}
!20 = !{!18, !7, i64 8}

; CHECK: define internal {} @diffe_ZL6matvecPKN5Eigen6MatrixIdLi2ELi2ELi0ELi2ELi2EEE(%"class.Eigen::Matrix"* noalias %W, %"class.Eigen::Matrix"* %"W'", double %differeturn) #5 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %malloccall = tail call i8* @malloc(i64 1) #12
; CHECK-NEXT:   %"malloccall'mi" = tail call noalias nonnull i8* @malloc(i64 1) #12
; CHECK-NEXT:   store i8 0, i8* %"malloccall'mi", align 1
; CHECK-NEXT:   %func = bitcast i8* %malloccall to %"struct.Eigen::internal::scalar_sum_op"*
; CHECK-NEXT:   %malloccall1 = tail call i8* @malloc(i64 24) #12
; CHECK-NEXT:   %"malloccall1'mi" = tail call noalias nonnull i8* @malloc(i64 24) #12
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull align 1 %"malloccall1'mi", i8 0, i64 24, i1 false)
; CHECK-NEXT:   %thisEval = bitcast i8* %malloccall1 to %"class.Eigen::internal::redux_evaluator"*
; CHECK-NEXT:   %"thisEval'ipc6" = bitcast i8* %"malloccall1'mi" to %"class.Eigen::internal::redux_evaluator"*
; CHECK-NEXT:   %_augmented = call { { { { {}, { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64, { {} }, double*, double*, { {} } } } } } @augmented__ZN5Eigen8internal15redux_evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEC2ERKS3_(%"class.Eigen::internal::redux_evaluator"* %thisEval, %"class.Eigen::internal::redux_evaluator"* %"thisEval'ipc6", %"class.Eigen::Matrix"* %W, %"class.Eigen::Matrix"* %"W'")
; CHECK-NEXT:   %0 = extractvalue { { { { {}, { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64, { {} }, double*, double*, { {} } } } } } %_augmented, 0
; CHECK-NEXT:   %"thisEval'ipc3" = bitcast i8* %"malloccall1'mi" to %"class.Eigen::internal::redux_evaluator"*
; CHECK-NEXT:   %"func'ipc4" = bitcast i8* %"malloccall'mi" to %"struct.Eigen::internal::scalar_sum_op"*
; CHECK-NEXT:   %call5_augmented = call { {} } @augmented__ZN5Eigen8internal10redux_implINS0_13scalar_sum_opIddEENS0_15redux_evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEELi3ELi2EE3runERKS7_RKS3_(%"class.Eigen::internal::redux_evaluator"* %thisEval, %"class.Eigen::internal::redux_evaluator"* %"thisEval'ipc3", %"struct.Eigen::internal::scalar_sum_op"* %func, %"struct.Eigen::internal::scalar_sum_op"* %"func'ipc4")
; CHECK-NEXT:   %"thisEval'ipc" = bitcast i8* %"malloccall1'mi" to %"class.Eigen::internal::redux_evaluator"*
; CHECK-NEXT:   %1 = call {} @diffenothing(%"class.Eigen::internal::redux_evaluator"* nonnull %thisEval, %"class.Eigen::internal::redux_evaluator"* %"thisEval'ipc") #12
; CHECK-NEXT:   %"thisEval'ipc2" = bitcast i8* %"malloccall1'mi" to %"class.Eigen::internal::redux_evaluator"*
; CHECK-NEXT:   %"func'ipc" = bitcast i8* %"malloccall'mi" to %"struct.Eigen::internal::scalar_sum_op"*
; CHECK-NEXT:   %2 = call {} @diffe_ZN5Eigen8internal10redux_implINS0_13scalar_sum_opIddEENS0_15redux_evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEELi3ELi2EE3runERKS7_RKS3_(%"class.Eigen::internal::redux_evaluator"* nonnull %thisEval, %"class.Eigen::internal::redux_evaluator"* %"thisEval'ipc2", %"struct.Eigen::internal::scalar_sum_op"* nonnull %func, %"struct.Eigen::internal::scalar_sum_op"* %"func'ipc", double %differeturn, {} undef)
; CHECK-NEXT:   %"thisEval'ipc5" = bitcast i8* %"malloccall1'mi" to %"class.Eigen::internal::redux_evaluator"*
; CHECK-NEXT:   %3 = call {} @diffe_ZN5Eigen8internal15redux_evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEC2ERKS3_(%"class.Eigen::internal::redux_evaluator"* nonnull %thisEval, %"class.Eigen::internal::redux_evaluator"* %"thisEval'ipc5", %"class.Eigen::Matrix"* nonnull %W, %"class.Eigen::Matrix"* %"W'", { { { {}, { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64, { {} }, double*, double*, { {} } } } } %0)
; CHECK-NEXT:   tail call void @free(i8* nonnull %"malloccall1'mi")
; CHECK-NEXT:   tail call void @free(i8* %malloccall1)
; CHECK-NEXT:   tail call void @free(i8* nonnull %"malloccall'mi")
; CHECK-NEXT:   tail call void @free(i8* %malloccall)
; CHECK-NEXT:   ret {} undef
; CHECK-NEXT: }

; CHECK: define internal {} @diffenothing(%"class.Eigen::internal::redux_evaluator"* %this, %"class.Eigen::internal::redux_evaluator"* %"this'") {
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret {} undef
; CHECK-NEXT: }

; CHECK: define internal { {} } @augmented__ZN5Eigen8internal10redux_implINS0_13scalar_sum_opIddEENS0_15redux_evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEELi3ELi2EE3runERKS7_RKS3_(%"class.Eigen::internal::redux_evaluator"* dereferenceable(24) %mat, %"class.Eigen::internal::redux_evaluator"* %"mat'", %"struct.Eigen::internal::scalar_sum_op"* dereferenceable(1) %func, %"struct.Eigen::internal::scalar_sum_op"* %"func'") local_unnamed_addr #3 align 2 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret { {} } undef
; CHECK-NEXT: }

; CHECK: define internal {} @diffe_ZN5Eigen8internal10redux_implINS0_13scalar_sum_opIddEENS0_15redux_evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEELi3ELi2EE3runERKS7_RKS3_(%"class.Eigen::internal::redux_evaluator"* dereferenceable(24) %mat, %"class.Eigen::internal::redux_evaluator"* %"mat'", %"struct.Eigen::internal::scalar_sum_op"* dereferenceable(1) %func, %"struct.Eigen::internal::scalar_sum_op"* %"func'", double %differeturn, {} %tapeArg) local_unnamed_addr #3 align 2 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"m_data'ipc" = bitcast %"class.Eigen::internal::redux_evaluator"* %"mat'" to <2 x double>**
; CHECK-NEXT:   %"from'ipl" = load <2 x double>*, <2 x double>** %"m_data'ipc", align 8
; CHECK-NEXT:   %"call3'de.0.vec.insert" = insertelement <2 x double> undef, double %differeturn, i32 0
; CHECK-NEXT:   %"call3'de.8.vec.insert" = shufflevector <2 x double> %"call3'de.0.vec.insert", <2 x double> undef, <2 x i32> zeroinitializer
; CHECK-NEXT:   %0 = load <2 x double>, <2 x double>* %"from'ipl", align 16
; CHECK-NEXT:   %1 = fadd fast <2 x double> %0, %"call3'de.8.vec.insert"
; CHECK-NEXT:   store <2 x double> %1, <2 x double>* %"from'ipl", align 16
; CHECK-NEXT:   ret {} undef
; CHECK-NEXT: }

; CHECK: define internal { {} } @augmented__ZN5Eigen8internal19variable_if_dynamicIlLi2EEC2El(%"class.Eigen::internal::variable_if_dynamic"* %this, %"class.Eigen::internal::variable_if_dynamic"* %"this'", i64 %v) unnamed_addr #3 align 2 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %cmp = icmp eq i64 %v, 2
; CHECK-NEXT:   br i1 %cmp, label %cond.end, label %cond.false
; CHECK: cond.false:                                       ; preds = %entry
; CHECK-NEXT:   tail call void @__assert_fail(i8* getelementptr inbounds ([14 x i8], [14 x i8]* @.str.8, i64 0, i64 0), i8* getelementptr inbounds ([58 x i8], [58 x i8]* @.str.9, i64 0, i64 0), i32 110, i8* getelementptr inbounds ([92 x i8], [92 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen8internal19variable_if_dynamicIlLi2EEC2El, i64 0, i64 0)) #14
; CHECK-NEXT:   unreachable
; CHECK: cond.end:                                         ; preds = %entry
; CHECK-NEXT:   ret { {} } undef
; CHECK-NEXT: }

; CHECK: define internal { { i64 }, i64 } @augmented__ZNK5Eigen15PlainObjectBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE4rowsEv(%"class.Eigen::PlainObjectBase"* %this, %"class.Eigen::PlainObjectBase"* %"this'") local_unnamed_addr #9 align 2 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %call = tail call i64 @_ZN5Eigen12DenseStorageIdLi4ELi2ELi2ELi0EE4rowsEv()
; CHECK-NEXT:   %.fca.0.0.insert = insertvalue { { i64 }, i64 } undef, i64 %call, 0, 0
; CHECK-NEXT:   %.fca.1.insert = insertvalue { { i64 }, i64 } %.fca.0.0.insert, i64 %call, 1
; CHECK-NEXT:   ret { { i64 }, i64 } %.fca.1.insert
; CHECK-NEXT: }

; CHECK: define internal { {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } @augmented__ZNK5Eigen9EigenBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE7derivedEv(%"struct.Eigen::EigenBase"* %this, %"struct.Eigen::EigenBase"* %"this'") local_unnamed_addr #0 align 2 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = bitcast %"struct.Eigen::EigenBase"* %this to %"class.Eigen::Matrix"*
; CHECK-NEXT:   %"'ipc" = bitcast %"struct.Eigen::EigenBase"* %"this'" to %"class.Eigen::Matrix"*
; CHECK-NEXT:   %.fca.1.insert = insertvalue { {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } undef, %"class.Eigen::Matrix"* %0, 1
; CHECK-NEXT:   %.fca.2.insert = insertvalue { {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } %.fca.1.insert, %"class.Eigen::Matrix"* %"'ipc", 2
; CHECK-NEXT:   ret { {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } %.fca.2.insert
; CHECK-NEXT: }

; CHECK: define internal { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64 } @augmented__ZNK5Eigen9EigenBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE4rowsEv(%"struct.Eigen::EigenBase"* %this, %"struct.Eigen::EigenBase"* %"this'") local_unnamed_addr #9 align 2 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %call_augmented = call { {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } @augmented__ZNK5Eigen9EigenBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE7derivedEv(%"struct.Eigen::EigenBase"* %this, %"struct.Eigen::EigenBase"* %"this'")
; CHECK-NEXT:   %antiptr_call = extractvalue { {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } %call_augmented, 2
; CHECK-NEXT:   %call = extractvalue { {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } %call_augmented, 1
; CHECK-NEXT:   %"'ipge" = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %antiptr_call, i64 0, i32 0
; CHECK-NEXT:   %0 = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %call, i64 0, i32 0
; CHECK-NEXT:   %call2_augmented = call { { i64 }, i64 } @augmented__ZNK5Eigen15PlainObjectBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE4rowsEv(%"class.Eigen::PlainObjectBase"* %0, %"class.Eigen::PlainObjectBase"* %"'ipge")
; CHECK-NEXT:   %subcache = extractvalue { { i64 }, i64 } %call2_augmented, 0
; CHECK-NEXT:   %subcache.fca.0.extract = extractvalue { i64 } %subcache, 0
; CHECK-NEXT:   %call2 = extractvalue { { i64 }, i64 } %call2_augmented, 1
; CHECK-NEXT:   %.fca.0.0.0.insert = insertvalue { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64 } undef, i64 %subcache.fca.0.extract, 0, 0, 0
; CHECK-NEXT:   %.fca.0.2.insert = insertvalue { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64 } %.fca.0.0.0.insert, %"class.Eigen::Matrix"* %antiptr_call, 0, 2
; CHECK-NEXT:   %.fca.0.3.insert = insertvalue { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64 } %.fca.0.2.insert, %"class.Eigen::Matrix"* %call, 0, 3
; CHECK-NEXT:   %.fca.1.insert = insertvalue { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64 } %.fca.0.3.insert, i64 %call2, 1
; CHECK-NEXT:   ret { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64 } %.fca.1.insert
; CHECK-NEXT: }

; CHECK: define internal { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } }, i64 } @augmented__ZNK5Eigen9DenseBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE9innerSizeEv(%"class.Eigen::DenseBase"* %this, %"class.Eigen::DenseBase"* %"this'") local_unnamed_addr #0 align 2 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = bitcast %"class.Eigen::DenseBase"* %this to %"struct.Eigen::EigenBase"*
; CHECK-NEXT:   %"'ipc" = bitcast %"class.Eigen::DenseBase"* %"this'" to %"struct.Eigen::EigenBase"*
; CHECK-NEXT:   %call_augmented = call { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64 } @augmented__ZNK5Eigen9EigenBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE4rowsEv(%"struct.Eigen::EigenBase"* %0, %"struct.Eigen::EigenBase"* %"'ipc")
; CHECK-NEXT:   %subcache = extractvalue { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64 } %call_augmented, 0
; CHECK-NEXT:   %subcache.fca.0.0.extract = extractvalue { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } %subcache, 0, 0
; CHECK-NEXT:   %subcache.fca.2.extract = extractvalue { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } %subcache, 2
; CHECK-NEXT:   %subcache.fca.3.extract = extractvalue { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } %subcache, 3
; CHECK-NEXT:   %call = extractvalue { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64 } %call_augmented, 1
; CHECK-NEXT:   %.fca.0.0.0.0.insert = insertvalue { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } }, i64 } undef, i64 %subcache.fca.0.0.extract, 0, 0, 0, 0
; CHECK-NEXT:   %.fca.0.0.2.insert = insertvalue { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } }, i64 } %.fca.0.0.0.0.insert, %"class.Eigen::Matrix"* %subcache.fca.2.extract, 0, 0, 2
; CHECK-NEXT:   %.fca.0.0.3.insert = insertvalue { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } }, i64 } %.fca.0.0.2.insert, %"class.Eigen::Matrix"* %subcache.fca.3.extract, 0, 0, 3
; CHECK-NEXT:   %.fca.1.insert = insertvalue { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } }, i64 } %.fca.0.0.3.insert, i64 %call, 1
; CHECK-NEXT:   ret { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } }, i64 } %.fca.1.insert
; CHECK-NEXT: }

; CHECK: define internal { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, i64 } @augmented__ZNK5Eigen6MatrixIdLi2ELi2ELi0ELi2ELi2EE11outerStrideEv(%"class.Eigen::Matrix"* %this, %"class.Eigen::Matrix"* %"this'") local_unnamed_addr #9 align 2 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = bitcast %"class.Eigen::Matrix"* %this to %"class.Eigen::DenseBase"*
; CHECK-NEXT:   %"'ipc" = bitcast %"class.Eigen::Matrix"* %"this'" to %"class.Eigen::DenseBase"*
; CHECK-NEXT:   %call_augmented = call { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } }, i64 } @augmented__ZNK5Eigen9DenseBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE9innerSizeEv(%"class.Eigen::DenseBase"* %0, %"class.Eigen::DenseBase"* %"'ipc")
; CHECK-NEXT:   %subcache = extractvalue { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } }, i64 } %call_augmented, 0
; CHECK-NEXT:   %subcache.fca.0.0.0.extract = extractvalue { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } %subcache, 0, 0, 0
; CHECK-NEXT:   %subcache.fca.0.2.extract = extractvalue { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } %subcache, 0, 2
; CHECK-NEXT:   %subcache.fca.0.3.extract = extractvalue { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } %subcache, 0, 3
; CHECK-NEXT:   %call = extractvalue { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } }, i64 } %call_augmented, 1
; CHECK-NEXT:   %.fca.0.0.0.0.0.insert = insertvalue { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, i64 } undef, i64 %subcache.fca.0.0.0.extract, 0, 0, 0, 0, 0
; CHECK-NEXT:   %.fca.0.0.0.2.insert = insertvalue { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, i64 } %.fca.0.0.0.0.0.insert, %"class.Eigen::Matrix"* %subcache.fca.0.2.extract, 0, 0, 0, 2
; CHECK-NEXT:   %.fca.0.0.0.3.insert = insertvalue { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, i64 } %.fca.0.0.0.2.insert, %"class.Eigen::Matrix"* %subcache.fca.0.3.extract, 0, 0, 0, 3
; CHECK-NEXT:   %.fca.1.insert = insertvalue { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, i64 } %.fca.0.0.0.3.insert, i64 %call, 1
; CHECK-NEXT:   ret { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, i64 } %.fca.1.insert
; CHECK-NEXT: }

; CHECK: define internal { { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64 } @augmented__ZNK5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEELi3EE11outerStrideEv(%"class.Eigen::DenseCoeffsBase"* %this, %"class.Eigen::DenseCoeffsBase"* %"this'") local_unnamed_addr #9 align 2 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = bitcast %"class.Eigen::DenseCoeffsBase"* %this to %"struct.Eigen::EigenBase"*
; CHECK-NEXT:   %"'ipc" = bitcast %"class.Eigen::DenseCoeffsBase"* %"this'" to %"struct.Eigen::EigenBase"*
; CHECK-NEXT:   %call_augmented = call { {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } @augmented__ZNK5Eigen9EigenBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE7derivedEv(%"struct.Eigen::EigenBase"* %0, %"struct.Eigen::EigenBase"* %"'ipc")
; CHECK-NEXT:   %antiptr_call = extractvalue { {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } %call_augmented, 2
; CHECK-NEXT:   %call = extractvalue { {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } %call_augmented, 1
; CHECK-NEXT:   %call2_augmented = call { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, i64 } @augmented__ZNK5Eigen6MatrixIdLi2ELi2ELi0ELi2ELi2EE11outerStrideEv(%"class.Eigen::Matrix"* %call, %"class.Eigen::Matrix"* %antiptr_call)
; CHECK-NEXT:   %subcache = extractvalue { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, i64 } %call2_augmented, 0
; CHECK-NEXT:   %subcache.fca.0.0.0.0.extract = extractvalue { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } } %subcache, 0, 0, 0, 0
; CHECK-NEXT:   %subcache.fca.0.0.2.extract = extractvalue { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } } %subcache, 0, 0, 2
; CHECK-NEXT:   %subcache.fca.0.0.3.extract = extractvalue { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } } %subcache, 0, 0, 3
; CHECK-NEXT:   %call2 = extractvalue { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, i64 } %call2_augmented, 1
; CHECK-NEXT:   %.fca.0.0.0.0.0.0.insert = insertvalue { { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64 } undef, i64 %subcache.fca.0.0.0.0.extract, 0, 0, 0, 0, 0, 0
; CHECK-NEXT:   %.fca.0.0.0.0.2.insert = insertvalue { { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64 } %.fca.0.0.0.0.0.0.insert, %"class.Eigen::Matrix"* %subcache.fca.0.0.2.extract, 0, 0, 0, 0, 2
; CHECK-NEXT:   %.fca.0.0.0.0.3.insert = insertvalue { { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64 } %.fca.0.0.0.0.2.insert, %"class.Eigen::Matrix"* %subcache.fca.0.0.3.extract, 0, 0, 0, 0, 3
; CHECK-NEXT:   %.fca.0.2.insert = insertvalue { { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64 } %.fca.0.0.0.0.3.insert, %"class.Eigen::Matrix"* %antiptr_call, 0, 2
; CHECK-NEXT:   %.fca.0.3.insert = insertvalue { { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64 } %.fca.0.2.insert, %"class.Eigen::Matrix"* %call, 0, 3
; CHECK-NEXT:   %.fca.1.insert = insertvalue { { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64 } %.fca.0.3.insert, i64 %call2, 1
; CHECK-NEXT:   ret { { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64 } %.fca.1.insert
; CHECK-NEXT: }

; CHECK: define internal { {}, double*, double* } @augmented__ZNK5Eigen12DenseStorageIdLi4ELi2ELi2ELi0EE4dataEv(%"class.Eigen::DenseStorage"* %this, %"class.Eigen::DenseStorage"* %"this'") local_unnamed_addr #0 align 2 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"arraydecay'ipge" = getelementptr inbounds %"class.Eigen::DenseStorage", %"class.Eigen::DenseStorage"* %"this'", i64 0, i32 0, i32 0, i64 0
; CHECK-NEXT:   %arraydecay = getelementptr inbounds %"class.Eigen::DenseStorage", %"class.Eigen::DenseStorage"* %this, i64 0, i32 0, i32 0, i64 0
; CHECK-NEXT:   %.fca.1.insert = insertvalue { {}, double*, double* } undef, double* %arraydecay, 1
; CHECK-NEXT:   %.fca.2.insert = insertvalue { {}, double*, double* } %.fca.1.insert, double* %"arraydecay'ipge", 2
; CHECK-NEXT:   ret { {}, double*, double* } %.fca.2.insert
; CHECK-NEXT: }

; CHECK: define internal { { {} }, double*, double* } @augmented__ZNK5Eigen15PlainObjectBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE4dataEv(%"class.Eigen::PlainObjectBase"* %this, %"class.Eigen::PlainObjectBase"* %"this'") local_unnamed_addr #9 align 2 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"m_storage'ipge" = getelementptr inbounds %"class.Eigen::PlainObjectBase", %"class.Eigen::PlainObjectBase"* %"this'", i64 0, i32 0
; CHECK-NEXT:   %m_storage = getelementptr inbounds %"class.Eigen::PlainObjectBase", %"class.Eigen::PlainObjectBase"* %this, i64 0, i32 0
; CHECK-NEXT:   %call_augmented = call { {}, double*, double* } @augmented__ZNK5Eigen12DenseStorageIdLi4ELi2ELi2ELi0EE4dataEv(%"class.Eigen::DenseStorage"* %m_storage, %"class.Eigen::DenseStorage"* %"m_storage'ipge")
; CHECK-NEXT:   %antiptr_call = extractvalue { {}, double*, double* } %call_augmented, 2
; CHECK-NEXT:   %call = extractvalue { {}, double*, double* } %call_augmented, 1
; CHECK-NEXT:   %.fca.1.insert = insertvalue { { {} }, double*, double* } undef, double* %call, 1
; CHECK-NEXT:   %.fca.2.insert = insertvalue { { {} }, double*, double* } %.fca.1.insert, double* %antiptr_call, 2
; CHECK-NEXT:   ret { { {} }, double*, double* } %.fca.2.insert
; CHECK-NEXT: }

; CHECK: define internal { {} } @augmented__ZN5Eigen8internal11noncopyableC2Ev(%"class.Eigen::internal::noncopyable"* %this, %"class.Eigen::internal::noncopyable"* %"this'") unnamed_addr #0 align 2 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret { {} } undef
; CHECK-NEXT: }

; CHECK: define internal { { {} } } @augmented__ZN5Eigen8internal14evaluator_baseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEC2Ev(%"struct.Eigen::internal::evaluator_base"* %this, %"struct.Eigen::internal::evaluator_base"* %"this'") unnamed_addr #9 align 2 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = bitcast %"struct.Eigen::internal::evaluator_base"* %this to %"class.Eigen::internal::noncopyable"*
; CHECK-NEXT:   %"'ipc" = bitcast %"struct.Eigen::internal::evaluator_base"* %"this'" to %"class.Eigen::internal::noncopyable"*
; CHECK-NEXT:   %_augmented = call { {} } @augmented__ZN5Eigen8internal11noncopyableC2Ev(%"class.Eigen::internal::noncopyable"* %0, %"class.Eigen::internal::noncopyable"* %"'ipc")
; CHECK-NEXT:   ret { { {} } } undef
; CHECK-NEXT: }

; CHECK: define internal { { {}, { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64, { {} }, double*, double*, { {} } } } @augmented__ZN5Eigen8internal9evaluatorINS_15PlainObjectBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEEEC2ERKS5_(%"struct.Eigen::internal::evaluator.6"* %this, %"struct.Eigen::internal::evaluator.6"* %"this'", %"class.Eigen::PlainObjectBase"* dereferenceable(32) %m, %"class.Eigen::PlainObjectBase"* %"m'") unnamed_addr #2 align 2 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = bitcast %"struct.Eigen::internal::evaluator.6"* %this to %"struct.Eigen::internal::evaluator_base"*
; CHECK-NEXT:   %"'ipc2" = bitcast %"struct.Eigen::internal::evaluator.6"* %"this'" to %"struct.Eigen::internal::evaluator_base"*
; CHECK-NEXT:   %_augmented3 = call { { {} } } @augmented__ZN5Eigen8internal14evaluator_baseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEC2Ev(%"struct.Eigen::internal::evaluator_base"* %0, %"struct.Eigen::internal::evaluator_base"* %"'ipc2")
; CHECK-NEXT:   %"m_data'ipge" = getelementptr inbounds %"struct.Eigen::internal::evaluator.6", %"struct.Eigen::internal::evaluator.6"* %"this'", i64 0, i32 0
; CHECK-NEXT:   %m_data = getelementptr inbounds %"struct.Eigen::internal::evaluator.6", %"struct.Eigen::internal::evaluator.6"* %this, i64 0, i32 0
; CHECK-NEXT:   %call_augmented = call { { {} }, double*, double* } @augmented__ZNK5Eigen15PlainObjectBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE4dataEv(%"class.Eigen::PlainObjectBase"* nonnull %m, %"class.Eigen::PlainObjectBase"* %"m'")
; CHECK-NEXT:   %antiptr_call = extractvalue { { {} }, double*, double* } %call_augmented, 2
; CHECK-NEXT:   %call = extractvalue { { {} }, double*, double* } %call_augmented, 1
; CHECK-NEXT:   store double* %antiptr_call, double** %"m_data'ipge", align 8
; CHECK-NEXT:   store double* %call, double** %m_data, align 8, !tbaa !8
; CHECK-NEXT:   %"m_outerStride'ipge" = getelementptr inbounds %"struct.Eigen::internal::evaluator.6", %"struct.Eigen::internal::evaluator.6"* %"this'", i64 0, i32 1
; CHECK-NEXT:   %m_outerStride = getelementptr inbounds %"struct.Eigen::internal::evaluator.6", %"struct.Eigen::internal::evaluator.6"* %this, i64 0, i32 1
; CHECK-NEXT:   %1 = bitcast %"class.Eigen::PlainObjectBase"* %m to %"class.Eigen::DenseCoeffsBase"*
; CHECK-NEXT:   %"'ipc" = bitcast %"class.Eigen::PlainObjectBase"* %"m'" to %"class.Eigen::DenseCoeffsBase"*
; CHECK-NEXT:   %call2_augmented = call { { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64 } @augmented__ZNK5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEELi3EE11outerStrideEv(%"class.Eigen::DenseCoeffsBase"* %1, %"class.Eigen::DenseCoeffsBase"* %"'ipc")
; CHECK-NEXT:   %subcache = extractvalue { { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64 } %call2_augmented, 0
; CHECK-NEXT:   %subcache.fca.0.0.0.0.0.extract = extractvalue { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } %subcache, 0, 0, 0, 0, 0
; CHECK-NEXT:   %subcache.fca.0.0.0.2.extract = extractvalue { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } %subcache, 0, 0, 0, 2
; CHECK-NEXT:   %subcache.fca.0.0.0.3.extract = extractvalue { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } %subcache, 0, 0, 0, 3
; CHECK-NEXT:   %subcache.fca.2.extract = extractvalue { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } %subcache, 2
; CHECK-NEXT:   %subcache.fca.3.extract = extractvalue { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } %subcache, 3
; CHECK-NEXT:   %call2 = extractvalue { { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64 } %call2_augmented, 1
; CHECK-NEXT:   %_augmented = call { {} } @augmented__ZN5Eigen8internal19variable_if_dynamicIlLi2EEC2El(%"class.Eigen::internal::variable_if_dynamic"* nonnull %m_outerStride, %"class.Eigen::internal::variable_if_dynamic"* nonnull %"m_outerStride'ipge", i64 %call2)
; CHECK-NEXT:   %.fca.0.1.0.0.0.0.0.insert = insertvalue { { {}, { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64, { {} }, double*, double*, { {} } } } undef, i64 %subcache.fca.0.0.0.0.0.extract, 0, 1, 0, 0, 0, 0, 0
; CHECK-NEXT:   %.fca.0.1.0.0.0.2.insert = insertvalue { { {}, { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64, { {} }, double*, double*, { {} } } } %.fca.0.1.0.0.0.0.0.insert, %"class.Eigen::Matrix"* %subcache.fca.0.0.0.2.extract, 0, 1, 0, 0, 0, 2
; CHECK-NEXT:   %.fca.0.1.0.0.0.3.insert = insertvalue { { {}, { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64, { {} }, double*, double*, { {} } } } %.fca.0.1.0.0.0.2.insert, %"class.Eigen::Matrix"* %subcache.fca.0.0.0.3.extract, 0, 1, 0, 0, 0, 3
; CHECK-NEXT:   %.fca.0.1.2.insert = insertvalue { { {}, { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64, { {} }, double*, double*, { {} } } } %.fca.0.1.0.0.0.3.insert, %"class.Eigen::Matrix"* %subcache.fca.2.extract, 0, 1, 2
; CHECK-NEXT:   %.fca.0.1.3.insert = insertvalue { { {}, { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64, { {} }, double*, double*, { {} } } } %.fca.0.1.2.insert, %"class.Eigen::Matrix"* %subcache.fca.3.extract, 0, 1, 3
; CHECK-NEXT:   %.fca.0.2.insert = insertvalue { { {}, { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64, { {} }, double*, double*, { {} } } } %.fca.0.1.3.insert, i64 %call2, 0, 2
; CHECK-NEXT:   %.fca.0.4.insert = insertvalue { { {}, { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64, { {} }, double*, double*, { {} } } } %.fca.0.2.insert, double* %antiptr_call, 0, 4
; CHECK-NEXT:   %.fca.0.5.insert = insertvalue { { {}, { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64, { {} }, double*, double*, { {} } } } %.fca.0.4.insert, double* %call, 0, 5
; CHECK-NEXT:   ret { { {}, { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64, { {} }, double*, double*, { {} } } } %.fca.0.5.insert
; CHECK-NEXT: }

; CHECK: define internal { { { {}, { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64, { {} }, double*, double*, { {} } } } } @augmented__ZN5Eigen8internal9evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEC2ERKS3_(%"struct.Eigen::internal::evaluator"* %this, %"struct.Eigen::internal::evaluator"* %"this'", %"class.Eigen::Matrix"* dereferenceable(32) %m, %"class.Eigen::Matrix"* %"m'") unnamed_addr #2 align 2 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = bitcast %"struct.Eigen::internal::evaluator"* %this to %"struct.Eigen::internal::evaluator.6"*
; CHECK-NEXT:   %"'ipge" = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %"m'", i64 0, i32 0
; CHECK-NEXT:   %1 = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %m, i64 0, i32 0
; CHECK-NEXT:   %"'ipc" = bitcast %"struct.Eigen::internal::evaluator"* %"this'" to %"struct.Eigen::internal::evaluator.6"*
; CHECK-NEXT:   %_augmented = call { { {}, { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64, { {} }, double*, double*, { {} } } } @augmented__ZN5Eigen8internal9evaluatorINS_15PlainObjectBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEEEC2ERKS5_(%"struct.Eigen::internal::evaluator.6"* %0, %"struct.Eigen::internal::evaluator.6"* %"'ipc", %"class.Eigen::PlainObjectBase"* nonnull %1, %"class.Eigen::PlainObjectBase"* %"'ipge")
; CHECK-NEXT:   %subcache = extractvalue { { {}, { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64, { {} }, double*, double*, { {} } } } %_augmented, 0
; CHECK-NEXT:   %subcache.fca.1.0.0.0.0.0.extract = extractvalue { {}, { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64, { {} }, double*, double*, { {} } } %subcache, 1, 0, 0, 0, 0, 0
; CHECK-NEXT:   %subcache.fca.1.0.0.0.2.extract = extractvalue { {}, { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64, { {} }, double*, double*, { {} } } %subcache, 1, 0, 0, 0, 2
; CHECK-NEXT:   %subcache.fca.1.0.0.0.3.extract = extractvalue { {}, { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64, { {} }, double*, double*, { {} } } %subcache, 1, 0, 0, 0, 3
; CHECK-NEXT:   %subcache.fca.1.2.extract = extractvalue { {}, { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64, { {} }, double*, double*, { {} } } %subcache, 1, 2
; CHECK-NEXT:   %subcache.fca.1.3.extract = extractvalue { {}, { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64, { {} }, double*, double*, { {} } } %subcache, 1, 3
; CHECK-NEXT:   %subcache.fca.2.extract = extractvalue { {}, { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64, { {} }, double*, double*, { {} } } %subcache, 2
; CHECK-NEXT:   %subcache.fca.4.extract = extractvalue { {}, { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64, { {} }, double*, double*, { {} } } %subcache, 4
; CHECK-NEXT:   %subcache.fca.5.extract = extractvalue { {}, { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64, { {} }, double*, double*, { {} } } %subcache, 5
; CHECK-NEXT:   %.fca.0.0.1.0.0.0.0.0.insert = insertvalue { { { {}, { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64, { {} }, double*, double*, { {} } } } } undef, i64 %subcache.fca.1.0.0.0.0.0.extract, 0, 0, 1, 0, 0, 0, 0, 0
; CHECK-NEXT:   %.fca.0.0.1.0.0.0.2.insert = insertvalue { { { {}, { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64, { {} }, double*, double*, { {} } } } } %.fca.0.0.1.0.0.0.0.0.insert, %"class.Eigen::Matrix"* %subcache.fca.1.0.0.0.2.extract, 0, 0, 1, 0, 0, 0, 2
; CHECK-NEXT:   %.fca.0.0.1.0.0.0.3.insert = insertvalue { { { {}, { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64, { {} }, double*, double*, { {} } } } } %.fca.0.0.1.0.0.0.2.insert, %"class.Eigen::Matrix"* %subcache.fca.1.0.0.0.3.extract, 0, 0, 1, 0, 0, 0, 3
; CHECK-NEXT:   %.fca.0.0.1.2.insert = insertvalue { { { {}, { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64, { {} }, double*, double*, { {} } } } } %.fca.0.0.1.0.0.0.3.insert, %"class.Eigen::Matrix"* %subcache.fca.1.2.extract, 0, 0, 1, 2
; CHECK-NEXT:   %.fca.0.0.1.3.insert = insertvalue { { { {}, { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64, { {} }, double*, double*, { {} } } } } %.fca.0.0.1.2.insert, %"class.Eigen::Matrix"* %subcache.fca.1.3.extract, 0, 0, 1, 3
; CHECK-NEXT:   %.fca.0.0.2.insert = insertvalue { { { {}, { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64, { {} }, double*, double*, { {} } } } } %.fca.0.0.1.3.insert, i64 %subcache.fca.2.extract, 0, 0, 2
; CHECK-NEXT:   %.fca.0.0.4.insert = insertvalue { { { {}, { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64, { {} }, double*, double*, { {} } } } } %.fca.0.0.2.insert, double* %subcache.fca.4.extract, 0, 0, 4
; CHECK-NEXT:   %.fca.0.0.5.insert = insertvalue { { { {}, { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64, { {} }, double*, double*, { {} } } } } %.fca.0.0.4.insert, double* %subcache.fca.5.extract, 0, 0, 5
; CHECK-NEXT:   ret { { { {}, { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64, { {} }, double*, double*, { {} } } } } %.fca.0.0.5.insert
; CHECK-NEXT: }

; CHECK: define internal { { { { {}, { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64, { {} }, double*, double*, { {} } } } } } @augmented__ZN5Eigen8internal15redux_evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEC2ERKS3_(%"class.Eigen::internal::redux_evaluator"* %this, %"class.Eigen::internal::redux_evaluator"* %"this'", %"class.Eigen::Matrix"* dereferenceable(32) %xpr, %"class.Eigen::Matrix"* %"xpr'") unnamed_addr #2 align 2 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"m_evaluator'ipge" = getelementptr inbounds %"class.Eigen::internal::redux_evaluator", %"class.Eigen::internal::redux_evaluator"* %"this'", i64 0, i32 0
; CHECK-NEXT:   %m_evaluator = getelementptr inbounds %"class.Eigen::internal::redux_evaluator", %"class.Eigen::internal::redux_evaluator"* %this, i64 0, i32 0
; CHECK-NEXT:   %_augmented = call { { { {}, { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64, { {} }, double*, double*, { {} } } } } @augmented__ZN5Eigen8internal9evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEC2ERKS3_(%"struct.Eigen::internal::evaluator"* %m_evaluator, %"struct.Eigen::internal::evaluator"* %"m_evaluator'ipge", %"class.Eigen::Matrix"* nonnull %xpr, %"class.Eigen::Matrix"* %"xpr'")
; CHECK-NEXT:   %subcache = extractvalue { { { {}, { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64, { {} }, double*, double*, { {} } } } } %_augmented, 0
; CHECK-NEXT:   %subcache.fca.0.1.0.0.0.0.0.extract = extractvalue { { {}, { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64, { {} }, double*, double*, { {} } } } %subcache, 0, 1, 0, 0, 0, 0, 0
; CHECK-NEXT:   %subcache.fca.0.1.0.0.0.2.extract = extractvalue { { {}, { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64, { {} }, double*, double*, { {} } } } %subcache, 0, 1, 0, 0, 0, 2
; CHECK-NEXT:   %subcache.fca.0.1.0.0.0.3.extract = extractvalue { { {}, { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64, { {} }, double*, double*, { {} } } } %subcache, 0, 1, 0, 0, 0, 3
; CHECK-NEXT:   %subcache.fca.0.1.2.extract = extractvalue { { {}, { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64, { {} }, double*, double*, { {} } } } %subcache, 0, 1, 2
; CHECK-NEXT:   %subcache.fca.0.1.3.extract = extractvalue { { {}, { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64, { {} }, double*, double*, { {} } } } %subcache, 0, 1, 3
; CHECK-NEXT:   %subcache.fca.0.2.extract = extractvalue { { {}, { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64, { {} }, double*, double*, { {} } } } %subcache, 0, 2
; CHECK-NEXT:   %subcache.fca.0.4.extract = extractvalue { { {}, { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64, { {} }, double*, double*, { {} } } } %subcache, 0, 4
; CHECK-NEXT:   %subcache.fca.0.5.extract = extractvalue { { {}, { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64, { {} }, double*, double*, { {} } } } %subcache, 0, 5
; CHECK-NEXT:   %"m_xpr'ipge" = getelementptr inbounds %"class.Eigen::internal::redux_evaluator", %"class.Eigen::internal::redux_evaluator"* %"this'", i64 0, i32 1
; CHECK-NEXT:   %m_xpr = getelementptr inbounds %"class.Eigen::internal::redux_evaluator", %"class.Eigen::internal::redux_evaluator"* %this, i64 0, i32 1
; CHECK-NEXT:   store %"class.Eigen::Matrix"* %"xpr'", %"class.Eigen::Matrix"** %"m_xpr'ipge", align 8
; CHECK-NEXT:   store %"class.Eigen::Matrix"* %xpr, %"class.Eigen::Matrix"** %m_xpr, align 8, !tbaa !6
; CHECK-NEXT:   %.fca.0.0.0.1.0.0.0.0.0.insert = insertvalue { { { { {}, { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64, { {} }, double*, double*, { {} } } } } } undef, i64 %subcache.fca.0.1.0.0.0.0.0.extract, 0, 0, 0, 1, 0, 0, 0, 0, 0
; CHECK-NEXT:   %.fca.0.0.0.1.0.0.0.2.insert = insertvalue { { { { {}, { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64, { {} }, double*, double*, { {} } } } } } %.fca.0.0.0.1.0.0.0.0.0.insert, %"class.Eigen::Matrix"* %subcache.fca.0.1.0.0.0.2.extract, 0, 0, 0, 1, 0, 0, 0, 2
; CHECK-NEXT:   %.fca.0.0.0.1.0.0.0.3.insert = insertvalue { { { { {}, { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64, { {} }, double*, double*, { {} } } } } } %.fca.0.0.0.1.0.0.0.2.insert, %"class.Eigen::Matrix"* %subcache.fca.0.1.0.0.0.3.extract, 0, 0, 0, 1, 0, 0, 0, 3
; CHECK-NEXT:   %.fca.0.0.0.1.2.insert = insertvalue { { { { {}, { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64, { {} }, double*, double*, { {} } } } } } %.fca.0.0.0.1.0.0.0.3.insert, %"class.Eigen::Matrix"* %subcache.fca.0.1.2.extract, 0, 0, 0, 1, 2
; CHECK-NEXT:   %.fca.0.0.0.1.3.insert = insertvalue { { { { {}, { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64, { {} }, double*, double*, { {} } } } } } %.fca.0.0.0.1.2.insert, %"class.Eigen::Matrix"* %subcache.fca.0.1.3.extract, 0, 0, 0, 1, 3
; CHECK-NEXT:   %.fca.0.0.0.2.insert = insertvalue { { { { {}, { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64, { {} }, double*, double*, { {} } } } } } %.fca.0.0.0.1.3.insert, i64 %subcache.fca.0.2.extract, 0, 0, 0, 2
; CHECK-NEXT:   %.fca.0.0.0.4.insert = insertvalue { { { { {}, { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64, { {} }, double*, double*, { {} } } } } } %.fca.0.0.0.2.insert, double* %subcache.fca.0.4.extract, 0, 0, 0, 4
; CHECK-NEXT:   %.fca.0.0.0.5.insert = insertvalue { { { { {}, { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64, { {} }, double*, double*, { {} } } } } } %.fca.0.0.0.4.insert, double* %subcache.fca.0.5.extract, 0, 0, 0, 5
; CHECK-NEXT:   ret { { { { {}, { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64, { {} }, double*, double*, { {} } } } } } %.fca.0.0.0.5.insert
; CHECK-NEXT: }

; CHECK: define internal {} @diffe_ZN5Eigen8internal15redux_evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEC2ERKS3_(%"class.Eigen::internal::redux_evaluator"* %this, %"class.Eigen::internal::redux_evaluator"* %"this'", %"class.Eigen::Matrix"* dereferenceable(32) %xpr, %"class.Eigen::Matrix"* %"xpr'", { { { {}, { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64, { {} }, double*, double*, { {} } } } } %tapeArg) unnamed_addr #2 align 2 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"m_evaluator'ipge" = getelementptr inbounds %"class.Eigen::internal::redux_evaluator", %"class.Eigen::internal::redux_evaluator"* %"this'", i64 0, i32 0
; CHECK-NEXT:   %m_evaluator = getelementptr inbounds %"class.Eigen::internal::redux_evaluator", %"class.Eigen::internal::redux_evaluator"* %this, i64 0, i32 0
; CHECK-NEXT:   %0 = extractvalue { { { {}, { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64, { {} }, double*, double*, { {} } } } } %tapeArg, 0
; CHECK-NEXT:   %1 = call {} @diffe_ZN5Eigen8internal9evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEC2ERKS3_(%"struct.Eigen::internal::evaluator"* %m_evaluator, %"struct.Eigen::internal::evaluator"* %"m_evaluator'ipge", %"class.Eigen::Matrix"* nonnull %xpr, %"class.Eigen::Matrix"* %"xpr'", { { {}, { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64, { {} }, double*, double*, { {} } } } %0)
; CHECK-NEXT:   ret {} undef
; CHECK-NEXT: }

; CHECK: define internal {} @diffe_ZN5Eigen8internal9evaluatorINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEC2ERKS3_(%"struct.Eigen::internal::evaluator"* %this, %"struct.Eigen::internal::evaluator"* %"this'", %"class.Eigen::Matrix"* dereferenceable(32) %m, %"class.Eigen::Matrix"* %"m'", { { {}, { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64, { {} }, double*, double*, { {} } } } %tapeArg) unnamed_addr #2 align 2 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"'ipge" = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %"m'", i64 0, i32 0
; CHECK-NEXT:   %0 = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %m, i64 0, i32 0
; CHECK-NEXT:   %1 = extractvalue { { {}, { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64, { {} }, double*, double*, { {} } } } %tapeArg, 0
; CHECK-NEXT:   %_unwrap = bitcast %"struct.Eigen::internal::evaluator"* %this to %"struct.Eigen::internal::evaluator.6"*
; CHECK-NEXT:   %"'ipc" = bitcast %"struct.Eigen::internal::evaluator"* %"this'" to %"struct.Eigen::internal::evaluator.6"*
; CHECK-NEXT:   %2 = call {} @diffe_ZN5Eigen8internal9evaluatorINS_15PlainObjectBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEEEC2ERKS5_(%"struct.Eigen::internal::evaluator.6"* %_unwrap, %"struct.Eigen::internal::evaluator.6"* %"'ipc", %"class.Eigen::PlainObjectBase"* nonnull %0, %"class.Eigen::PlainObjectBase"* %"'ipge", { {}, { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64, { {} }, double*, double*, { {} } } %1)
; CHECK-NEXT:   ret {} undef
; CHECK-NEXT: }

; CHECK: define internal {} @diffe_ZN5Eigen8internal9evaluatorINS_15PlainObjectBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEEEC2ERKS5_(%"struct.Eigen::internal::evaluator.6"* %this, %"struct.Eigen::internal::evaluator.6"* %"this'", %"class.Eigen::PlainObjectBase"* dereferenceable(32) %m, %"class.Eigen::PlainObjectBase"* %"m'", { {}, { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64, { {} }, double*, double*, { {} } } %tapeArg) unnamed_addr #2 align 2 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"m_outerStride'ipge" = getelementptr inbounds %"struct.Eigen::internal::evaluator.6", %"struct.Eigen::internal::evaluator.6"* %"this'", i64 0, i32 1
; CHECK-NEXT:   %m_outerStride = getelementptr inbounds %"struct.Eigen::internal::evaluator.6", %"struct.Eigen::internal::evaluator.6"* %this, i64 0, i32 1
; CHECK-NEXT:   %0 = extractvalue { {}, { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64, { {} }, double*, double*, { {} } } %tapeArg, 1
; CHECK-NEXT:   %1 = extractvalue { {}, { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* }, i64, { {} }, double*, double*, { {} } } %tapeArg, 2
; CHECK-NEXT:   %2 = call {} @diffe_ZN5Eigen8internal19variable_if_dynamicIlLi2EEC2El(%"class.Eigen::internal::variable_if_dynamic"* nonnull %m_outerStride, %"class.Eigen::internal::variable_if_dynamic"* nonnull %"m_outerStride'ipge", i64 %1, {} undef)
; CHECK-NEXT:   %_unwrap = bitcast %"class.Eigen::PlainObjectBase"* %m to %"class.Eigen::DenseCoeffsBase"*
; CHECK-NEXT:   %"'ipc" = bitcast %"class.Eigen::PlainObjectBase"* %"m'" to %"class.Eigen::DenseCoeffsBase"*
; CHECK-NEXT:   %3 = call {} @diffe_ZNK5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEELi3EE11outerStrideEv(%"class.Eigen::DenseCoeffsBase"* %_unwrap, %"class.Eigen::DenseCoeffsBase"* %"'ipc", { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } %0)
; CHECK-NEXT:   %4 = call {} @diffe_ZNK5Eigen15PlainObjectBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE4dataEv(%"class.Eigen::PlainObjectBase"* nonnull %m, %"class.Eigen::PlainObjectBase"* %"m'", { {} } undef)
; CHECK-NEXT:   %_unwrap2 = bitcast %"struct.Eigen::internal::evaluator.6"* %this to %"struct.Eigen::internal::evaluator_base"*
; CHECK-NEXT:   %"'ipc3" = bitcast %"struct.Eigen::internal::evaluator.6"* %"this'" to %"struct.Eigen::internal::evaluator_base"*
; CHECK-NEXT:   %5 = call {} @diffe_ZN5Eigen8internal14evaluator_baseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEC2Ev(%"struct.Eigen::internal::evaluator_base"* %_unwrap2, %"struct.Eigen::internal::evaluator_base"* %"'ipc3", { {} } undef)
; CHECK-NEXT:   ret {} undef
; CHECK-NEXT: }

; CHECK: define internal {} @diffe_ZN5Eigen8internal19variable_if_dynamicIlLi2EEC2El(%"class.Eigen::internal::variable_if_dynamic"* %this, %"class.Eigen::internal::variable_if_dynamic"* %"this'", i64 %v, {} %tapeArg) unnamed_addr #3 align 2 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %cmp = icmp eq i64 %v, 2
; CHECK-NEXT:   br i1 %cmp, label %invertcond.end, label %cond.false
; CHECK: cond.false:                                       ; preds = %entry
; CHECK-NEXT:   tail call void @__assert_fail(i8* getelementptr inbounds ([14 x i8], [14 x i8]* @.str.8, i64 0, i64 0), i8* getelementptr inbounds ([58 x i8], [58 x i8]* @.str.9, i64 0, i64 0), i32 110, i8* getelementptr inbounds ([92 x i8], [92 x i8]* @__PRETTY_FUNCTION__._ZN5Eigen8internal19variable_if_dynamicIlLi2EEC2El, i64 0, i64 0)) #14
; CHECK-NEXT:   unreachable
; CHECK: invertcond.end:                                   ; preds = %entry
; CHECK-NEXT:   ret {} undef
; CHECK-NEXT: }

; CHECK: define internal {} @diffe_ZNK5Eigen15DenseCoeffsBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEELi3EE11outerStrideEv(%"class.Eigen::DenseCoeffsBase"* %this, %"class.Eigen::DenseCoeffsBase"* %"this'", { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } %tapeArg) local_unnamed_addr #9 align 2 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = extractvalue { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } %tapeArg, 3
; CHECK-NEXT:   %1 = extractvalue { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } %tapeArg, 0
; CHECK-NEXT:   %"call'ip_phi_fromtape_unwrap" = extractvalue { { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } %tapeArg, 2
; CHECK-NEXT:   %2 = call {} @diffe_ZNK5Eigen6MatrixIdLi2ELi2ELi0ELi2ELi2EE11outerStrideEv(%"class.Eigen::Matrix"* %0, %"class.Eigen::Matrix"* %"call'ip_phi_fromtape_unwrap", { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } } %1)
; CHECK-NEXT:   %_unwrap = bitcast %"class.Eigen::DenseCoeffsBase"* %this to %"struct.Eigen::EigenBase"*
; CHECK-NEXT:   %"'ipc" = bitcast %"class.Eigen::DenseCoeffsBase"* %"this'" to %"struct.Eigen::EigenBase"*
; CHECK-NEXT:   %3 = call {} @diffe_ZNK5Eigen9EigenBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE7derivedEv(%"struct.Eigen::EigenBase"* %_unwrap, %"struct.Eigen::EigenBase"* %"'ipc", {} undef)
; CHECK-NEXT:   ret {} undef
; CHECK-NEXT: }

; CHECK: define internal {} @diffe_ZNK5Eigen6MatrixIdLi2ELi2ELi0ELi2ELi2EE11outerStrideEv(%"class.Eigen::Matrix"* %this, %"class.Eigen::Matrix"* %"this'", { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } } %tapeArg) local_unnamed_addr #9 align 2 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = extractvalue { { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } } %tapeArg, 0
; CHECK-NEXT:   %_unwrap = bitcast %"class.Eigen::Matrix"* %this to %"class.Eigen::DenseBase"*
; CHECK-NEXT:   %"'ipc" = bitcast %"class.Eigen::Matrix"* %"this'" to %"class.Eigen::DenseBase"*
; CHECK-NEXT:   %1 = call {} @diffe_ZNK5Eigen9DenseBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE9innerSizeEv(%"class.Eigen::DenseBase"* %_unwrap, %"class.Eigen::DenseBase"* %"'ipc", { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } %0)
; CHECK-NEXT:   ret {} undef
; CHECK-NEXT: }

; CHECK: define internal {} @diffe_ZNK5Eigen9DenseBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE9innerSizeEv(%"class.Eigen::DenseBase"* %this, %"class.Eigen::DenseBase"* %"this'", { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } %tapeArg) local_unnamed_addr #0 align 2 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = extractvalue { { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } } %tapeArg, 0
; CHECK-NEXT:   %_unwrap = bitcast %"class.Eigen::DenseBase"* %this to %"struct.Eigen::EigenBase"*
; CHECK-NEXT:   %"'ipc" = bitcast %"class.Eigen::DenseBase"* %"this'" to %"struct.Eigen::EigenBase"*
; CHECK-NEXT:   %1 = call {} @diffe_ZNK5Eigen9EigenBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE4rowsEv(%"struct.Eigen::EigenBase"* %_unwrap, %"struct.Eigen::EigenBase"* %"'ipc", { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } %0)
; CHECK-NEXT:   ret {} undef
; CHECK-NEXT: }

; CHECK: define internal {} @diffe_ZNK5Eigen9EigenBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE4rowsEv(%"struct.Eigen::EigenBase"* %this, %"struct.Eigen::EigenBase"* %"this'", { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } %tapeArg) local_unnamed_addr #9 align 2 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = extractvalue { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } %tapeArg, 3
; CHECK-NEXT:   %"call'ip_phi" = extractvalue { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } %tapeArg, 2
; CHECK-NEXT:   %"'ipge" = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %"call'ip_phi", i64 0, i32 0
; CHECK-NEXT:   %1 = getelementptr inbounds %"class.Eigen::Matrix", %"class.Eigen::Matrix"* %0, i64 0, i32 0
; CHECK-NEXT:   %2 = extractvalue { { i64 }, {}, %"class.Eigen::Matrix"*, %"class.Eigen::Matrix"* } %tapeArg, 0
; CHECK-NEXT:   %3 = call {} @diffe_ZNK5Eigen15PlainObjectBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE4rowsEv(%"class.Eigen::PlainObjectBase"* %1, %"class.Eigen::PlainObjectBase"* %"'ipge", { i64 } %2)
; CHECK-NEXT:   %4 = call {} @diffe_ZNK5Eigen9EigenBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE7derivedEv(%"struct.Eigen::EigenBase"* %this, %"struct.Eigen::EigenBase"* %"this'", {} undef)
; CHECK-NEXT:   ret {} undef
; CHECK-NEXT: }

; CHECK: define internal {} @diffe_ZNK5Eigen15PlainObjectBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE4rowsEv(%"class.Eigen::PlainObjectBase"* %this, %"class.Eigen::PlainObjectBase"* %"this'", { i64 } %tapeArg) local_unnamed_addr #9 align 2 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret {} undef
; CHECK-NEXT: }

; CHECK: define internal {} @diffe_ZNK5Eigen9EigenBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE7derivedEv(%"struct.Eigen::EigenBase"* %this, %"struct.Eigen::EigenBase"* %"this'", {} %tapeArg) local_unnamed_addr #0 align 2 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret {} undef
; CHECK-NEXT: }

; CHECK: define internal {} @diffe_ZNK5Eigen15PlainObjectBaseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEE4dataEv(%"class.Eigen::PlainObjectBase"* %this, %"class.Eigen::PlainObjectBase"* %"this'", { {} } %tapeArg) local_unnamed_addr #9 align 2 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"m_storage'ipge" = getelementptr inbounds %"class.Eigen::PlainObjectBase", %"class.Eigen::PlainObjectBase"* %"this'", i64 0, i32 0
; CHECK-NEXT:   %m_storage = getelementptr inbounds %"class.Eigen::PlainObjectBase", %"class.Eigen::PlainObjectBase"* %this, i64 0, i32 0
; CHECK-NEXT:   %0 = call {} @diffe_ZNK5Eigen12DenseStorageIdLi4ELi2ELi2ELi0EE4dataEv(%"class.Eigen::DenseStorage"* %m_storage, %"class.Eigen::DenseStorage"* %"m_storage'ipge", {} undef)
; CHECK-NEXT:   ret {} undef
; CHECK-NEXT: }

; CHECK: define internal {} @diffe_ZNK5Eigen12DenseStorageIdLi4ELi2ELi2ELi0EE4dataEv(%"class.Eigen::DenseStorage"* %this, %"class.Eigen::DenseStorage"* %"this'", {} %tapeArg) local_unnamed_addr #0 align 2 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret {} undef
; CHECK-NEXT: }

; CHECK: define internal {} @diffe_ZN5Eigen8internal14evaluator_baseINS_6MatrixIdLi2ELi2ELi0ELi2ELi2EEEEC2Ev(%"struct.Eigen::internal::evaluator_base"* %this, %"struct.Eigen::internal::evaluator_base"* %"this'", { {} } %tapeArg) unnamed_addr #9 align 2 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %_unwrap = bitcast %"struct.Eigen::internal::evaluator_base"* %this to %"class.Eigen::internal::noncopyable"*
; CHECK-NEXT:   %"'ipc" = bitcast %"struct.Eigen::internal::evaluator_base"* %"this'" to %"class.Eigen::internal::noncopyable"*
; CHECK-NEXT:   %0 = call {} @diffe_ZN5Eigen8internal11noncopyableC2Ev(%"class.Eigen::internal::noncopyable"* %_unwrap, %"class.Eigen::internal::noncopyable"* %"'ipc", {} undef)
; CHECK-NEXT:   ret {} undef
; CHECK-NEXT: }

; CHECK: define internal {} @diffe_ZN5Eigen8internal11noncopyableC2Ev(%"class.Eigen::internal::noncopyable"* %this, %"class.Eigen::internal::noncopyable"* %"this'", {} %tapeArg) unnamed_addr #0 align 2 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret {} undef
; CHECK-NEXT: }
