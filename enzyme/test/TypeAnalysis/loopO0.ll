; RUN: %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=callee -o /dev/null | FileCheck %s

source_filename = "/mnt/pci4/wmdata/Enzyme2/enzyme/test/Integration/ReverseMode/eigentensor.cpp"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%"struct.Eigen::TensorEvaluator.10" = type { [4 x i64], { float* } }

define internal void @callee() {
entry:
  %m_rightImpl.i = alloca %"struct.Eigen::TensorEvaluator.10", align 8
  %tmp15 = getelementptr inbounds %"struct.Eigen::TensorEvaluator.10", %"struct.Eigen::TensorEvaluator.10"* %m_rightImpl.i, i32 0, i32 0
  br label %for.cond.i103

for.cond.i103:                                    ; preds = %for.body.i107, %entry
  %iv = phi i64 [ %iv.next, %for.body.i107 ], [ 0, %entry ]
  %iv.next = add nuw nsw i64 %iv, 1
  %cmp.i102 = icmp slt i64 %iv, 4
  br i1 %cmp.i102, label %for.body.i107, label %_ZN5Eigen6DSizesIlLi4EEC2Ev.exit

for.body.i107:                                    ; preds = %for.cond.i103
  %arrayidx.i = getelementptr inbounds [4 x i64], [4 x i64]* %tmp15, i64 0, i64 %iv
  store i64 0, i64* %arrayidx.i, align 8, !tbaa !2
  br label %for.cond.i103

_ZN5Eigen6DSizesIlLi4EEC2Ev.exit:                 ; preds = %for.cond.i103
  %m_kernelArg.i = getelementptr inbounds %"struct.Eigen::TensorEvaluator.10", %"struct.Eigen::TensorEvaluator.10"* %m_rightImpl.i, i32 0, i32 1
  %m_data.i123 = getelementptr inbounds { float* }, { float* }* %m_kernelArg.i, i32 0, i32 0
  store float* null, float** %m_data.i123, align 8, !tbaa !6
  ret void
}

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0, i64 8}
!3 = !{!4, i64 8, !"long"}
!4 = !{!5, i64 1, !"omnipotent char"}
!5 = !{!"Simple C++ TBAA"}
!6 = !{!7, !8, i64 0, i64 8}
!7 = !{!4, i64 24, !"_ZTSN5Eigen13TensorStorageIfNS_6DSizesIlLi2EEELi0EEE", !8, i64 0, i64 8, !9, i64 8, i64 16}
!8 = !{!4, i64 8, !"any pointer"}
!9 = !{!4, i64 16, !"_ZTSN5Eigen6DSizesIlLi2EEE"}

; CHECK: callee - {} |
; CHECK-NEXT: entry
; CHECK-NEXT:   %m_rightImpl.i = alloca %"struct.Eigen::TensorEvaluator.10", align 8: {[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer, [-1,7]:Integer, [-1,8]:Integer, [-1,9]:Integer, [-1,10]:Integer, [-1,11]:Integer, [-1,12]:Integer, [-1,13]:Integer, [-1,14]:Integer, [-1,15]:Integer, [-1,16]:Integer, [-1,17]:Integer, [-1,18]:Integer, [-1,19]:Integer, [-1,20]:Integer, [-1,21]:Integer, [-1,22]:Integer, [-1,23]:Integer, [-1,24]:Integer, [-1,25]:Integer, [-1,26]:Integer, [-1,27]:Integer, [-1,28]:Integer, [-1,29]:Integer, [-1,30]:Integer, [-1,31]:Integer, [-1,32]:Pointer}
; CHECK-NEXT:   %tmp15 = getelementptr inbounds %"struct.Eigen::TensorEvaluator.10", %"struct.Eigen::TensorEvaluator.10"* %m_rightImpl.i, i32 0, i32 0: {[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer, [-1,7]:Integer, [-1,8]:Integer, [-1,9]:Integer, [-1,10]:Integer, [-1,11]:Integer, [-1,12]:Integer, [-1,13]:Integer, [-1,14]:Integer, [-1,15]:Integer, [-1,16]:Integer, [-1,17]:Integer, [-1,18]:Integer, [-1,19]:Integer, [-1,20]:Integer, [-1,21]:Integer, [-1,22]:Integer, [-1,23]:Integer, [-1,24]:Integer, [-1,25]:Integer, [-1,26]:Integer, [-1,27]:Integer, [-1,28]:Integer, [-1,29]:Integer, [-1,30]:Integer, [-1,31]:Integer}
; CHECK-NEXT:   br label %for.cond.i103: {}
; CHECK-NEXT: for.cond.i103
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %for.body.i107 ], [ 0, %entry ]: {[-1]:Integer}
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1: {[-1]:Integer}
; CHECK-NEXT:   %cmp.i102 = icmp slt i64 %iv, 4: {[-1]:Integer}
; CHECK-NEXT:   br i1 %cmp.i102, label %for.body.i107, label %_ZN5Eigen6DSizesIlLi4EEC2Ev.exit: {}
; CHECK-NEXT: for.body.i107
; CHECK-NEXT:   %arrayidx.i = getelementptr inbounds [4 x i64], [4 x i64]* %tmp15, i64 0, i64 %iv: {[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer, [-1,7]:Integer}
; CHECK-NEXT:   store i64 0, i64* %arrayidx.i, align 8, !tbaa !2: {}
; CHECK-NEXT:   br label %for.cond.i103: {}
; CHECK-NEXT: _ZN5Eigen6DSizesIlLi4EEC2Ev.exit
; CHECK-NEXT:   %m_kernelArg.i = getelementptr inbounds %"struct.Eigen::TensorEvaluator.10", %"struct.Eigen::TensorEvaluator.10"* %m_rightImpl.i, i32 0, i32 1: {[-1]:Pointer, [-1,0]:Pointer}
; CHECK-NEXT:   %m_data.i123 = getelementptr inbounds { float* }, { float* }* %m_kernelArg.i, i32 0, i32 0: {[-1]:Pointer, [-1,0]:Pointer}
; CHECK-NEXT:   store float* null, float** %m_data.i123, align 8, !tbaa !6: {}
; CHECK-NEXT:   ret void: {}
