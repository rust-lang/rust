; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -adce -loop-deletion -correlated-propagation -simplifycfg -S | FileCheck %s

source_filename = "/home/wmoses/git/Enzyme/enzyme/lulesh/RAJAProxies/lulesh-v2.0/RAJA/lulesh.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable willreturn mustprogress
define void @_ZN6Domain1xEl({ double*, i64 }* %_M_start.i, i64 %idx) {
entry:
  %gep = getelementptr inbounds { double*, i64 }, { double*, i64 }* %_M_start.i, i64 0, i32 0
  %i = load double*, double** %gep, align 8, !tbaa !4
  %add.ptr.i = getelementptr inbounds double, double* %i, i64 %idx
  store double 1.000000e+00, double* %add.ptr.i, align 8
  ret void
}

define void @caller(i8* %call21, i8* %call30) local_unnamed_addr {
entry:
  call void @_Z17__enzyme_autodiffPvS_S_(i8* bitcast (void ({ double*, i64 }*)* @_ZL16LagrangeLeapFrogP6Domain to i8*), i8* nonnull %call21, i8* nonnull %call30)
  ret void
}

declare void @_Z17__enzyme_autodiffPvS_S_(i8*, i8*, i8*)

; Function Attrs: inlinehint uwtable mustprogress
define internal void @_ZL16LagrangeLeapFrogP6Domain({ double*, i64 }* %domain) {
entry:
  %m_sizeZ.i = getelementptr inbounds { double*, i64 }, { double*, i64 }* %domain, i64 0, i32 1
  %i3 = load i64, i64* %m_sizeZ.i, align 8, !tbaa !10
  br label %for.body53.us

for.body53.us:                                    ; preds = %for.body53.us, %entry
  %i = phi i64 [ %inc.us, %for.body53.us ], [ 0, %entry ]
  call void @_ZN6Domain1xEl({ double*, i64 }* %domain, i64 %i)
  %inc.us = add nuw i64 %i, 1
  %exitcond161.not = icmp eq i64 %i, %i3
  br i1 %exitcond161.not, label %for.cond.cleanup52, label %for.body53.us

for.cond.cleanup52:                               ; preds = %for.body53.us
  ret void
}

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{!"clang version 12.0.1 (git@github.com:llvm/llvm-project 4973ce53ca8abfc14233a3d8b3045673e0e8543c)"}
!4 = !{!5, !7, i64 0}
!5 = !{!"_ZTSSt12_Vector_baseIdSaIdEE", !6, i64 0}
!6 = !{!"_ZTSNSt12_Vector_baseIdSaIdEE12_Vector_implE", !7, i64 0, !7, i64 8, !7, i64 16}
!7 = !{!"any pointer", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C++ TBAA"}
!10 = !{!11, !11, i64 0}
!11 = !{!"long", !8, i64 0}

; CHECK: define internal void @diffe_ZL16LagrangeLeapFrogP6Domain({ double*, i64 }* %domain, { double*, i64 }* %"domain'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %m_sizeZ.i = getelementptr inbounds { double*, i64 }, { double*, i64 }* %domain, i64 0, i32 1
; CHECK-NEXT:   %i3 = load i64, i64* %m_sizeZ.i, align 8, !tbaa !10
; CHECK-NEXT:   %0 = add nuw i64 %i3, 1
; CHECK-NEXT:   %mallocsize = mul nuw nsw i64 %0, 8
; CHECK-NEXT:   %malloccall = tail call noalias nonnull i8* @malloc(i64 %mallocsize)
; CHECK-NEXT:   %_augmented_malloccache = bitcast i8* %malloccall to double**
; CHECK-NEXT:   br label %for.body53.us

; CHECK: for.body53.us:                                    ; preds = %for.body53.us, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %for.body53.us ], [ 0, %entry ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %_augmented = call double* @augmented__ZN6Domain1xEl({ double*, i64 }* %domain, { double*, i64 }* %"domain'", i64 %iv)
; CHECK-NEXT:   %1 = getelementptr inbounds double*, double** %_augmented_malloccache, i64 %iv
; CHECK-NEXT:   store double* %_augmented, double** %1
; CHECK-NEXT:   %exitcond161.not = icmp eq i64 %iv, %i3
; CHECK-NEXT:   br i1 %exitcond161.not, label %invertfor.body53.us, label %for.body53.us

; CHECK: invertentry:                                      ; preds = %invertfor.body53.us
; CHECK-NEXT:   tail call void @free(i8* nonnull %malloccall)
; CHECK-NEXT:   ret void

; CHECK: invertfor.body53.us:                              ; preds = %for.body53.us, %incinvertfor.body53.us
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %5, %incinvertfor.body53.us ], [ %i3, %for.body53.us ]
; CHECK-NEXT:   %2 = getelementptr inbounds double*, double** %_augmented_malloccache, i64 %"iv'ac.0"
; CHECK-NEXT:   %3 = load double*, double** %2
; CHECK-NEXT:   call void @diffe_ZN6Domain1xEl({ double*, i64 }* %domain, { double*, i64 }* %"domain'", i64 %"iv'ac.0", double* %3)
; CHECK-NEXT:   %4 = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %4, label %invertentry, label %incinvertfor.body53.us

; CHECK: incinvertfor.body53.us:                           ; preds = %invertfor.body53.us
; CHECK-NEXT:   %5 = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertfor.body53.us
; CHECK-NEXT: }
