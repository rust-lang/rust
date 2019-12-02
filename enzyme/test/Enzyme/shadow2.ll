; RUN: %opt < %s %loadEnzyme -enzyme -enzyme_preopt=false -mem2reg -sroa -instsimplify -simplifycfg -S | FileCheck %s

declare double @__enzyme_autodiff(i8*, ...)

define void @caller(i64* %mat, i64* %dmat) {
entry:
  %call11 = call double (i8*, ...) @__enzyme_autodiff(i8* bitcast (double (i64*)* @myfunc to i8*), metadata !"diffe_dup", i64* %mat, i64* %dmat)
  ret void
}

define double @myfunc(i64* %mat) {
entry:
  %call642 = call i64 @zz(i64* nonnull %mat)
  %cmp743 = icmp sgt i64 %call642, 1
  br i1 %cmp743, label %for.cond.cleanup13, label %for.cond.cleanup8

for.cond.cleanup13:                               ; preds = %for.body14, %for.cond10.preheader
  br label %for.cond.cleanup8

for.cond.cleanup8: 
  %result = phi double [ 0.000000e+00, %for.cond.cleanup13 ], [ 1.000000e+00, %entry ]
  ret double %result
}

define i64 @zz(i64* %this) {
entry:
  %call = call i64 @loader(i64* %this)
  ret i64 %call
}

; Function Attrs: norecurse nounwind uwtable
define i64 @loader(i64* %m_cols) {
entry:
  %0 = load i64, i64* %m_cols, align 8, !tbaa !15
  ret i64 %0
}

!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C++ TBAA"}
!9 = !{!"long", !4, i64 0}
!7 = !{!"any pointer", !4, i64 0}
!14 = !{!"_ZTSN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EEE", !7, i64 0, !9, i64 8, !9, i64 16}
!15 = !{!14, !9, i64 16}

; CHECK: @diffemyfunc
