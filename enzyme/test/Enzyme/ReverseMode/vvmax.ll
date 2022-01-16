; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -instsimplify -S | FileCheck %s

; ModuleID = '<stdin>'
source_filename = "/home/runner/work/Enzyme/Enzyme/enzyme/test/Integration/multivecmax.cpp"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

; Function Attrs: norecurse nounwind uwtable
define void @caller(double* %x, double* %dx) {
entry:
  call void @_Z17__enzyme_autodiffPvPdS0_i(i8* bitcast (double (double*, i64)* @rmax to i8*), double* nonnull %x, double* nonnull %dx, i64 5)
  ret void
}

declare void @_Z17__enzyme_autodiffPvPdS0_i(i8*, double*, double*, i64)

; Function Attrs: nobuiltin
declare dso_local noalias nonnull i8* @_Znwm(i64) local_unnamed_addr #1

; Function Attrs: nounwind uwtable
define double @rmax(double* %arg, i64 %arg1) {
entry:
  %c1 = icmp sgt i64 %arg1, 0
  br i1 %c1, label %bb11, label %end

bb11:                                             ; preds = %bb78, %bb3
  %tiv9 = phi i64 [ %tiv.next10, %bb11 ], [ 0, %entry ]
  %tiv.next10 = add nuw nsw i64 %tiv9, 1
  %tmp50 = call i8* @_Znwm(i64 8) #2
  %tmp51 = bitcast i8* %tmp50 to double*
  %tmp80 = icmp eq i64 %tiv.next10, %arg1
  br i1 %tmp80, label %bb81, label %bb11

bb81:                                             ; preds = %.loopexit, %bb5
  %tmp87 = icmp eq double* %tmp51, %arg
  br i1 %tmp87, label %end, label %bb88

bb88:                                             ; preds = %bb81
  br label %end

end:                                             ; preds = %bb88, %bb81, %bb
  %tmp91 = phi double [ 1.000000e+00, %bb81 ], [ 2.000000e+00, %bb88 ], [ 0.000000e+00, %entry ]
  ret double %tmp91
}

attributes #0 = { argmemonly nounwind }
attributes #1 = { nobuiltin }
attributes #2 = { nounwind }

!0 = !{!1, !1, i64 0}
!1 = !{!"double", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C++ TBAA"}

; CHECK: define internal void @differmax(double* %arg, double* %"arg'", i64 %arg1, double %differeturn)
; CHECK:  %"tmp51!manual_lcssa{{.*}}" = phi double* [ %tmp51, %bb88 ], [ %tmp51, %bb81 ], [ undef, %entry ]
; OLD:  %"tmp50!manual_lcssa" = phi i8* [ %tmp50, %bb81 ], [ %tmp50, %bb88 ], [ undef, %entry ]
