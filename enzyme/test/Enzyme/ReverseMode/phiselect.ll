; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -early-cse-memssa -simplifycfg -instsimplify -adce -simplifycfg -loop-deletion -simplifycfg -S | FileCheck %s

; ModuleID = 'wa.cpp'
source_filename = "wa.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@enzyme_dup = dso_local local_unnamed_addr global i32 0, align 4
@enzyme_out = dso_local local_unnamed_addr global i32 0, align 4
@enzyme_const = dso_local local_unnamed_addr global i32 0, align 4

; Function Attrs: norecurse nounwind readonly uwtable willreturn mustprogress
define dso_local double @_Z3fooPdy(double* noalias nocapture readonly %in, i64 %N) {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %_Z13cubicSpline3ddd.exit
  %out.010 = phi double [ %add, %_Z13cubicSpline3ddd.exit ], [ undef, %entry ]
  %i.09 = phi i64 [ %inc, %_Z13cubicSpline3ddd.exit ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds double, double* %in, i64 %i.09
  %0 = load double, double* %arrayidx, align 8, !tbaa !2
  %cmp.i = fcmp ugt double %0, 1.000000e+00
  br i1 %cmp.i, label %if.else.i, label %if.then.i

if.then.i:                                        ; preds = %for.body
  %mul4.i = fmul double %0, 1.500000e+00
  %mul5.i = fmul double %0, %mul4.i
  %div6.i = fmul double %0, 5.000000e-01
  %sub.i = fsub double 1.000000e+00, %div6.i
  %mul7.i = fmul double %mul5.i, %sub.i
  %sub8.i = fsub double 1.000000e+00, %mul7.i
  %mul9.i = fmul double %sub8.i, 0x3FD461D59AE78A99
  br label %_Z13cubicSpline3ddd.exit

if.else.i:                                        ; preds = %for.body
  %cmp10.i = fcmp ugt double %0, 2.000000e+00
  br i1 %cmp10.i, label %_Z13cubicSpline3ddd.exit, label %if.then11.i

if.then11.i:                                      ; preds = %if.else.i
  %sub12.i = fsub double 2.000000e+00, %0
  %mul14.i = fmul double %sub12.i, 0x3FB461D59AE78A99
  %mul15.i = fmul double %sub12.i, %mul14.i
  %mul16.i = fmul double %sub12.i, %mul15.i
  br label %_Z13cubicSpline3ddd.exit

_Z13cubicSpline3ddd.exit:                         ; preds = %if.then.i, %if.else.i, %if.then11.i
  %retval.0.i = phi double [ %mul9.i, %if.then.i ], [ %mul16.i, %if.then11.i ], [ 0.000000e+00, %if.else.i ]
  %mul = fmul double %retval.0.i, %retval.0.i
  %add = fadd double %out.010, %mul
  %inc = add nuw i64 %i.09, 1
  %exitcond.not = icmp eq i64 %inc, %N
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body, !llvm.loop !6

for.cond.cleanup:                                 ; preds = %_Z13cubicSpline3ddd.exit, %entry
  ret double %add
}

; Function Attrs: uwtable mustprogress
define void @_Z6callerPdS_y(double* %x, double* %d_x, i64 %N) {
entry:
  %0 = load i32, i32* @enzyme_dup, align 4, !tbaa !9
  %1 = load i32, i32* @enzyme_const, align 4, !tbaa !9
  %call = tail call double @_Z17__enzyme_autodiffPviPdS0_iy(i8* bitcast (double (double*, i64)* @_Z3fooPdy to i8*), i32 %0, double* %x, double* %d_x, i32 %1, i64 %N)
  ret void
}

declare double @_Z17__enzyme_autodiffPviPdS0_iy(i8*, i32, double*, double*, i32, i64)

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 13.0.0 (git@github.com:llvm/llvm-project b3015368039a55ad88270aa8fa5b9c5bd83aae15)"}
!2 = !{!3, !3, i64 0}
!3 = !{!"double", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C++ TBAA"}
!6 = distinct !{!6, !7, !8}
!7 = !{!"llvm.loop.mustprogress"}
!8 = !{!"llvm.loop.unroll.disable"}
!9 = !{!10, !10, i64 0}
!10 = !{!"int", !4, i64 0}

; CHECK: define internal void @diffe_Z3fooPdy(double* noalias nocapture readonly %in, double* nocapture %"in'", i64 %N, double %differeturn)
; CHECK-NOT: malloc