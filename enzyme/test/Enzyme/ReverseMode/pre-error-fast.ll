; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -adce -loop-deletion -correlated-propagation -simplifycfg -S | FileCheck %s

source_filename = "/mnt/Data/git/Enzyme/enzyme/test/Integration/eigensumsqdyn.cpp"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare double @__enzyme_autodiff(...)

define i32 @caller(double* %A, double* %Ap, i64* %B, i64* %Bp) {
  %call = call double (...) @__enzyme_autodiff(i8* bitcast (double (double*, i64*)* @matvec to i8*), double* %A, double* %Ap, i64* %B)
  ret i32 0
}

; Function Attrs: noinline nounwind uwtable
define internal double @matvec(double* %place, i64* %m_rows) {
entry:
  call void @subfn(double* %place, i64* nonnull %m_rows)
  %r1 = load double, double* %place, align 8, !tbaa !2
  %c2 = load i64, i64* %m_rows, align 8, !tbaa !6
  %cmp64.i.i = icmp sgt i64 %c2, 1
  br i1 %cmp64.i.i, label %for.body.i.i, label %exit

for.body.i.i:                                     ; preds = %entry
  %arrayidx.i.i45.i.i = getelementptr inbounds double, double* %place, i64 1
  %z3 = load double, double* %arrayidx.i.i45.i.i, align 8, !tbaa !2
  %add.i42.i.i = fadd double %r1, %z3
  br label %exit

exit:                                             ; preds = %for.body.i.i, %entry
  %res.0.lcssa.i.i = phi double [ %r1, %entry ], [ %add.i42.i.i, %for.body.i.i ]
  ret double %res.0.lcssa.i.i
}

define linkonce_odr dso_local void @subfn(double* %place, i64* %m_rows) {
entry:
  %rows = load i64, i64* %m_rows, align 8
  br label %for1

for1:                                             ; preds = %end2, %entry
  %i = phi i64 [ 0, %entry ], [ %nexti, %end2 ]
  %nexti = add nuw nsw i64 %i, 1
  br label %for2

for2:                                             ; preds = %for2, %for1
  %j = phi i64 [ %nextj, %for2 ], [ 1, %for1 ]
  %res = phi double [ %add, %for2 ], [ 0.000000e+00, %for1 ]
  %nextj = add nuw nsw i64 %j, 1
  %arrayidx = getelementptr inbounds double, double* %place, i64 %j
  %loaded = load double, double* %arrayidx, align 8, !tbaa !2
  %mul = fmul double %loaded, 2.000000e+00
  %add = fadd double %res, %mul
  %cond2 = icmp eq i64 %nextj, %rows
  br i1 %cond2, label %end2, label %for2

end2:                                             ; preds = %for2
  %tostore = getelementptr inbounds double, double* %place, i64 %i
  store double %add, double* %tostore, align 8, !tbaa !2
  %cond1 = icmp eq i64 %nexti, 4
  br i1 %cond1, label %exit, label %for1

exit:                                             ; preds = %end2
  ret void
}

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0}
!3 = !{!"double", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C++ TBAA"}
!6 = !{!7, !9, i64 8}
!7 = !{!"_ZTSN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EEE", !8, i64 0, !9, i64 8, !9, i64 16}
!8 = !{!"any pointer", !4, i64 0}
!9 = !{!"long", !4, i64 0}

; CHECK: define internal i64 @augmented_subfn(double* %place, double* %"place'", i64* %m_rows)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %rows = load i64, i64* %m_rows, align 8
; CHECK-NEXT:   br label %for1

; CHECK: for1:                                             ; preds = %end2, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %end2 ], [ 0, %entry ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   br label %for2

; CHECK: for2:                                             ; preds = %for2, %for1
; CHECK-NEXT:   %iv1 = phi i64 [ %iv.next2, %for2 ], [ 0, %for1 ]
; CHECK-NEXT:   %res = phi double [ %add, %for2 ], [ 0.000000e+00, %for1 ]
; CHECK-NEXT:   %iv.next2 = add nuw nsw i64 %iv1, 1
; CHECK-NEXT:   %nextj = add nuw nsw i64 %iv.next2, 1
; CHECK-NEXT:   %arrayidx = getelementptr inbounds double, double* %place, i64 %iv.next2
; CHECK-NEXT:   %loaded = load double, double* %arrayidx, align 8, !tbaa !2
; CHECK-NEXT:   %mul = fmul double %loaded, 2.000000e+00
; CHECK-NEXT:   %add = fadd double %res, %mul
; CHECK-NEXT:   %cond2 = icmp eq i64 %nextj, %rows
; CHECK-NEXT:   br i1 %cond2, label %end2, label %for2

; CHECK: end2:                                             ; preds = %for2
; CHECK-NEXT:   %tostore = getelementptr inbounds double, double* %place, i64 %iv
; CHECK-NEXT:   store double %add, double* %tostore, align 8, !tbaa !2
; CHECK-NEXT:   %cond1 = icmp eq i64 %iv.next, 4
; CHECK-NEXT:   br i1 %cond1, label %exit, label %for1

; CHECK: exit:                                             ; preds = %end2
; CHECK-NEXT:   ret i64 %rows
; CHECK-NEXT: }

; CHECK: define internal void @diffesubfn(double* %place, double* %"place'", i64* %m_rows, i64 %rows)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[_unwrap:.+]] = add i64 %rows, -2
; CHECK-NEXT:   br label %invertend2

; CHECK: invertentry:                                      ; preds = %invertfor1
; CHECK-NEXT:   ret void

; CHECK: invertfor1:                                       ; preds = %invertfor2
; CHECK-NEXT:   %[[eq1:.+]] = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %[[eq1]], label %invertentry, label %incinvertfor1

; CHECK: incinvertfor1:                                    ; preds = %invertfor1
; CHECK-NEXT:   %[[sub1:.+]] = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertend2

; CHECK: invertfor2:                                       ; preds = %invertend2, %incinvertfor2
; CHECK-NEXT:   %"iv1'ac.0" = phi i64 [ %[[_unwrap:.+]], %invertend2 ], [ %[[sub:.+]], %incinvertfor2 ]
; CHECK-NEXT:   %m0diffeloaded = fmul fast double %[[addde:.+]], 2.000000e+00
; CHECK-NEXT:   %iv.next2_unwrap = add nuw nsw i64 %"iv1'ac.0", 1
; CHECK-NEXT:   %[[arrayidxipg:.+]] = getelementptr inbounds double, double* %"place'", i64 %iv.next2_unwrap
; CHECK-NEXT:   %[[ld:.+]] = load double, double* %[[arrayidxipg]], align 8
; CHECK-NEXT:   %[[ldadd:.+]] = fadd fast double %[[ld]], %m0diffeloaded
; CHECK-NEXT:   store double %[[ldadd]], double* %[[arrayidxipg]], align 8
; CHECK-NEXT:   %[[eq2:.+]] = icmp eq i64 %"iv1'ac.0", 0
; CHECK-NEXT:   %{{.+}} = select {{(fast )?}}i1 %[[eq2]], double 0.000000e+00, double %[[addde]]
; CHECK-NEXT:   br i1 %[[eq2]], label %invertfor1, label %incinvertfor2

; CHECK: incinvertfor2:                                    ; preds = %invertfor2
; CHECK-NEXT:   %[[sub]] = add nsw i64 %"iv1'ac.0", -1
; CHECK-NEXT:   br label %invertfor2

; CHECK: invertend2:                                       ; preds = %entry, %incinvertfor1
; CHECK-NEXT:   %"add'de.1" = phi double [ 0.000000e+00, %entry ], [ 0.000000e+00, %incinvertfor1 ]
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ 3, %entry ], [ %[[sub1]], %incinvertfor1 ]
; CHECK-NEXT:   %[[tostoreipg:.+]] = getelementptr inbounds double, double* %"place'", i64 %"iv'ac.0"
; CHECK-NEXT:   %[[ldst:.+]] = load double, double* %[[tostoreipg]], align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %[[tostoreipg:.+]], align 8
; CHECK-NEXT:   %[[addde]] = fadd fast double %"add'de.1", %[[ldst]]
; CHECK-NEXT:   br label %invertfor2
; CHECK-NEXT: }
