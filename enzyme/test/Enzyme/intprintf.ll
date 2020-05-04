; RUN: %opt < %s %loadEnzyme -enzyme -enzyme_preopt=false -mem2reg -sroa -simplifycfg -instcombine -gvn -adce -S | FileCheck %s

@.str = private unnamed_addr constant [46 x i8] c"final result t=%f x(t)=%f, -0.2=%f, steps=%d\0A\00", align 1

; Function Attrs: nounwind uwtable
define dso_local double @foobar(double %t) #3 {
entry:
  br label %while

while:                              ; preds = %entry, %while.body.us.i.i.i
  %0 = phi double [ %mul2, %while ], [ 1.000000e+00, %entry ]
  %i = phi i32 [ %inc, %while ], [ 0, %entry ]
  %inc = add nuw nsw i32 %i, 1
  %mul2 = fmul fast double %t, %0
  %cmp2 = fcmp fast ugt double %mul2, 2.000000e+00
  br i1 %cmp2, label %exit, label %while

exit:
  %phi = phi i32 [ %inc, %while ]
  %a3 = zext i32 %phi to i64
  %call2 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([46 x i8], [46 x i8]* @.str, i64 0, i64 0), double %t, double 3.141592e+00, double -2.000000e-01, i64 %a3)
  ret double %mul2
}

; Function Attrs: nounwind
declare dso_local i32 @printf(i8* nocapture readonly, ...)

; Function Attrs: norecurse nounwind uwtable
define double @caller(double %inp) {
entry:
  %call = call fast double @__enzyme_autodiff(i8* bitcast (double (double)* @foobar to i8*), double %inp)
  ret double %call
}

declare double @__enzyme_autodiff(i8*, double)

; CHECK: define internal { double } @diffefoobar(double %t, double %differeturn) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   br label %while

; CHECK: while:
; CHECK-NEXT:   %0 = phi i8* [ null, %entry ], [ %_realloccache, %while ]
; CHECK-NEXT:   %iv = phi i64 [ 0, %entry ], [ %iv.next, %while ]
; CHECK-NEXT:   %1 = phi double [ 1.000000e+00, %entry ], [ %mul2, %while ]
; CHECK-NEXT:   %[[ivtrunc:.+]] = trunc i64 %iv to i32
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %[[bytesalloc:.+]] = shl nuw nsw i64 %iv.next, 3
; CHECK-NEXT:   %_realloccache = call i8* @realloc(i8* %0, i64 %[[bytesalloc]])
; CHECK-NEXT:   %_realloccast = bitcast i8* %_realloccache to double*
; CHECK-NEXT:   %[[storeloc:.+]] = getelementptr inbounds double, double* %_realloccast, i64 %iv
; CHECK-NEXT:   store double %1, double* %[[storeloc]], align 8, !invariant.group !0
; CHECK-NEXT:   %inc = add nuw nsw i32 %[[ivtrunc]], 1
; CHECK-NEXT:   %mul2 = fmul fast double %1, %t
; CHECK-NEXT:   %cmp2 = fcmp fast ugt double %mul2, 2.000000e+00
; CHECK-NEXT:   br i1 %cmp2, label %exit, label %while

; CHECK: exit:
; CHECK-NEXT:   %a3 = zext i32 %inc to i64
; CHECK-NEXT:   %call2 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([46 x i8], [46 x i8]* @.str, i64 0, i64 0), double %t, double 0x400921FAFC8B007A, double -2.000000e-01, i64 %a3)
; CHECK-NEXT:   %5 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([46 x i8], [46 x i8]* @.str, i64 0, i64 0), double %t, double 0x400921FAFC8B007A, double -2.000000e-01, i64 %a3)
; CHECK-NEXT:   br label %invertwhile

; CHECK: invertentry:
; CHECK-NEXT:   %6 = insertvalue { double } undef, double %9, 0
; CHECK-NEXT:   tail call void @free(i8* nonnull %_realloccache)
; CHECK-NEXT:   ret { double } %6

; CHECK: invertwhile:
; CHECK-NEXT:   %"t'de.0" = phi double [ 0.000000e+00, %exit ], [ %9, %incinvertwhile ]
; CHECK-NEXT:   %"mul2'de.0" = phi double [ %differeturn, %exit ], [ %[[m0diffe:.+]], %incinvertwhile ]
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %iv, %exit ], [ %11, %incinvertwhile ]
; CHECK-NEXT:   %7 = getelementptr inbounds double, double* %_realloccast, i64 %"iv'ac.0"
; CHECK-NEXT:   %8 = load double, double* %7, align 8, !invariant.group !0
; CHECK-NEXT:   %[[m1diffet:.+]] = fmul fast double %"mul2'de.0", %8
; CHECK-NEXT:   %9 = fadd fast double %"t'de.0", %[[m1diffet]]
; CHECK-NEXT:   %10 = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %10, label %invertentry, label %incinvertwhile

; CHECK: incinvertwhile:
; CHECK-NEXT:   %[[m0diffe]] = fmul fast double %"mul2'de.0", %t
; CHECK-NEXT:   %11 = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertwhile
; CHECK-NEXT: }
