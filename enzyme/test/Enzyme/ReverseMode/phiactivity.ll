; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -simplifycfg -instcombine -early-cse -adce -S | FileCheck %s

@.str = private unnamed_addr constant [46 x i8] c"final result t=%f x(t)=%f, -0.2=%f, steps=%d\0A\00", align 1

define dso_local double @foobar(double %t) {
entry:
  br label %while

while:                              ; preds = %entry, %while.body.us.i.i.i
  %0 = phi double [ %mul2, %while ], [ 1.000000e+00, %entry ]
  %i = phi i32 [ %inc, %while ], [ 0, %entry ]
  %inc = add nuw nsw i32 %i, 1
  %mul2 = fmul fast double %t, %0
  %conv = sitofp i32 %inc to double
  %mul = fmul fast double %conv, %t
  %cmp2 = fcmp fast ugt double %mul, 2.000000e+00
  br i1 %cmp2, label %exit, label %while

exit:
  %phi = phi i32 [ %inc, %while ]
  %a3 = zext i32 %phi to i64
  %call2 = tail call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([46 x i8], [46 x i8]* @.str, i64 0, i64 0), double %t, double 3.141592e+00, double -2.000000e-01, i64 %a3)
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

; CHECK: define internal { double } @diffefoobar(double %t, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   br label %while

; CHECK: while:
; CHECK-NEXT:   %[[prevalloc:.+]] = phi i8* [ null, %entry ], [ %[[_realloccache:.+]], %[[mergeblk:.+]] ]
; CHECK-NEXT:   %iv = phi i64 [ 0, %entry ], [ %iv.next, %[[mergeblk]] ]
; CHECK-NEXT:   %[[mphi:.+]] = phi double [ 1.000000e+00, %entry ], [ %mul2, %[[mergeblk]] ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1

; CHECK-NEXT:   %[[nexttrunc0:.+]] = and i64 %iv.next, 1
; CHECK-NEXT:   %[[nexttrunc:.+]] = icmp ne i64 %[[nexttrunc0]], 0
; CHECK-NEXT:   %[[popcnt:.+]] = call i64 @llvm.ctpop.i64(i64 %iv.next)
; CHECK-NEXT:   %[[le2:.+]] = icmp ult i64 %[[popcnt:.+]], 3
; CHECK-NEXT:   %[[shouldgrow:.+]] = and i1 %[[le2]], %[[nexttrunc]]
; CHECK-NEXT:   br i1 %[[shouldgrow]], label %grow.i, label %[[mergeblk]]

; CHECK: grow.i:
; CHECK-NEXT:   %[[ctlz:.+]] = call i64 @llvm.ctlz.i64(i64 %iv.next, i1 true)
; CHECK-NEXT:   %[[maxbit:.+]] = sub nuw nsw i64 64, %[[ctlz]]
; CHECK-NEXT:   %[[numbytes:.+]] = shl i64 8, %[[maxbit]]
; CHECK-NEXT:   %[[growalloc:.+]] = call i8* @realloc(i8* %[[prevalloc]], i64 %[[numbytes]])
; CHECK-NEXT:   br label %[[mergeblk]]

; CHECK: [[mergeblk]]:
; CHECK-NEXT:   %[[_realloccache]] = phi i8* [ %[[growalloc]], %grow.i ], [ %[[prevalloc]], %while ]

; CHECK-NEXT:   %[[_realloccast:.+]] = bitcast i8* %[[_realloccache]] to double*
; CHECK-NEXT:   %[[loc1:.+]] = getelementptr inbounds double, double* %[[_realloccast]], i64 %iv
; CHECK-NEXT:   store double %[[mphi]], double* %[[loc1]], align 8, !invariant.group ![[grp:[0-9]+]]
; CHECK-NEXT:   %[[trunc:.+]] = trunc i64 %iv to i32
; CHECK-NEXT:   %inc = add nuw nsw i32 %[[trunc]], 1
; CHECK-NEXT:   %mul2 = fmul fast double %[[mphi]], %t
; CHECK-NEXT:   %conv = sitofp i32 %inc to double
; CHECK-NEXT:   %mul = fmul fast double %conv, %t
; CHECK-NEXT:   %cmp2 = fcmp fast ugt double %mul, 2.000000e+00
; CHECK-NEXT:   br i1 %cmp2, label %exit, label %while

; CHECK: exit:
; CHECK-NEXT:   %a3 = zext i32 %inc to i64
; CHECK-NEXT:   %call2 = tail call i32 (i8*, ...) @printf(i8* {{(noundef )?}}nonnull dereferenceable(1) getelementptr inbounds ([46 x i8], [46 x i8]* @.str, i64 0, i64 0), double %t, double 0x400921FAFC8B007A, double -2.000000e-01, i64 %a3)
; CHECK-NEXT:   br label %invertwhile

; CHECK: invertentry:                                      ; preds = %invertwhile
; CHECK-NEXT:   %[[res:.+]] = insertvalue { double } undef, double %[[dt2:.+]], 0
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[_realloccache]])
; CHECK-NEXT:   ret { double } %[[res]]

; CHECK: invertwhile:                                      ; preds = %exit, %incinvertwhile
; CHECK-NEXT:   %"t'de.0" = phi double [ 0.000000e+00, %exit ], [ %[[dt2]], %incinvertwhile ]
; CHECK-NEXT:   %"mul2'de.0" = phi double [ %differeturn, %exit ], [ %[[m0diffe:.+]], %incinvertwhile ]
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %iv, %exit ], [ %[[dinc:.+]], %incinvertwhile ]
; CHECK-NEXT:   %[[philoc:.+]] = getelementptr inbounds double, double* %[[_realloccast]], i64 %"iv'ac.0"
; CHECK-NEXT:   %[[prevphi:.+]] = load double, double* %[[philoc]], align 8, !invariant.group ![[grp]]
; CHECK-NEXT:   %[[m1diffet:.+]] = fmul fast double %"mul2'de.0", %[[prevphi]]
; CHECK-NEXT:   %[[dt2]] = fadd fast double %"t'de.0", %[[m1diffet]]
; CHECK-NEXT:   %[[dcmp:.+]] = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %[[dcmp]], label %invertentry, label %incinvertwhile

; CHECK: incinvertwhile:                                   ; preds = %invertwhile
; CHECK-NEXT:   %[[m0diffe]] = fmul fast double %"mul2'de.0", %t
; CHECK-NEXT:   %[[dinc]] = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertwhile
; CHECK-NEXT: }
