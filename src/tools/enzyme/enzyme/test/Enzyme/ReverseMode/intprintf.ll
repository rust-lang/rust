; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -simplifycfg -instcombine -gvn -adce -S | FileCheck %s

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

; CHECK: define internal { double } @diffefoobar(double %t, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   br label %while

; CHECK: while:
; CHECK-NEXT:   %0 = phi i8* [ null, %entry ], [ %[[phiptr:.+]], %[[mergeblk:.+]] ]
; CHECK-NEXT:   %iv = phi i64 [ 0, %entry ], [ %iv.next, %[[mergeblk]] ]
; CHECK-NEXT:   %1 = phi double [ 1.000000e+00, %entry ], [ %mul2, %[[mergeblk]] ]
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
; CHECK-NEXT:   %[[growalloc:.+]] = call i8* @realloc(i8* %0, i64 %[[numbytes]])
; CHECK-NEXT:   br label %[[mergeblk]]

; CHECK: [[mergeblk]]:
; CHECK-NEXT:   %[[phiptr]] = phi i8* [ %[[growalloc]], %grow.i ], [ %0, %while ]
; CHECK-NEXT:   %[[_realloccast:.+]] = bitcast i8* %[[phiptr]] to double*

; CHECK-NEXT:   %[[storeloc:.+]] = getelementptr inbounds double, double* %[[_realloccast]], i64 %iv
; CHECK-NEXT:   store double %1, double* %[[storeloc]], align 8, !invariant.group ![[grp:[0-9]+]]
; CHECK-NEXT:   %mul2 = fmul fast double %1, %t
; CHECK-NEXT:   %cmp2 = fcmp fast ugt double %mul2, 2.000000e+00
; CHECK-NEXT:   br i1 %cmp2, label %exit, label %while

; CHECK: exit:
; CHECK-NEXT:   %[[ivtrunc:.+]] = trunc i64 %iv to i32
; CHECK-NEXT:   %inc = add nuw nsw i32 %[[ivtrunc]], 1
; CHECK-NEXT:   %a3 = zext i32 %inc to i64
; CHECK-NEXT:   %call2 = tail call i32 (i8*, ...) @printf
; CHECK-NEXT:   br label %invertwhile

; CHECK: invertentry:
; CHECK-NEXT:   %[[res:.+]] = insertvalue { double } undef, double %[[d9:.+]], 0
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[phiptr]])
; CHECK-NEXT:   ret { double } %[[res]]

; CHECK: invertwhile:
; CHECK-NEXT:   %"t'de.0" = phi double [ 0.000000e+00, %exit ], [ %[[d9]], %incinvertwhile ]
; CHECK-NEXT:   %"mul2'de.0" = phi double [ %differeturn, %exit ], [ %[[m0diffe:.+]], %incinvertwhile ]
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %iv, %exit ], [ %[[d11:.+]], %incinvertwhile ]
; CHECK-NEXT:   %[[gepalloc:.+]] = getelementptr inbounds double, double* %[[_realloccast]], i64 %"iv'ac.0"
; CHECK-NEXT:   %[[d8:.+]] = load double, double* %[[gepalloc]], align 8, !invariant.group ![[grp]]
; CHECK-NEXT:   %[[m1diffet:.+]] = fmul fast double %"mul2'de.0", %[[d8]]
; CHECK-NEXT:   %[[d9]] = fadd fast double %"t'de.0", %[[m1diffet]]
; CHECK-NEXT:   %[[icmp:.+]] = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %[[icmp]], label %invertentry, label %incinvertwhile

; CHECK: incinvertwhile:
; CHECK-NEXT:   %[[m0diffe]] = fmul fast double %"mul2'de.0", %t
; CHECK-NEXT:   %[[d11]] = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertwhile
; CHECK-NEXT: }
