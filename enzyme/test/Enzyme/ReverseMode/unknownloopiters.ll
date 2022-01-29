; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -inline -mem2reg -instsimplify -adce -loop-deletion -correlated-propagation -simplifycfg -S | FileCheck %s

; Function Attrs: nounwind uwtable
define dso_local double @unknowniters(double* nocapture readonly %x) #0 {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.cond, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.cond ], [ 0, %entry ]
  %total.0 = phi double [ %add, %for.cond ], [ 0.000000e+00, %entry ]
  %arrayidx = getelementptr inbounds double, double* %x, i64 %indvars.iv
  %0 = load double, double* %arrayidx, align 8, !tbaa !2
  %add = fadd fast double %0, %total.0
  %call = tail call i32 (...) @done() #2
  %tobool = icmp eq i32 %call, 0
  %indvars.iv.next = add nuw i64 %indvars.iv, 1
  br i1 %tobool, label %for.cond, label %if.then

if.then:                                          ; preds = %for.cond
  ret double %add
}

declare dso_local i32 @done(...) local_unnamed_addr #1

; Function Attrs: nounwind uwtable
define dso_local void @ddynsum(double* %x, double* %xp) local_unnamed_addr #0 {
entry:
  %0 = tail call double (double (double*)*, ...) @__enzyme_autodiff(double (double*)* nonnull @unknowniters, double* %x, double* %xp)
  ret void
}

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double*)*, ...) #2

attributes #0 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { readonly "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #2 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0}
!3 = !{!"double", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}


; CHECK: define dso_local void @ddynsum(double* %x, double* %xp)

; CHECK-NEXT: entry:
; CHECK-NEXT:   br label %for.cond.i

; CHECK: for.cond.i:                                       ; preds = %for.cond.i, %entry
; CHECK-NEXT:   %[[iv:.+]] = phi i64 [ %[[ivnext:.+]], %for.cond.i ], [ 0, %entry ]
; CHECK-NEXT:   %[[ivnext]] = add nuw nsw i64 %[[iv]], 1
; CHECK-NEXT:   %call.i = call i32 (...) @done()
; CHECK-NEXT:   %tobool.i = icmp eq i32 %call.i, 0
; CHECK-NEXT:   br i1 %tobool.i, label %for.cond.i, label %[[antiloop:.+]]

; CHECK: [[antiloop]]:
; CHECK-NEXT:   %[[antiiv:.+]] = phi i64 [ %[[antiivnext:.+]], %[[incantiloop:.+]] ], [ %[[iv]], %for.cond.i ]
; CHECK-NEXT:   %[[arrayidxipgi:.+]] = getelementptr inbounds double, double* %xp, i64 %[[antiiv]]
; CHECK-NEXT:   %[[load:.+]] = load double, double* %[[arrayidxipgi]]
; CHECK-NEXT:   %[[fadd:.+]] = fadd fast double %[[load]], 1.000000e+00
; CHECK-NEXT:   store double %[[fadd]], double* %[[arrayidxipgi]]
; CHECK-NEXT:   %[[cmp:.+]] = icmp eq i64 %[[antiiv]], 0
; CHECK-NEXT:   br i1 %[[cmp]], label %diffeunknowniters.exit, label %[[incantiloop]]

; CHECK: [[incantiloop]]:
; TODO the following can have nuw on it because its known non 0
; CHECK-NEXT:   %[[antiivnext]] = add nsw i64 %[[antiiv]], -1
; CHECK-NEXT:   br label %[[antiloop]]

; CHECK: diffeunknowniters.exit:
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
