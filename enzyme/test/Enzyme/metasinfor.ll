; RUN: %opt < %s %loadEnzyme -enzyme -enzyme_preopt=false -mem2reg -instsimplify -adce -loop-deletion -correlated-propagation -simplifycfg -S | FileCheck %s

; Function Attrs: nounwind readnone speculatable
declare double @llvm.fabs.f64(double) #1

; Function Attrs: noinline nounwind readonly uwtable
define dso_local double @metasin(double* %a) #2 {
entry:
  %0 = load double, double* %a
  %1 = call fast double @llvm.sin.f64(double %0)
  ret double %1
}

; Function Attrs: nounwind readnone speculatable
declare double @llvm.sin.f64(double) #1

; Function Attrs: nounwind uwtable
define dso_local double @f(double %a, i32 %n) #0 {
entry:
  %a.addr = alloca double, align 8
  store double %a, double* %a.addr, align 8, !tbaa !7
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %sum.0 = phi double [ 0.000000e+00, %entry ], [ %add, %for.body ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %cmp = icmp ugt i32 %i.0, %n
  br i1 %cmp, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.cond
  ret double %sum.0

for.body:                                         ; preds = %for.cond
  %call = call fast double @metasin(double* nonnull %a.addr) #7
  %add = fadd fast double %sum.0, %call
  %inc = add i32 %i.0, 1
  br label %for.cond
}

; Function Attrs: nounwind uwtable
define dso_local i32 @caller(i32 %argc, i8** %argv) #0 {
entry:
  %z = call double (...) @__enzyme_autodiff.f64(double (double, i32)* @f, double 1.000000e+00, i32 10)
  ret i32 0
}

declare double @__enzyme_autodiff.f64(...)

attributes #0 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { noinline nounwind readonly uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #3 = { argmemonly nounwind }
attributes #4 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #5 = { noreturn nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #6 = { nounwind }
attributes #7 = { nounwind readonly }
attributes #8 = { noreturn nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0}
!3 = !{!"float", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
!6 = !{!7, !7, i64 0}
!7 = !{!"double", !4, i64 0}
!8 = !{!9, !9, i64 0}
!9 = !{!"any pointer", !4, i64 0}
!10 = !{!11, !11, i64 0}
!11 = !{!"int", !4, i64 0}

; CHECK: define internal { double } @diffef(double %a, i32 %n, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"a.addr'ipa" = alloca double, align 8
; CHECK-NEXT:   %0 = bitcast double* %"a.addr'ipa" to i8*
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull align 8 %0, i8 0, i64 8, i1 false)
; CHECK-NEXT:   %a.addr = alloca double, align 8
; CHECK-NEXT:   store double %a, double* %a.addr, align 8, !tbaa !2
; CHECK-NEXT:   br label %for.cond

; TODO THIS LOOP SHOULD BE KILLED
; CHECK: for.cond:                                         ; preds = %for.cond, %entry
; CHECK-NEXT:   %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.cond ]
; CHECK-NEXT:   %1 = trunc i64 %iv to i32
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %cmp = icmp ugt i32 %1, %n
; CHECK-NEXT:   br i1 %cmp, label %invertfor.cond, label %for.cond

; CHECK: invertentry:                                      ; preds = %invertfor.cond
; CHECK-NEXT:   %2 = load double, double* %"a.addr'ipa", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"a.addr'ipa", align 8
; CHECK-NEXT:   %3 = insertvalue { double } undef, double %2, 0
; CHECK-NEXT:   ret { double } %3

; CHECK: invertfor.cond:                                   ; preds = %for.cond, %incinvertfor.cond
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %6, %incinvertfor.cond ], [ %iv, %for.cond ]
; CHECK-NEXT:   %4 = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   %5 = select{{( fast)?}} i1 %4, double 0.000000e+00, double %differeturn
; CHECK-NEXT:   br i1 %4, label %invertentry, label %incinvertfor.cond

; CHECK: incinvertfor.cond:                                ; preds = %invertfor.cond
; CHECK-NEXT:   %6 = sub nuw nsw i64 %"iv'ac.0", 1
; CHECK-NEXT:   %7 = call {} @diffemetasin(double* nonnull %a.addr, double* nonnull %"a.addr'ipa", double %5)
; CHECK-NEXT:   br label %invertfor.cond
; CHECK-NEXT: }
