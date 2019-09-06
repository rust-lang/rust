; RUN: opt < %s %loadEnzyme -enzyme -inline -mem2reg -instsimplify -adce -loop-deletion -correlated-propagation -simplifycfg -ipconstprop -deadargelim -S -early-cse -instcombine | FileCheck %s

; #include <stdio.h>
; #include <stdlib.h>
; 
; double recsum(double* x, unsigned n) {
;     if (n == 0) return 0;
;     if (n == 1) return x[0];
;     return recsum(x, n/2) + recsum(x + n/2, n - n/2);
; }
; 
; void dsum(double* x, double* xp, unsigned n) {
;     __builtin_autodiff(recsum, x, xp, n);
; }

; Function Attrs: nounwind readonly uwtable
define dso_local double @recsum(double* %x, i32 %n) #0 {
entry:
  switch i32 %n, label %if.end3 [
    i32 0, label %return
    i32 1, label %if.then2
  ]

if.then2:                                         ; preds = %entry
  %0 = load double, double* %x, align 8, !tbaa !2
  br label %return

if.end3:                                          ; preds = %entry
  %div = lshr i32 %n, 1
  %call = tail call fast double @recsum(double* %x, i32 %div)
  %idx.ext = zext i32 %div to i64
  %add.ptr = getelementptr inbounds double, double* %x, i64 %idx.ext
  %sub = sub i32 %n, %div
  %call6 = tail call fast double @recsum(double* %add.ptr, i32 %sub)
  %add = fadd fast double %call6, %call
  ret double %add

return:                                           ; preds = %entry, %if.then2
  %retval.0 = phi double [ %0, %if.then2 ], [ 0.000000e+00, %entry ]
  ret double %retval.0
}

; Function Attrs: nounwind uwtable
define dso_local void @dsum(double* %x, double* %xp, i32 %n) local_unnamed_addr #1 {
entry:
  %0 = tail call double (double (double*, i32)*, ...) @__enzyme_autodiff(double (double*, i32)* nonnull @recsum, double* %x, double* %xp, i32 %n)
  ret void
}

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double*, i32)*, ...) #2

attributes #0 = { nounwind readonly uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #2 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0}
!3 = !{!"double", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}

; CHECK: define internal double @differecsum.1(double* %x, double* %"x'", i32 %n)
; CHECK-NEXT: entry:
; CHECK-NEXT:   switch i32 %n, label %if.end3 [
; CHECK-NEXT:     i32 0, label %invertreturn
; CHECK-NEXT:     i32 1, label %if.then2
; CHECK-NEXT:   ]

; CHECK: if.then2:                                         ; preds = %entry
; CHECK-NEXT:   %0 = load double, double* %x, align 8, !tbaa !2
; CHECK-NEXT:   br label %invertreturn

; CHECK: if.end3:                                          ; preds = %entry
; CHECK-NEXT:   %div = lshr i32 %n, 1
; CHECK-NEXT:   %idx.ext = zext i32 %div to i64
; CHECK-NEXT:   %add.ptr = getelementptr inbounds double, double* %x, i64 %idx.ext
; CHECK-NEXT:   %"add.ptr'ipg" = getelementptr double, double* %"x'", i64 %idx.ext
; CHECK-NEXT:   %1 = sub i32 %n, %div
; CHECK-NEXT:   %2 = call double @differecsum.1(double* %add.ptr, double* %"add.ptr'ipg", i32 %1)
; CHECK-NEXT:   %3 = call double @differecsum.1(double* %x, double* %"x'", i32 %div)
; CHECK-NEXT:   %add = fadd fast double %2, %3
; CHECK-NEXT:   br label %invertentry

; CHECK: invertentry:                                      ; preds = %invertreturn, %if.end3, %invertif.then2
; CHECK-NEXT:   %toreturn.0 = phi double [ %add, %if.end3 ], [ %retval.0, %invertif.then2 ], [ %retval.0, %invertreturn ]
; CHECK-NEXT:   ret double %toreturn.0

; CHECK: invertif.then2:                                   ; preds = %invertreturn
; CHECK-NEXT:   %4 = load double, double* %"x'", align 8
; CHECK-NEXT:   %5 = fadd fast double %4, 1.000000e+00
; CHECK-NEXT:   store double %5, double* %"x'", align 8
; CHECK-NEXT:   br label %invertentry

; CHECK: invertreturn:                                     ; preds = %entry, %if.then2
; CHECK-NEXT:   %6 = phi i1 [ true, %if.then2 ], [ false, %entry ]
; CHECK-NEXT:   %retval.0 = phi double [ %0, %if.then2 ], [ 0.000000e+00, %entry ]
; CHECK-NEXT:   br i1 %6, label %invertif.then2, label %invertentry
; CHECK-NEXT: }
