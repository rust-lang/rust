; RUN: opt < %s %loadEnzyme -enzyme -mem2reg -instsimplify -simplifycfg -S -gvn -dse -dse | FileCheck %s

; __attribute__((noinline))
; void function(double y, double z, double *x) {
;     double m = y * z;
;     *x = m;
; }
; 
; __attribute__((noinline))
; void addOne(double *x) {
;     *x += 1;
; }
; 
; __attribute__((noinline))
; void function0(double y, double z, double *x) {
;     function(y, z, x);
;     addOne(x);
; }
; 
; double test_derivative(double *x, double *xp, double y, double z) {
;   return __builtin_autodiff(function0, y, z, x, xp);
; }

; Function Attrs: noinline norecurse nounwind uwtable
define dso_local void @function(double %y, double %z, double* nocapture %x) local_unnamed_addr #0 {
entry:
  %mul = fmul fast double %z, %y
  store double %mul, double* %x, align 8, !tbaa !2
  ret void
}

; Function Attrs: noinline norecurse nounwind uwtable
define dso_local void @addOne(double* nocapture %x) local_unnamed_addr #0 {
entry:
  %0 = load double, double* %x, align 8, !tbaa !2
  %add = fadd fast double %0, 1.000000e+00
  store double %add, double* %x, align 8, !tbaa !2
  ret void
}

; Function Attrs: noinline norecurse nounwind uwtable
define dso_local void @function0(double %y, double %z, double* nocapture %x) #0 {
entry:
  tail call void @function(double %y, double %z, double* %x)
  tail call void @addOne(double* %x)
  ret void
}

; Function Attrs: nounwind uwtable
define dso_local double @test_derivative(double* %x, double* %xp, double %y, double %z) local_unnamed_addr #1 {
entry:
  %0 = tail call double (void (double, double, double*)*, ...) @__enzyme_autodiff(void (double, double, double*)* nonnull @function0, double %y, double %z, double* %x, double* %xp)
  ret double %0
}

; Function Attrs: nounwind
declare double @__enzyme_autodiff(void (double, double, double*)*, ...) #2

attributes #0 = { noinline norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
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


; CHECK: define dso_local double @test_derivative(double* %x, double* %xp, double %y, double %z) 
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call { double, double } @diffefunction0(double %y, double %z, double* %x, double* %xp)
; CHECK-NEXT:   %1 = extractvalue { double, double } %0, 0
; CHECK-NEXT:   ret double %1
; CHECK-NEXT: }

; CHECK: define internal {{(dso_local )?}}{ double, double } @diffefunction0(double %y, double %z, double* nocapture %x, double* %"x'") 
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call { {} } @augmented_function(double %y, double %z, double* %x, double* %"x'")
; CHECK-NEXT:   %1 = call {} @diffeaddOne(double* %x, double* %"x'")
; CHECK-NEXT:   %[[result:.+]] = call { double, double } @diffefunction(double %y, double %z, double* %x, double* %"x'", {} undef)
; CHECK-NEXT:   ret { double, double } %[[result]]
; CHECK-NEXT: }

; CHECK: define internal {{(dso_local )?}}{} @diffeaddOne(double* nocapture %x, double* %"x'") 
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = load double, double* %x, align 8, !tbaa !2
; CHECK-NEXT:   %add = fadd fast double %0, 1.000000e+00
; CHECK-NEXT:   store double %add, double* %x, align 8, !tbaa !2
; CHECK-NEXT:   ret {} undef
; CHECK-NEXT: }

; CHECK: define internal {{(dso_local )?}}{ {} } @augmented_function(double %y, double %z, double* nocapture %x, double* %"x'") 
; CHECK-NEXT: entry:
; CHECK-NEXT:   %mul = fmul fast double %z, %y
; CHECK-NEXT:   store double %mul, double* %x, align 8, !tbaa !2
; CHECK-NEXT:   ret { {} } undef
; CHECK-NEXT: }

; CHECK: define internal {{(dso_local )?}}{ double, double } @diffefunction(double %y, double %z, double* nocapture %x, double* %"x'", {} %tapeArg) 
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = load double, double* %"x'"
; CHECK-NEXT:   store double 0.000000e+00, double* %"x'"
; CHECK-NEXT:   %[[m0diffez:.+]] = fmul fast double %0, %y
; CHECK-NEXT:   %[[m1diffey:.+]] = fmul fast double %0, %z
; CHECK-NEXT:   %1 = insertvalue { double, double } undef, double %[[m1diffey]], 0
; CHECK-NEXT:   %2 = insertvalue { double, double } %1, double %[[m0diffez]], 1
; CHECK-NEXT:   ret { double, double } %2
; CHECK-NEXT: }
