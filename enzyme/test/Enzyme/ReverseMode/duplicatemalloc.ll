; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -inline -mem2reg -simplifycfg -instsimplify -adce -S | FileCheck %s

; #include <stdlib.h>
; #include <stdio.h>
;
; __attribute__((noinline))
; double f(double* x) {
;     return x[0];
; }
;
; double malloced(double x, unsigned long n) {
;     double *array = malloc(sizeof(double)*n);
;     array[0] = x;
;     double res = f(array);
;     free(array);
;     return res * res;
; }
;
; double derivative(double x, unsigned long n) {
;     return __builtin_autodiff(malloced, x, n);
; }
;
; int main(int argc, char** argv) {
;     double x = atof(argv[1]);
;     int n = atoi(argv[2]);
;     printf("original =%f derivative=%f\n", malloced(x, n), derivative(x, n));
; }

; Function Attrs: noinline norecurse nounwind readonly uwtable
define dso_local double @f(double* nocapture readonly %x) local_unnamed_addr #0 {
entry:
  %0 = load double, double* %x, align 8, !tbaa !2
  ret double %0
}

; Function Attrs: nounwind uwtable
define dso_local double @malloced(double %x, i64 %n) #1 {
entry:
  %mul = shl i64 %n, 3
  %call = tail call i8* @malloc(i64 %mul)
  %0 = bitcast i8* %call to double*
  store double %x, double* %0, align 8, !tbaa !2
  %call1 = tail call fast double @f(double* %0)
  %call2 = tail call i32 (double*, ...) bitcast (i32 (...)* @free to i32 (double*, ...)*)(double* %0) #4
  %mul3 = fmul fast double %call1, %call1
  ret double %mul3
}

; Function Attrs: nounwind
declare dso_local noalias i8* @malloc(i64) local_unnamed_addr #2

declare dso_local i32 @free(...) local_unnamed_addr #3

; Function Attrs: nounwind uwtable
define dso_local void @derivative(double %x, i64 %n) local_unnamed_addr #1 {
entry:
  %0 = tail call double (double (double, i64)*, ...) @__enzyme_autodiff(double (double, i64)* nonnull @malloced, double %x, i64 %n)
  ret void
}

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double, i64)*, ...) #4

attributes #0 = { noinline norecurse nounwind readonly uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #2 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #3 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #4 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0}
!3 = !{!"double", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}


; CHECK: define dso_local void @derivative(double %x, i64 %n)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %mul.i = shl i64 %n, 3
; CHECK-NEXT:   %call.i = call i8* @malloc(i64 %mul.i)
; CHECK-NEXT:   %[[dcall:.+]] = {{(call noalias nonnull i8\* @malloc\(i64 %mul.i\) (#[0-9]+)?[[:space:]].*call void @llvm.memset.p0i8.i64\(i8\* nonnull (align 1 )?%"call'mi.i", i8 0, i64 %mul.i, (i32 1, )?i1 false\)|call i8\* @calloc\(i64 1, i64 %mul.i\))}}
; CHECK-NEXT:   %[[ipci:.+]] = bitcast i8* %[[dcall]] to double*
; CHECK-NEXT:   %[[bccall:.+]] = bitcast i8* %call.i to double*
; CHECK-NEXT:   store double %x, double* %[[bccall]], align 8, !tbaa !2
; CHECK-NEXT:   %[[fresult:.+]] = call fast double @augmented_f(double* %[[bccall]], double* %[[ipci]])
;; TODO MAKE NON AUGMENTED:   %[[fresult:.+]] = call fast double @f(double* %[[bccall]])
; CHECK-NEXT:   %[[factor:.+]] = fadd fast double %[[fresult]], %[[fresult]]
; CHECK-NEXT:   call void @diffef(double* %[[bccall]], double* %[[ipci]], double %[[factor]])
; NOTE BETTER 03 / dead store elimination can get rid of the next line which is optional
;   since its being free'd next
; CHECK-NEXT:   store double 0.000000e+00, double* %[[ipci]], align 8
; CHECK-NEXT:   call void bitcast (i32 (...)* @free to void (i8*)*)(i8* nonnull %[[dcall]])
; CHECK-NEXT:   call void bitcast (i32 (...)* @free to void (i8*)*)(i8* %call.i)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal {{(dso_local )?}}void @diffef(double* nocapture readonly %x, double* nocapture %"x'", double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = load double, double* %"x'", align 8
; CHECK-NEXT:   %1 = fadd fast double %0, %differeturn
; CHECK-NEXT:   store double %1, double* %"x'", align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
