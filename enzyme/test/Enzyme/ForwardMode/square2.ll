; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

; #include <stdio.h>

; double __enzyme_fwddiff(void*, ...);

; __attribute__((noinline))
; void square_(const double* src, double* dest) {
;     *dest = *src * *src;
; }

; double square(double x) {
;     double y;
;     square_(&x, &y);
;     return y;
; }

; double dsquare(double x) {
;     return __enzyme_fwddiff((void*)square, x, 1.0);
; }


define dso_local void @square_(double* nocapture readonly %src, double* nocapture %dest) local_unnamed_addr #0 {
entry:
  %0 = load double, double* %src, align 8
  %mul = fmul double %0, %0
  store double %mul, double* %dest, align 8
  ret void
}

define dso_local double @square(double %x) #1 {
entry:
  %x.addr = alloca double, align 8
  %y = alloca double, align 8
  store double %x, double* %x.addr, align 8
  %0 = bitcast double* %y to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %0) #4
  call void @square_(double* nonnull %x.addr, double* nonnull %y)
  %1 = load double, double* %y, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %0) #4
  ret double %1
}

declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #2

declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #2

define dso_local double @dsquare(double %x) local_unnamed_addr #1 {
entry:
  %call = tail call double (i8*, ...) @__enzyme_fwddiff(i8* bitcast (double (double)* @square to i8*), double %x, double 1.000000e+00) #4
  ret double %call
}

declare dso_local double @__enzyme_fwddiff(i8*, ...) local_unnamed_addr #3

attributes #0 = { norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { argmemonly nounwind }
attributes #3 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind }


; CHECK: define internal double @fwddiffesquare(double %x, double %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"x.addr'ipa" = alloca double, align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"x.addr'ipa", align 8
; CHECK-NEXT:   %x.addr = alloca double, align 8
; CHECK-NEXT:   %"y'ipa" = alloca double, align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"y'ipa", align 8
; CHECK-NEXT:   %y = alloca double, align 8
; CHECK-NEXT:   store double %x, double* %x.addr, align 8
; CHECK-NEXT:   store double %"x'", double* %"x.addr'ipa", align 8
; CHECK-NEXT:   call void @fwddiffesquare_(double* %x.addr, double* %"x.addr'ipa", double* %y, double* %"y'ipa")
; CHECK-NEXT:   %[[i0:.+]] = load double, double* %"y'ipa", align 8
; CHECK-NEXT:   ret double %[[i0:.+]]
; CHECK-NEXT: }

; CHECK: define internal void @fwddiffesquare_(double* nocapture readonly %src, double* nocapture %"src'", double* nocapture %dest, double* nocapture %"dest'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[i1:.+]] = load double, double* %"src'", align 8
; CHECK-NEXT:   %0 = load double, double* %src, align 8
; CHECK-NEXT:   %mul = fmul double %0, %0
; CHECK-NEXT:   %[[i2:.+]] = fmul fast double %[[i1]], %0
; CHECK-NEXT:   %[[i3:.+]] = fmul fast double %[[i1]], %0
; CHECK-NEXT:   %[[i4:.+]] = fadd fast double %[[i2]], %[[i3]]
; CHECK-NEXT:   store double %mul, double* %dest, align 8
; CHECK-NEXT:   store double %[[i4]], double* %"dest'", align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
