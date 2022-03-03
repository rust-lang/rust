; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -instsimplify -simplifycfg -adce -S | FileCheck %s

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


%struct.Gradients = type { double, double, double }

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

define dso_local %struct.Gradients @dsquare(double %x) local_unnamed_addr #1 {
entry:
  %call = tail call %struct.Gradients (i8*, ...) @__enzyme_fwddiff(i8* bitcast (double (double)* @square to i8*), metadata !"enzyme_width", i64 3, double %x, double 1.0, double 2.0, double 3.0) #4
  ret %struct.Gradients %call
}

declare dso_local %struct.Gradients @__enzyme_fwddiff(i8*, ...) local_unnamed_addr #3

attributes #0 = { norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { argmemonly nounwind }
attributes #3 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind }


; CHECK: define internal void @fwddiffe3square_(double* nocapture readonly %src, [3 x double*] %"src'", double* nocapture %dest, [3 x double*] %"dest'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = load double, double* %src, align 8
; CHECK-NEXT:   %1 = extractvalue [3 x double*] %"src'", 0
; CHECK-NEXT:   %2 = load double, double* %1, align 8
; CHECK-NEXT:   %3 = extractvalue [3 x double*] %"src'", 1
; CHECK-NEXT:   %4 = load double, double* %3, align 8
; CHECK-NEXT:   %5 = extractvalue [3 x double*] %"src'", 2
; CHECK-NEXT:   %6 = load double, double* %5, align 8
; CHECK-NEXT:   %mul = fmul double %0, %0
; CHECK-NEXT:   %7 = fmul fast double %2, %0
; CHECK-NEXT:   %8 = fmul fast double %2, %0
; CHECK-NEXT:   %9 = fadd fast double %7, %8
; CHECK-NEXT:   %10 = fmul fast double %4, %0
; CHECK-NEXT:   %11 = fmul fast double %4, %0
; CHECK-NEXT:   %12 = fadd fast double %10, %11
; CHECK-NEXT:   %13 = fmul fast double %6, %0
; CHECK-NEXT:   %14 = fmul fast double %6, %0
; CHECK-NEXT:   %15 = fadd fast double %13, %14
; CHECK-NEXT:   store double %mul, double* %dest, align 8
; CHECK-NEXT:   %16 = extractvalue [3 x double*] %"dest'", 0
; CHECK-NEXT:   store double %9, double* %16, align 8
; CHECK-NEXT:   %17 = extractvalue [3 x double*] %"dest'", 1
; CHECK-NEXT:   store double %12, double* %17, align 8
; CHECK-NEXT:   %18 = extractvalue [3 x double*] %"dest'", 2
; CHECK-NEXT:   store double %15, double* %18, align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }