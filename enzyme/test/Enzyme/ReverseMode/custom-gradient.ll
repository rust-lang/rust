; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s


; #include <stdio.h>

; double __enzyme_autodiff(void*, double);

; __attribute__((noinline))
; void square_(const double* src, double* dest) {
;     *dest = *src * *src;
; }

; void* augment_square_(const double* src, const double *d_src, double* dest, double* d_dest) {
;     *dest = *src * *src;
;     return NULL;
; }

; void gradient_square_(const double* src, double *d_src, const double* dest, const double* d_dest, void* tape) {
;     *d_src = *d_dest * *src * 2; 
; }

; void* __enzyme_register_gradient_square[] = {
;     (void*)square_,
;     (void*)augment_square_,
;     (void*)gradient_square_,
; };


; double square(double x) {
;     double y;
;     square_(&x, &y);
;     return y;
; }

; double dsquare(double x) {
;     return __enzyme_autodiff((void*)square, x);
; }


; int main() {
;     double res = dsquare(3.0);
;     printf("res=%f\n", res);
; }

@__enzyme_register_gradient_square = dso_local local_unnamed_addr global [3 x i8*] [i8* bitcast (void (double*, double*)* @square_ to i8*), i8* bitcast (i8* (double*, double*, double*, double*)* @augment_square_ to i8*), i8* bitcast (void (double*, double*, double*, double*, i8*)* @gradient_square_ to i8*)], align 16
@.str = private unnamed_addr constant [8 x i8] c"res=%f\0A\00", align 1

define dso_local void @square_(double* nocapture readonly %src, double* nocapture %dest) #0 {
entry:
  %0 = load double, double* %src, align 8
  %mul = fmul double %0, %0
  store double %mul, double* %dest, align 8
  ret void
}

define dso_local noalias i8* @augment_square_(double* nocapture readonly %src, double* nocapture readnone %d_src, double* nocapture %dest, double* nocapture readnone %d_dest) #1 {
entry:
  %0 = load double, double* %src, align 8
  %mul = fmul double %0, %0
  store double %mul, double* %dest, align 8
  ret i8* null
}

define dso_local void @gradient_square_(double* nocapture readonly %src, double* nocapture %d_src, double* nocapture readnone %dest, double* nocapture readonly %d_dest, i8* nocapture readnone %tape) #1 {
entry:
  %0 = load double, double* %d_dest, align 8
  %1 = load double, double* %src, align 8
  %mul = fmul double %0, %1
  %mul1 = fmul double %mul, 2.000000e+00
  store double %mul1, double* %d_src, align 8
  ret void
}

define dso_local double @square(double %x) #2 {
entry:
  %x.addr = alloca double, align 8
  %y = alloca double, align 8
  store double %x, double* %x.addr, align 8
  %0 = bitcast double* %y to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %0) #6
  call void @square_(double* nonnull %x.addr, double* nonnull %y)
  %1 = load double, double* %y, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %0) #6
  ret double %1
}

declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #3

declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #3

define dso_local double @dsquare(double %x) local_unnamed_addr #2 {
entry:
  %call = tail call double @__enzyme_autodiff(i8* bitcast (double (double)* @square to i8*), double %x) #6
  ret double %call
}

declare dso_local double @__enzyme_autodiff(i8*, double) local_unnamed_addr #4

define dso_local i32 @main() local_unnamed_addr #2 {
entry:
  %call.i = tail call double @__enzyme_autodiff(i8* bitcast (double (double)* @square to i8*), double 3.000000e+00) #6
  %call1 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str, i64 0, i64 0), double %call.i)
  ret i32 0
}

declare dso_local i32 @printf(i8* nocapture readonly, ...) local_unnamed_addr #5

attributes #0 = { noinline norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { argmemonly nounwind }
attributes #4 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { nounwind }


; CHECK: define internal { double } @diffesquare(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"x.addr'ipa" = alloca double, align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"x.addr'ipa", align 8
; CHECK-NEXT:   %x.addr = alloca double, align 8
; CHECK-NEXT:   %"y'ipa" = alloca double, align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"y'ipa", align 8
; CHECK-NEXT:   %y = alloca double, align 8
; CHECK-NEXT:   store double %x, double* %x.addr, align 8
; CHECK-NEXT:   %0 = load double, double* %"y'ipa", align 8
; CHECK-NEXT:   %1 = fadd fast double %0, %differeturn
; CHECK-NEXT:   store double %1, double* %"y'ipa", align 8
; CHECK-NEXT:   call void @fixgradient_square_(double* %x.addr, double* %"x.addr'ipa", double* %y, double* %"y'ipa")
; CHECK-NEXT:   %2 = load double, double* %"x.addr'ipa", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"x.addr'ipa", align 8
; CHECK-NEXT:   %3 = insertvalue { double } undef, double %2, 0
; CHECK-NEXT:   ret { double } %3
; CHECK-NEXT: }

; CHECK: define internal void @fixgradient_square_(double*{{( %0)?}}, double*{{( %1)?}}, double*{{( %2)?}}, double*{{( %3)?}})
; CHECK-NEXT: entry:
; CHECK-NEXT:   %4 = call i8* @augment_square_(double* %0, double* %1, double* %2, double* %3)
; CHECK-NEXT:   call void @gradient_square_(double* %0, double* %1, double* %2, double* %3, i8* %4)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }