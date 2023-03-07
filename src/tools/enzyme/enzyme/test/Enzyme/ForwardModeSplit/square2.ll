; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -instsimplify -simplifycfg -adce -S | FileCheck %s

; #include <stdio.h>

; double __enzyme_fwdsplit(void*, ...);

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
;     return __enzyme_fwdsplit((void*)square, x, 1.0);
; }


define dso_local void @square_(double* nocapture readonly %src, double* nocapture noalias writeonly %dest) local_unnamed_addr #0 {
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
  %call = tail call double (i8*, ...) @__enzyme_fwdsplit(i8* bitcast (double (double)* @square to i8*), metadata !"enzyme_nofree", double %x, double 1.000000e+00, i8* null) #4
  ret double %call
}

declare dso_local double @__enzyme_fwdsplit(i8*, ...) local_unnamed_addr #3

attributes #0 = { norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { argmemonly nounwind }
attributes #3 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind }


; CHECK: define internal double @fwddiffesquare(double %x, double %"x'", i8* %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %x.addr = alloca double, i64 1, align 8
; CHECK-NEXT:   %"x.addr'ai" = alloca double, i64 1, align 8
; CHECK-NEXT:   %[[q1:.+]] = bitcast double* %"x.addr'ai" to i8*
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull dereferenceable(8) dereferenceable_or_null(8) %[[q1]], i8 0, i64 8, i1 false)
; CHECK-NEXT:   %"y'ai" = alloca double, i64 1, align 8
; CHECK-NEXT:   %[[q2:.]] = bitcast double* %"y'ai" to i8*
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull dereferenceable(8) dereferenceable_or_null(8) %[[q2]], i8 0, i64 8, i1 false)
; CHECK-NEXT:   store double %x, double* %x.addr, align 8
; CHECK-NEXT:   store double %"x'", double* %"x.addr'ai", align 8
; CHECK-NEXT:   call void @fwddiffesquare_(double* %x.addr, double* %"x.addr'ai", double* undef, double* %"y'ai")
; CHECK-NEXT:   %[[i1:.+]] = load double, double* %"y'ai", align 8
; CHECK-NEXT:   ret double %[[i1]]
; CHECK-NEXT: }

; CHECK: define internal void @fwddiffesquare_(double* nocapture readonly %src, double* nocapture %"src'", double* noalias nocapture writeonly %dest, double* nocapture %"dest'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[i1:.+]] = load double, double* %"src'", align 8
; CHECK-NEXT:   %0 = load double, double* %src
; CHECK-NEXT:   %[[i2:.+]] = fmul fast double %[[i1]], %0
; CHECK-NEXT:   %[[i3:.+]] = fmul fast double %[[i1]], %0
; CHECK-NEXT:   %[[i4:.+]] = fadd fast double %[[i2]], %[[i3]]
; CHECK-NEXT:   store double %[[i4]], double* %"dest'", align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
