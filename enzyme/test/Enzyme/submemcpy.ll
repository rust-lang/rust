; RUN: %opt < %s %loadEnzyme -enzyme -enzyme_preopt=false -mem2reg -sroa -simplifycfg -instcombine -adce -S | FileCheck %s
source_filename = "submemcpy.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind readnone uwtable
define dso_local zeroext i1 @approx_fp_equality_float(float %f1, float %f2, double %threshold) local_unnamed_addr #0 {
entry:
  %sub = fsub float %f1, %f2
  %0 = tail call float @llvm.fabs.f32(float %sub)
  %1 = fpext float %0 to double
  %cmp = fcmp ule double %1, %threshold
  ret i1 %cmp
}

; Function Attrs: noinline nounwind uwtable
define dso_local void @submemcpy(i8* nocapture %sdst, i8* nocapture readonly %ssrc, i32 %sN) local_unnamed_addr #1 {
entry:
  %conv = sext i32 %sN to i64
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %sdst, i8* align 1 %ssrc, i64 %conv, i1 false)
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1) #2

; Function Attrs: nounwind uwtable
define dso_local void @foo(float* nocapture %fdst, float* nocapture readonly %fsrc, i32 %fN) #3 {
entry:
  %0 = bitcast float* %fdst to i8*
  %1 = bitcast float* %fsrc to i8*
  %mul = shl i32 %fN, 2
  tail call void @submemcpy(i8* %0, i8* %1, i32 %mul)
  ret void
}

; Function Attrs: nounwind uwtable
define dso_local void @call(float* %cdst, float* %cdstp, float* %csrc, float* %csrcp, i32 %cN) local_unnamed_addr #3 {
entry:
  %call = tail call double (i8*, ...) @__enzyme_autodiff(i8* bitcast (void (float*, float*, i32)* @foo to i8*), float* %cdst, float* %cdstp, float* %csrc, float* %csrcp, i32 %cN) #6
  ret void
}

declare dso_local double @__enzyme_autodiff(i8*, ...) local_unnamed_addr #4

; Function Attrs: nounwind readnone speculatable
declare float @llvm.fabs.f32(float) #5

attributes #0 = { nounwind readnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { argmemonly nounwind }
attributes #3 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { nounwind readnone speculatable }
attributes #6 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}

; CHECK: define internal void @diffefoo(float* nocapture %fdst, float* nocapture %"fdst'", float* nocapture readonly %fsrc, float* nocapture %"fsrc'", i32 %fN)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[ipc:.+]] = bitcast float* %"fdst'" to i8*
; CHECK-NEXT:   %[[uw:.+]] = bitcast float* %fdst to i8*
; CHECK-NEXT:   %[[ipc2:.+]] = bitcast float* %"fsrc'" to i8*
; CHECK-NEXT:   %[[uw1:.+]] = bitcast float* %fsrc to i8*
; CHECK-NEXT:   %[[mul:.+]] = shl i32 %fN, 2
; CHECK-NEXT:   call void @diffesubmemcpy(i8* %[[uw]], i8* %[[ipc]], i8* %[[uw1]], i8* %[[ipc2]], i32 %[[mul]])
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @diffesubmemcpy(i8* nocapture %sdst, i8* nocapture %"sdst'", i8* nocapture readonly %ssrc, i8* nocapture %"ssrc'", i32 %sN)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %conv = sext i32 %sN to i64
; CHECK-NEXT:   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %sdst, i8* align 1 %ssrc, i64 %conv, i1 false)
; CHECK-NEXT:   %0 = bitcast i8* %"sdst'" to float*
; CHECK-NEXT:   %1 = bitcast i8* %"ssrc'" to float*
; CHECK-NEXT:   %2 = lshr i64 %conv, 2
; CHECK-NEXT:   call void @__enzyme_memcpyadd_floatda1sa1(float* %0, float* %1, i64 %2)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
