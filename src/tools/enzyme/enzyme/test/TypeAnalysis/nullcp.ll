; RUN: %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=compute_loops -o /dev/null | FileCheck %s

source_filename = "nullcp.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@.str = private unnamed_addr constant [8 x i8] c"res=%f\0A\00", align 1
@diffe_dup = external dso_local local_unnamed_addr global i32, align 4



; Function Attrs: nounwind uwtable
define dso_local void @compute_loops(double %x, i8* nocapture %out) #1 {
entry:
  %0 = bitcast i8* %out to double*
  %arrayidx = getelementptr inbounds i8, i8* %out, i64 32
  %1 = bitcast i8* %arrayidx to double*
  store double %x, double* %1, align 8, !tbaa !2
  tail call void @copy(double* %0, double* null, i64 3)
  ret void
}

; Function Attrs: nounwind uwtable
define dso_local void @copy(double* nocapture %dst, double* readonly %src, i64 %n) local_unnamed_addr #1 {
entry:
  %cmp = icmp eq double* %src, null
  br i1 %cmp, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %0 = bitcast double* %dst to i8*
  %1 = bitcast double* %src to i8*
  %mul = shl i64 %n, 3
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %0, i8* align 8 %1, i64 %mul, i1 false)
  br label %if.end

if.end:                                           ; preds = %entry, %if.then
  ret void
}


; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1) #2

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #2

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1) #2

; Function Attrs: nounwind
declare dso_local i32 @printf(i8* nocapture readonly, ...) local_unnamed_addr #3

declare dso_local double @__enzyme_autodiff(i8*, double, i32, i8*, i8*) local_unnamed_addr #4

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #2

; Function Attrs: nounwind readnone speculatable
declare float @llvm.fabs.f32(float) #5

attributes #0 = { nounwind readnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #2 = { argmemonly nounwind }
attributes #3 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #4 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #5 = { nounwind readnone speculatable }
attributes #6 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0}
!3 = !{!"double", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
!6 = !{!7, !7, i64 0}
!7 = !{!"int", !4, i64 0}

; CHECK: compute_loops - {} |{[-1]:Float@double}:{} {[-1]:Pointer}:{} 
; CHECK-NEXT: double %x: {[-1]:Float@double}
; CHECK-NEXT: i8* %out: {[-1]:Pointer, [-1,32]:Float@double}
; CHECK-NEXT: entry
; CHECK-NEXT:   %0 = bitcast i8* %out to double*: {[-1]:Pointer, [-1,32]:Float@double}
; CHECK-NEXT:   %arrayidx = getelementptr inbounds i8, i8* %out, i64 32: {[-1]:Pointer, [-1,0]:Float@double}
; CHECK-NEXT:   %1 = bitcast i8* %arrayidx to double*: {[-1]:Pointer, [-1,0]:Float@double}
; CHECK-NEXT:   store double %x, double* %1, align 8, !tbaa !2: {}
; CHECK-NEXT:   tail call void @copy(double* %0, double* null, i64 3): {}
; CHECK-NEXT:   ret void: {}
; CHECK: copy - {} |{[-1]:Pointer, [-1,32]:Float@double}:{} {[-1]:Pointer, [-1,-1]:Anything}:{0,} {[-1]:Integer}:{3,} 
; CHECK-NEXT: double* %dst: {[-1]:Pointer, [-1,32]:Float@double}
; CHECK-NEXT: double* %src: {[-1]:Pointer, [-1,-1]:Anything}
; CHECK-NEXT: i64 %n: {[-1]:Integer}
; CHECK-NEXT: entry
; CHECK-NEXT:   %cmp = icmp eq double* %src, null: {[-1]:Integer}
; CHECK-NEXT:   br i1 %cmp, label %if.end, label %if.then: {}
; CHECK-NEXT: if.then
; CHECK-NEXT:   %0 = bitcast double* %dst to i8*: {[-1]:Pointer, [-1,32]:Float@double}
; CHECK-NEXT:   %1 = bitcast double* %src to i8*: {[-1]:Pointer, [-1,-1]:Anything}
; CHECK-NEXT:   %mul = shl i64 %n, 3: {[-1]:Integer}
; CHECK-NEXT:   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %0, i8* align 8 %1, i64 %mul, i1 false): {}
; CHECK-NEXT:   br label %if.end: {}
; CHECK-NEXT: if.end
; CHECK-NEXT:   ret void: {}
