; TODO handle llvm 13
; RUN: if [ %llvmver -lt 13 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -adce -loop-deletion -correlated-propagation -simplifycfg -S | FileCheck %s; fi
source_filename = "incloop.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }

@stderr = external dso_local local_unnamed_addr global %struct._IO_FILE*, align 8
@.str = private unnamed_addr constant [68 x i8] c"Assertion Failed: fabs( [%s = %g] - [%s = %g] ) > %g at %s:%d (%s)\0A\00", align 1
@.str.1 = private unnamed_addr constant [4 x i8] c"res\00", align 1
@.str.2 = private unnamed_addr constant [3 x i8] c"dx\00", align 1
@.str.3 = private unnamed_addr constant [10 x i8] c"incloop.c\00", align 1
@__PRETTY_FUNCTION__.main = private unnamed_addr constant [23 x i8] c"int main(int, char **)\00", align 1

; Function Attrs: nounwind readnone uwtable
define dso_local zeroext i1 @approx_fp_equality_float(float %f1, float %f2, double %threshold) local_unnamed_addr #0 {
entry:
  %sub = fsub fast float %f1, %f2
  %0 = tail call fast float @llvm.fabs.f32(float %sub)
  %1 = fpext float %0 to double
  %cmp = fcmp fast ule double %1, %threshold
  ret i1 %cmp
}

; Function Attrs: nounwind readnone speculatable
declare double @llvm.fabs.f64(double) #1

; Function Attrs: nounwind uwtable
define dso_local double @compute_loops(double* nocapture %a, i32 %b, i32 %n) #2 {
entry:
  %cmp = icmp sgt i32 %b, 0
  %mul = mul nsw i32 %n, %b
  %cmp119 = icmp sgt i32 %mul, 0
  br i1 %cmp119, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %0 = zext i32 %b to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %sum.0.lcssa = phi double [ 0.000000e+00, %entry ], [ %add, %for.body ]
  ret double %sum.0.lcssa

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %sum.020 = phi double [ 0.000000e+00, %for.body.preheader ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds double, double* %a, i64 %indvars.iv
  %1 = load double, double* %arrayidx, align 8, !tbaa !2
  %mul4 = fmul fast double %1, %1
  %add = fadd fast double %mul4, %sum.020
  store double 0.000000e+00, double* %arrayidx, align 8, !tbaa !2
  %indvars.iv.next = add nsw nuw i64 %indvars.iv, %0
  %2 = trunc i64 %indvars.iv.next to i32
  %cmp1 = icmp sgt i32 %mul, %2
  br i1 %cmp1, label %for.body, label %for.cond.cleanup
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #3

; Function Attrs: nounwind
declare void @llvm.assume(i1) #4

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #3

; Function Attrs: nounwind uwtable
define dso_local i32 @main(i32 %argc, i8** nocapture readnone %argv) local_unnamed_addr #2 {
entry:
  %x = alloca double, align 8
  %0 = bitcast double* %x to i8*
  br label %for.body

for.cond.cleanup:                                 ; preds = %if.end
  ret i32 0

for.body:                                         ; preds = %entry, %if.end
  %i.012 = phi i32 [ -99, %entry ], [ %inc, %if.end ]
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %0) #4
  %conv = sitofp i32 %i.012 to double
  %div = fmul fast double %conv, 1.000000e-02
  store double %div, double* %x, align 8, !tbaa !2
  %1 = call double (...) @__enzyme_autodiff.f64(double (double*, i32, i32)* nonnull @compute_loops, double* nonnull %x, double* nonnull %x, i32 1, i32 1) #4
  %2 = load double, double* %x, align 8, !tbaa !2
  %mul = fmul fast double %2, %2
  %sub = fsub fast double 1.000000e+00, %mul
  %3 = call fast double @llvm.sqrt.f64(double %sub)
  %div1 = fdiv fast double 1.000000e+00, %3
  %sub2 = fsub fast double %1, %div1
  %4 = call fast double @llvm.fabs.f64(double %sub2)
  %cmp3 = fcmp fast ogt double %4, 1.000000e-10
  br i1 %cmp3, label %if.then, label %if.end

if.then:                                          ; preds = %for.body
  %5 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !6
  %call = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %5, i8* getelementptr inbounds ([68 x i8], [68 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.1, i64 0, i64 0), double %1, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.2, i64 0, i64 0), double %div1, double 1.000000e-10, i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str.3, i64 0, i64 0), i32 35, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @__PRETTY_FUNCTION__.main, i64 0, i64 0)) #7
  call void @abort() #8
  unreachable

if.end:                                           ; preds = %for.body
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %0) #4
  %inc = add nsw i32 %i.012, 1
  %cmp = icmp slt i32 %i.012, 99
  br i1 %cmp, label %for.body, label %for.cond.cleanup
}

declare double @__enzyme_autodiff.f64(...) local_unnamed_addr

; Function Attrs: nounwind readnone speculatable
declare double @llvm.sqrt.f64(double) #1

; Function Attrs: nounwind
declare dso_local i32 @fprintf(%struct._IO_FILE* nocapture, i8* nocapture readonly, ...) local_unnamed_addr #5

; Function Attrs: noreturn nounwind
declare dso_local void @abort() local_unnamed_addr #6

; Function Attrs: nounwind readnone speculatable
declare float @llvm.fabs.f32(float) #1

attributes #0 = { nounwind readnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #3 = { argmemonly nounwind }
attributes #4 = { nounwind }
attributes #5 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #6 = { noreturn nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #7 = { cold }
attributes #8 = { noreturn nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0}
!3 = !{!"double", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
!6 = !{!7, !7, i64 0}
!7 = !{!"any pointer", !4, i64 0}

; CHECK: define internal void @diffecompute_loops(double* nocapture %a, double* nocapture %"a'", i32 %b, i32 %n, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %mul = mul nsw i32 %n, %b
; CHECK-NEXT:   %cmp119 = icmp sgt i32 %mul, 0
; CHECK-NEXT:   br i1 %cmp119, label %for.body.preheader, label %invertfor.cond.cleanup

; CHECK: for.body.preheader:                               ; preds = %entry
; CHECK-NEXT:   %0 = zext i32 %b to i64
; CHECK-NEXT:   %1 = mul i32 %n, %b
; CHECK-NEXT:   %2 = add i32 %1, -1
; CHECK-NEXT:   %3 = zext i32 %2 to i64
; CHECK-NEXT:   %.lhs.trunc = trunc i64 %3 to i32
; CHECK-NEXT:   %.rhs.trunc = trunc i64 %0 to i32
; CHECK-NEXT:   %4 = udiv i32 %.lhs.trunc, %.rhs.trunc
; CHECK-NEXT:   %.zext = zext i32 %4 to i64
; CHECK-NEXT:   %5 = add nuw{{( nsw)?}} i64 %.zext, 1
; CHECK-NEXT:   %mallocsize = mul nuw nsw i64 %5, 8
; CHECK-NEXT:   %malloccall = tail call noalias nonnull i8* @malloc(i64 %mallocsize)
; CHECK-NEXT:   %_malloccache = bitcast i8* %malloccall to double*
; CHECK-NEXT:   br label %for.body

; CHECK: for.body:                                         ; preds = %for.body, %for.body.preheader
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %for.body ], [ 0, %for.body.preheader ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %6 = mul i64 %0, %iv
; CHECK-NEXT:   %arrayidx = getelementptr inbounds double, double* %a, i64 %6
; CHECK-NEXT:   %7 = load double, double* %arrayidx, align 8, !tbaa !2
; CHECK-NEXT:   store double 0.000000e+00, double* %arrayidx, align 8, !tbaa !2
; CHECK-NEXT:   %8 = getelementptr inbounds double, double* %_malloccache, i64 %iv
; CHECK-NEXT:   store double %7, double* %8, align 8, !tbaa !2, !invariant.group ![[igroup:[0-9]+]]
; CHECK-NEXT:   %indvars.iv.next = add nuw nsw i64 %6, %0
; CHECK-NEXT:   %9 = trunc i64 %indvars.iv.next to i32
; CHECK-NEXT:   %cmp1 = icmp sgt i32 %mul, %9
; CHECK-NEXT:   br i1 %cmp1, label %for.body, label %invertfor.cond.cleanup

; CHECK: invertentry:                                      ; preds = %invertfor.cond.cleanup, %invertfor.body.preheader
; CHECK-NEXT:   ret void

; CHECK: invertfor.body.preheader:                         ; preds = %invertfor.body
; CHECK-NEXT:   %10 = bitcast double* %_cache.0 to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %10)
; CHECK-NEXT:   br label %invertentry

; CHECK: invertfor.cond.cleanup.loopexit:                  ; preds = %invertfor.cond.cleanup
; CHECK-NEXT:   %_unwrap = mul i32 %n, %b
; CHECK-NEXT:   %_unwrap1 = add i32 %_unwrap, -1
; CHECK-NEXT:   %_unwrap2 = zext i32 %_unwrap1 to i64
; CHECK-NEXT:   %_unwrap3 = zext i32 %b to i64
; CHECK-NEXT:   %_unwrap4.lhs.trunc = trunc i64 %_unwrap2 to i32
; CHECK-NEXT:   %_unwrap4.rhs.trunc = trunc i64 %_unwrap3 to i32
; CHECK-NEXT:   %[[unwrap412:.+]] = udiv i32 %_unwrap4.lhs.trunc, %_unwrap4.rhs.trunc
; CHECK-NEXT:   %_unwrap4.zext = zext i32 %[[unwrap412]] to i64
; CHECK-NEXT:   br label %invertfor.body

; CHECK: invertfor.cond.cleanup:                           ; preds = %entry, %for.body
; CHECK-NEXT:   %_cache.0 = phi double* [ undef, %entry ], [ %_malloccache, %for.body ]
; CHECK-NEXT:   %11 = select{{( fast)?}} i1 %cmp119, double %differeturn, double 0.000000e+00
; CHECK-NEXT:   br i1 %cmp119, label %invertfor.cond.cleanup.loopexit, label %invertentry

; CHECK: invertfor.body:                                   ; preds = %incinvertfor.body, %invertfor.cond.cleanup.loopexit
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %_unwrap4.zext, %invertfor.cond.cleanup.loopexit ], [ %19, %incinvertfor.body ]
; CHECK-NEXT:   %_unwrap5 = zext i32 %b to i64
; CHECK-NEXT:   %_unwrap6 = mul i64 %_unwrap5, %"iv'ac.0"
; CHECK-NEXT:   %"arrayidx'ipg_unwrap" = getelementptr inbounds double, double* %"a'", i64 %_unwrap6
; CHECK-NEXT:   store double 0.000000e+00, double* %"arrayidx'ipg_unwrap", align 8
; CHECK-NEXT:   %12 = getelementptr inbounds double, double* %_cache.0, i64 %"iv'ac.0"
; CHECK-NEXT:   %13 = load double, double* %12, align 8, !tbaa !2, !invariant.group ![[igroup]]
; CHECK-NEXT:   %m0diffe = fmul fast double %differeturn, %13
; CHECK-NEXT:   %m1diffe = fmul fast double %differeturn, %13
; CHECK-NEXT:   %14 = fadd fast double %m0diffe, %m1diffe
; CHECK-NEXT:   %15 = load double, double* %"arrayidx'ipg_unwrap", align 8
; CHECK-NEXT:   %16 = fadd fast double %15, %14
; CHECK-NEXT:   store double %16, double* %"arrayidx'ipg_unwrap", align 8
; CHECK-NEXT:   %17 = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   %18 = select{{( fast)?}} i1 %17, double 0.000000e+00, double %differeturn
; CHECK-NEXT:   br i1 %17, label %invertfor.body.preheader, label %incinvertfor.body

; CHECK: incinvertfor.body:                                ; preds = %invertfor.body
; CHECK-NEXT:   %19 = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertfor.body
; CHECK-NEXT: }
