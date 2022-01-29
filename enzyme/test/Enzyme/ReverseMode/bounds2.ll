; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -early-cse -instsimplify -jump-threading -adce -S | FileCheck %s

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }

@stderr = external dso_local local_unnamed_addr global %struct._IO_FILE*, align 8
@.str = private unnamed_addr constant [68 x i8] c"Assertion Failed: fabs( [%s = %g] - [%s = %g] ) > %g at %s:%d (%s)\0A\00", align 1
@.str.1 = private unnamed_addr constant [6 x i8] c"da[i]\00", align 1
@.str.2 = private unnamed_addr constant [5 x i8] c"1.0f\00", align 1
@.str.3 = private unnamed_addr constant [9 x i8] c"bounds.c\00", align 1
@__PRETTY_FUNCTION__.main = private unnamed_addr constant [23 x i8] c"int main(int, char **)\00", align 1

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
define dso_local float @lookup(float* nocapture readonly %data, i32 %i, i32 %bound) local_unnamed_addr #1 {
entry:
  %cmp = icmp sge i32 %i, %bound
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @exit(i32 1) #8
  unreachable

if.end:                                           ; preds = %entry
  %idxprom = sext i32 %i to i64
  %arrayidx = getelementptr inbounds float, float* %data, i64 %idxprom
  %0 = load float, float* %arrayidx, align 4, !tbaa !2
  ret float %0
}

; Function Attrs: noreturn nounwind
declare dso_local void @exit(i32) local_unnamed_addr #2

; Function Attrs: nounwind uwtable
define dso_local float @bounds(float* nocapture readonly %a, i32 %bound) #3 {
entry:
  %cmp7 = icmp sgt i32 %bound, 0
  br i1 %cmp7, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %sum.0.lcssa = phi float [ 0.000000e+00, %entry ], [ %add, %for.body ]
  ret float %sum.0.lcssa

for.body:                                         ; preds = %entry, %for.body
  %i.09 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %sum.08 = phi float [ %add, %for.body ], [ 0.000000e+00, %entry ]
  %call = tail call float @lookup(float* %a, i32 %i.09, i32 %bound)
  %add = fadd float %sum.08, %call
  %inc = add nuw nsw i32 %i.09, 1
  %exitcond = icmp eq i32 %inc, %bound
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #4

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #4

; Function Attrs: nounwind uwtable
define dso_local i32 @main(i32 %argc, i8** nocapture readnone %argv) local_unnamed_addr #3 {
entry:
  %a = alloca [10 x float], align 16
  %da = alloca [10 x float], align 16
  %0 = bitcast [10 x float]* %a to i8*
  call void @llvm.lifetime.start.p0i8(i64 40, i8* nonnull %0) #9
  call void @llvm.memset.p0i8.i64(i8* nonnull align 16 %0, i8 0, i64 40, i1 false)
  %1 = getelementptr inbounds [10 x float], [10 x float]* %a, i64 0, i64 0
  store float 1.000000e+00, float* %1, align 16
  %2 = bitcast [10 x float]* %da to i8*
  call void @llvm.lifetime.start.p0i8(i64 40, i8* nonnull %2) #9
  call void @llvm.memset.p0i8.i64(i8* nonnull align 16 %2, i8 0, i64 40, i1 false)
  %arraydecay1 = getelementptr inbounds [10 x float], [10 x float]* %da, i64 0, i64 0
  %call = call double (i8*, ...) @__enzyme_autodiff(i8* bitcast (float (float*, i32)* @bounds to i8*), float* nonnull %1, float* nonnull %arraydecay1, i32 10) #9
  %3 = load float, float* %arraydecay1, align 16, !tbaa !2
  %sub = fadd float %3, -1.000000e+00
  %4 = call float @llvm.fabs.f32(float %sub)
  %5 = fpext float %4 to double
  %cmp2 = fcmp ogt double %5, 1.000000e-10
  br i1 %cmp2, label %if.then, label %for.cond

for.cond:                                         ; preds = %entry
  %arrayidx.1 = getelementptr inbounds [10 x float], [10 x float]* %da, i64 0, i64 1
  %6 = load float, float* %arrayidx.1, align 4, !tbaa !2
  %sub.1 = fadd float %6, -1.000000e+00
  %7 = call float @llvm.fabs.f32(float %sub.1)
  %8 = fpext float %7 to double
  %cmp2.1 = fcmp ogt double %8, 1.000000e-10
  br i1 %cmp2.1, label %if.then, label %for.cond.1

if.then:                                          ; preds = %for.cond.8, %for.cond.7, %for.cond.6, %for.cond.5, %for.cond.4, %for.cond.3, %for.cond.2, %for.cond.1, %for.cond, %entry
  %.lcssa = phi float [ %3, %entry ], [ %6, %for.cond ], [ %10, %for.cond.1 ], [ %13, %for.cond.2 ], [ %16, %for.cond.3 ], [ %19, %for.cond.4 ], [ %22, %for.cond.5 ], [ %25, %for.cond.6 ], [ %28, %for.cond.7 ], [ %31, %for.cond.8 ]
  %9 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !6
  %conv6 = fpext float %.lcssa to double
  %call7 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %9, i8* getelementptr inbounds ([68 x i8], [68 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.1, i64 0, i64 0), double %conv6, i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.2, i64 0, i64 0), double 1.000000e+00, double 1.000000e-10, i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str.3, i64 0, i64 0), i32 48, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @__PRETTY_FUNCTION__.main, i64 0, i64 0)) #10
  call void @abort() #8
  unreachable

for.cond.1:                                       ; preds = %for.cond
  %arrayidx.2 = getelementptr inbounds [10 x float], [10 x float]* %da, i64 0, i64 2
  %10 = load float, float* %arrayidx.2, align 8, !tbaa !2
  %sub.2 = fadd float %10, -1.000000e+00
  %11 = call float @llvm.fabs.f32(float %sub.2)
  %12 = fpext float %11 to double
  %cmp2.2 = fcmp ogt double %12, 1.000000e-10
  br i1 %cmp2.2, label %if.then, label %for.cond.2

for.cond.2:                                       ; preds = %for.cond.1
  %arrayidx.3 = getelementptr inbounds [10 x float], [10 x float]* %da, i64 0, i64 3
  %13 = load float, float* %arrayidx.3, align 4, !tbaa !2
  %sub.3 = fadd float %13, -1.000000e+00
  %14 = call float @llvm.fabs.f32(float %sub.3)
  %15 = fpext float %14 to double
  %cmp2.3 = fcmp ogt double %15, 1.000000e-10
  br i1 %cmp2.3, label %if.then, label %for.cond.3

for.cond.3:                                       ; preds = %for.cond.2
  %arrayidx.4 = getelementptr inbounds [10 x float], [10 x float]* %da, i64 0, i64 4
  %16 = load float, float* %arrayidx.4, align 16, !tbaa !2
  %sub.4 = fadd float %16, -1.000000e+00
  %17 = call float @llvm.fabs.f32(float %sub.4)
  %18 = fpext float %17 to double
  %cmp2.4 = fcmp ogt double %18, 1.000000e-10
  br i1 %cmp2.4, label %if.then, label %for.cond.4

for.cond.4:                                       ; preds = %for.cond.3
  %arrayidx.5 = getelementptr inbounds [10 x float], [10 x float]* %da, i64 0, i64 5
  %19 = load float, float* %arrayidx.5, align 4, !tbaa !2
  %sub.5 = fadd float %19, -1.000000e+00
  %20 = call float @llvm.fabs.f32(float %sub.5)
  %21 = fpext float %20 to double
  %cmp2.5 = fcmp ogt double %21, 1.000000e-10
  br i1 %cmp2.5, label %if.then, label %for.cond.5

for.cond.5:                                       ; preds = %for.cond.4
  %arrayidx.6 = getelementptr inbounds [10 x float], [10 x float]* %da, i64 0, i64 6
  %22 = load float, float* %arrayidx.6, align 8, !tbaa !2
  %sub.6 = fadd float %22, -1.000000e+00
  %23 = call float @llvm.fabs.f32(float %sub.6)
  %24 = fpext float %23 to double
  %cmp2.6 = fcmp ogt double %24, 1.000000e-10
  br i1 %cmp2.6, label %if.then, label %for.cond.6

for.cond.6:                                       ; preds = %for.cond.5
  %arrayidx.7 = getelementptr inbounds [10 x float], [10 x float]* %da, i64 0, i64 7
  %25 = load float, float* %arrayidx.7, align 4, !tbaa !2
  %sub.7 = fadd float %25, -1.000000e+00
  %26 = call float @llvm.fabs.f32(float %sub.7)
  %27 = fpext float %26 to double
  %cmp2.7 = fcmp ogt double %27, 1.000000e-10
  br i1 %cmp2.7, label %if.then, label %for.cond.7

for.cond.7:                                       ; preds = %for.cond.6
  %arrayidx.8 = getelementptr inbounds [10 x float], [10 x float]* %da, i64 0, i64 8
  %28 = load float, float* %arrayidx.8, align 16, !tbaa !2
  %sub.8 = fadd float %28, -1.000000e+00
  %29 = call float @llvm.fabs.f32(float %sub.8)
  %30 = fpext float %29 to double
  %cmp2.8 = fcmp ogt double %30, 1.000000e-10
  br i1 %cmp2.8, label %if.then, label %for.cond.8

for.cond.8:                                       ; preds = %for.cond.7
  %arrayidx.9 = getelementptr inbounds [10 x float], [10 x float]* %da, i64 0, i64 9
  %31 = load float, float* %arrayidx.9, align 4, !tbaa !2
  %sub.9 = fadd float %31, -1.000000e+00
  %32 = call float @llvm.fabs.f32(float %sub.9)
  %33 = fpext float %32 to double
  %cmp2.9 = fcmp ogt double %33, 1.000000e-10
  br i1 %cmp2.9, label %if.then, label %for.cond.9

for.cond.9:                                       ; preds = %for.cond.8
  call void @llvm.lifetime.end.p0i8(i64 40, i8* nonnull %2) #9
  call void @llvm.lifetime.end.p0i8(i64 40, i8* nonnull %0) #9
  ret i32 0
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1) #4

declare dso_local double @__enzyme_autodiff(i8*, ...) local_unnamed_addr #5

; Function Attrs: nounwind
declare dso_local i32 @fprintf(%struct._IO_FILE* nocapture, i8* nocapture readonly, ...) local_unnamed_addr #6

; Function Attrs: noreturn nounwind
declare dso_local void @abort() local_unnamed_addr #2

; Function Attrs: nounwind readnone speculatable
declare float @llvm.fabs.f32(float) #7

attributes #0 = { nounwind readnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { noreturn nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { argmemonly nounwind }
attributes #5 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #7 = { nounwind readnone speculatable }
attributes #8 = { noreturn nounwind }
attributes #9 = { nounwind }
attributes #10 = { cold }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0}
!3 = !{!"float", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
!6 = !{!7, !7, i64 0}
!7 = !{!"any pointer", !4, i64 0}

; CHECK: define internal void @diffebounds(float* nocapture readonly %a, float* nocapture %"a'", i32 %bound, float %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %cmp7 = icmp sgt i32 %bound, 0
; CHECK-NEXT:   br i1 %cmp7, label %for.body.preheader, label %invertentry

; CHECK: for.body.preheader:                               ; preds = %entry
; CHECK-NEXT:   br label %for.body

; CHECK: for.body:                                         ; preds = %for.body, %for.body.preheader
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %for.body ], [ 0, %for.body.preheader ] 
; CHECK-DAG:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-DAG:   %0 = trunc i64 %iv to i32
; CHECK-NEXT:   @augmented_lookup(float* %a, float* %"a'", i32 %0, i32 %bound)
; CHECK-NEXT:   %inc = add nuw nsw i32 %0, 1
; CHECK-NEXT:   %exitcond = icmp eq i32 %inc, %bound
; CHECK-NEXT:   br i1 %exitcond, label %invertfor.cond.cleanup, label %for.body

; CHECK: invertentry:                                      ; preds = %entry, %invertfor.body, %invertfor.cond.cleanup
; CHECK-NEXT:   ret void

; CHECK: mergeinvertfor.body_for.cond.cleanup.loopexit:    ; preds = %invertfor.cond.cleanup
; CHECK-NEXT:   %_unwrap = add i32 %bound, -1
; CHECK-NEXT:   %_unwrap1 = zext i32 %_unwrap to i64
; CHECK-NEXT:   br label %invertfor.body

; CHECK: invertfor.cond.cleanup:                           ; preds = %for.body
; CHECK-NEXT:   %1 = select{{( fast)?}} i1 %cmp7, float %differeturn, float 0.000000e+00
; CHECK-NEXT:   br i1 %cmp7, label %mergeinvertfor.body_for.cond.cleanup.loopexit, label %invertentry

; CHECK: invertfor.body:                                   ; preds = %incinvertfor.body, %mergeinvertfor.body_for.cond.cleanup.loopexit
; CHECK-NEXT:   %"add'de.0" = phi float [ %1, %mergeinvertfor.body_for.cond.cleanup.loopexit ], [ %[[asel:.+]], %incinvertfor.body ]
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %_unwrap1, %mergeinvertfor.body_for.cond.cleanup.loopexit ], [ %[[ivsub:.+]], %incinvertfor.body ]
; CHECK-NEXT:   %[[trunc:.+]] = trunc i64 %"iv'ac.0" to i32
; CHECK-NEXT:   call void @diffelookup(float* %a, float* %"a'", i32 %[[trunc]], i32 %bound, float %"add'de.0")
; CHECK-NEXT:   %[[ecmp:.+]] = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   %[[asel]] = select{{( fast)?}} i1 %[[ecmp]], float 0.000000e+00, float %"add'de.0"
; CHECK-NEXT:   br i1 %[[ecmp]], label %invertentry, label %incinvertfor.body

; CHECK: incinvertfor.body:                                ; preds = %invertfor.body
; CHECK-NEXT:   %[[ivsub]] = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertfor.body
; CHECK-NEXT: }

; CHECK: define internal void @augmented_lookup(float* nocapture readonly %data, float* nocapture %"data'", i32 %i, i32 %bound)
; CHECK: entry:
; CHECK-NEXT:   %cmp = icmp sge i32 %i, %bound
; CHECK-NEXT:   br i1 %cmp, label %if.then, label %if.end

; CHECK: if.then:                                          ; preds = %entry
; CHECK-NEXT:   tail call void @exit(i32 1)
; CHECK-NEXT:   unreachable

; CHECK: if.end:                                           ; preds = %entry
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @diffelookup(float* nocapture readonly %data, float* nocapture %"data'", i32 %i, i32 %bound, float %differeturn)
; CHECK-NEXT: invertentry:
; CHECK-NEXT:   %[[idxprom:.+]] = sext i32 %i to i64
; CHECK-NEXT:   %[[arrayidxipg:.+]] = getelementptr inbounds float, float* %"data'", i64 %[[idxprom]]
; CHECK-NEXT:   %[[l1:.+]] = load float, float* %[[arrayidxipg]], align 4
; CHECK-NEXT:   %[[a1:.+]] = fadd fast float %[[l1]], %differeturn
; CHECK-NEXT:   store float %[[a1]], float* %[[arrayidxipg]], align 4
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
