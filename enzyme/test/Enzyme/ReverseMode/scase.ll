; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -adce -loop-deletion -correlated-propagation -simplifycfg -S | FileCheck %s
source_filename = "/mnt/Data/git/Enzyme/enzyme/test/Integration/taylorlog.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }

@stderr = external dso_local local_unnamed_addr global %struct._IO_FILE*, align 8
@.str = private unnamed_addr constant [68 x i8] c"Assertion Failed: fabs( [%s = %g] - [%s = %g] ) > %g at %s:%d (%s)\0A\00", align 1
@.str.1 = private unnamed_addr constant [4 x i8] c"ret\00", align 1
@.str.2 = private unnamed_addr constant [4 x i8] c"2.0\00", align 1
@.str.3 = private unnamed_addr constant [57 x i8] c"/mnt/Data/git/Enzyme/enzyme/test/Integration/taylorlog.c\00", align 1
@__PRETTY_FUNCTION__.main = private unnamed_addr constant [23 x i8] c"int main(int, char **)\00", align 1

; Function Attrs: nounwind uwtable
define dso_local i32 @main(i32 %argc, i8** nocapture readnone %argv) local_unnamed_addr #0 {
entry:
  %call = tail call fast double @__enzyme_autodiff(i8* bitcast (double (double, i32)* @taylorlog to i8*), double 5.000000e-01, i32 10000) #6
  %sub = fadd fast double %call, -2.000000e+00
  %0 = tail call fast double @llvm.fabs.f64(double %sub)
  %cmp = fcmp fast ogt double %0, 0x3E7AD7F29ABCAF48
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %1 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !2
  %call1 = tail call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %1, i8* getelementptr inbounds ([68 x i8], [68 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.1, i64 0, i64 0), double %call, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.2, i64 0, i64 0), double 2.000000e+00, double 0x3E7AD7F29ABCAF48, i8* getelementptr inbounds ([57 x i8], [57 x i8]* @.str.3, i64 0, i64 0), i32 31, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @__PRETTY_FUNCTION__.main, i64 0, i64 0)) #7
  tail call void @abort() #8
  unreachable

if.end:                                           ; preds = %entry
  ret i32 0
}

declare dso_local double @__enzyme_autodiff(i8*, double, i32) local_unnamed_addr #1

; Function Attrs: nounwind readnone uwtable
define internal double @taylorlog(double %x, i32 %SINCOSN) #2 {
entry:
  %cmp8 = icmp eq i32 %SINCOSN, 0
  br i1 %cmp8, label %for.cond.cleanup, label %for.body

for.body:                                         ; preds = %for.body, %for.body.preheader.new
  %i = phi i32 [ 1, %entry ], [ %inc, %for.body ]
  %sum.09 = phi double [ 0.000000e+00, %entry ], [ %add, %for.body ]
  %conv = sitofp i32 %i to double
  %z = tail call fast double @llvm.pow.f64(double %x, double %conv)
  %div = fdiv fast double %z, %conv
  %add = fadd fast double %div, %sum.09
  %inc = add nuw nsw i32 %i, 1
  %end = icmp eq i32 %inc, %SINCOSN
  br i1 %end, label %lcssa, label %for.body

lcssa:
  %lcmp.mod = icmp ne i32 %SINCOSN, 1
  br i1 %lcmp.mod, label %for.cond.cleanup, label %bad

bad:
  br label %for.cond.cleanup

for.cond.cleanup:
  %total = phi double [ 0.000000e+00, %entry ], [ %add, %lcssa ], [ %x, %bad ]
  ret double %total
}

; Function Attrs: nounwind readnone speculatable
declare double @llvm.fabs.f64(double) #3

; Function Attrs: nounwind
declare dso_local i32 @fprintf(%struct._IO_FILE* nocapture, i8* nocapture readonly, ...) local_unnamed_addr #4

; Function Attrs: noreturn nounwind
declare dso_local void @abort() local_unnamed_addr #5

; Function Attrs: nounwind readnone speculatable
declare double @llvm.pow.f64(double, double) #3

attributes #0 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #2 = { nounwind readnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #3 = { nounwind readnone speculatable }
attributes #4 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #5 = { noreturn nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #6 = { nounwind }
attributes #7 = { cold }
attributes #8 = { noreturn nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0}
!3 = !{!"any pointer", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}

; CHECK: define internal { double } @diffetaylorlog(double %x, i32 %SINCOSN, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %cmp8 = icmp eq i32 %SINCOSN, 0
; CHECK-NEXT:   %lcmp.mod_unwrap = icmp ne i32 %SINCOSN, 1
; CHECK-NEXT:   %anot1_ = xor i1 %cmp8, true
; CHECK-NEXT:   %[[andVal:.+]] = and i1 %lcmp.mod_unwrap, %anot1_
; CHECK-NEXT:   %bnot1_ = xor i1 %lcmp.mod_unwrap, true
; CHECK-NEXT:   %0 = select{{( fast)?}} i1 %bnot1_, double %differeturn, double 0.000000e+00
; CHECK-NEXT:   %1 = select{{( fast)?}} i1 %[[andVal]], double %differeturn, double 0.000000e+00
; CHECK-NEXT:   br i1 %cmp8, label %invertentry, label %staging

; CHECK: invertentry:                                      ; preds = %invertfor.body, %entry
; CHECK-NEXT:   %"x'de.0" = phi double [ %0, %entry ], [ %7, %invertfor.body ]
; CHECK-NEXT:   %2 = insertvalue { double } undef, double %"x'de.0", 0
; CHECK-NEXT:   ret { double } %2

; CHECK: invertfor.body:                                   ; preds = %staging, %incinvertfor.body
; CHECK-NEXT:   %"x'de.1" = phi double [ %0, %staging ], [ %7, %incinvertfor.body ]
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %_unwrap2, %staging ], [ %10, %incinvertfor.body ]
; CHECK-NEXT:   %iv.next_unwrap = add nuw nsw i64 %"iv'ac.0", 1
; CHECK-NEXT:   %_unwrap = trunc i64 %iv.next_unwrap to i32
; CHECK-NEXT:   %conv_unwrap = sitofp i32 %_unwrap to double
; CHECK-NEXT:   %d0diffez = fdiv fast double %1, %conv_unwrap
; CHECK-NEXT:   %3 = fsub fast double %conv_unwrap, 1.000000e+00
; CHECK-NEXT:   %4 = call fast double @llvm.pow.f64(double %x, double %3)
; CHECK-NEXT:   %5 = fmul fast double %d0diffez, %4
; CHECK-NEXT:   %6 = fmul fast double %5, %conv_unwrap
; CHECK-NEXT:   %7 = fadd fast double %"x'de.1", %6
; CHECK-NEXT:   %8 = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   %9 = select{{( fast)?}} i1 %8, double 0.000000e+00, double %1
; CHECK-NEXT:   br i1 %8, label %invertentry, label %incinvertfor.body

; CHECK: incinvertfor.body:                                ; preds = %invertfor.body
; CHECK-NEXT:   %10 = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertfor.body

; CHECK: staging:                                          ; preds = %entry
; CHECK-NEXT:   %_unwrap1 = add i32 %SINCOSN, -2
; CHECK-NEXT:   %_unwrap2 = zext i32 %_unwrap1 to i64
; CHECK-NEXT:   br label %invertfor.body
; CHECK-NEXT: }
