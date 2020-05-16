; RUN: %opt < %s %loadEnzyme -enzyme -enzyme_preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

source_filename = "readwriteread.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@.str = private unnamed_addr constant [20 x i8] c"dx is %f ret is %f\0A\00", align 1
@.str.1 = private unnamed_addr constant [17 x i8] c"*dx == 3*2.0*2.0\00", align 1
@.str.2 = private unnamed_addr constant [16 x i8] c"readwriteread.c\00", align 1
@__PRETTY_FUNCTION__.main = private unnamed_addr constant [23 x i8] c"int main(int, char **)\00", align 1

; Function Attrs: norecurse nounwind readonly uwtable
define dso_local double @f_read(double* nocapture readonly %x) local_unnamed_addr #0 {
entry:
  %f0 = load double, double* %x, align 8, !tbaa !2
  %mul = fmul double %f0, %f0
  ret double %mul
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

; Function Attrs: norecurse nounwind uwtable
define dso_local void @g_write(double* nocapture %x, double %product) local_unnamed_addr #2 {
entry:
  %0 = load double, double* %x, align 8, !tbaa !2
  %mul = fmul double %0, %product
  store double %mul, double* %x, align 8, !tbaa !2
  ret void
}

; Function Attrs: norecurse nounwind readonly uwtable
define dso_local double @h_read(double* nocapture readonly %x) local_unnamed_addr #0 {
entry:
  %0 = load double, double* %x, align 8, !tbaa !2
  ret double %0
}

; Function Attrs: norecurse nounwind uwtable
define dso_local double @readwriteread_helper(double* nocapture %x) local_unnamed_addr #2 {
entry:
  %call = tail call double @f_read(double* %x)
  tail call void @g_write(double* %x, double %call)
  %call1 = tail call double @h_read(double* %x)
  ret double %call1
}

; Function Attrs: norecurse nounwind uwtable
define dso_local void @readwriteread(double* noalias nocapture %x, double* noalias nocapture %ret) #2 {
entry:
  %call = tail call double @readwriteread_helper(double* %x)
  store double %call, double* %ret, align 8, !tbaa !2
  ret void
}

; Function Attrs: nounwind uwtable
define dso_local i32 @main(i32 %argc, i8** nocapture readnone %argv) local_unnamed_addr #3 {
entry:
  %ret = alloca double, align 8
  %dret = alloca double, align 8
  %0 = bitcast double* %ret to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %0) #7
  store double 0.000000e+00, double* %ret, align 8, !tbaa !2
  %1 = bitcast double* %dret to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1) #7
  store double 1.000000e+00, double* %dret, align 8, !tbaa !2
  %call = tail call noalias i8* @malloc(i64 8) #7
  %2 = bitcast i8* %call to double*
  %call1 = tail call noalias i8* @malloc(i64 8) #7
  %3 = bitcast i8* %call1 to double*
  store double 2.000000e+00, double* %2, align 8, !tbaa !2
  store double 0.000000e+00, double* %3, align 8, !tbaa !2
  %call2 = call double (i8*, ...) @__enzyme_autodiff(i8* bitcast (void (double*, double*)* @readwriteread to i8*), i8* %call, i8* %call1, double* nonnull %ret, double* nonnull %dret) #7
  %4 = load double, double* %3, align 8, !tbaa !2
  %5 = load double, double* %ret, align 8, !tbaa !2
  %call3 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([20 x i8], [20 x i8]* @.str, i64 0, i64 0), double %4, double %5)
  %6 = load double, double* %3, align 8, !tbaa !2
  %cmp = fcmp oeq double %6, 1.200000e+01
  br i1 %cmp, label %cond.end, label %cond.false

cond.false:                                       ; preds = %entry
  call void @__assert_fail(i8* getelementptr inbounds ([17 x i8], [17 x i8]* @.str.1, i64 0, i64 0), i8* getelementptr inbounds ([16 x i8], [16 x i8]* @.str.2, i64 0, i64 0), i32 44, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @__PRETTY_FUNCTION__.main, i64 0, i64 0)) #8
  unreachable

cond.end:                                         ; preds = %entry
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1) #7
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %0) #7
  ret i32 0
}

; Function Attrs: nounwind
declare dso_local noalias i8* @malloc(i64) local_unnamed_addr #4

declare dso_local double @__enzyme_autodiff(i8*, ...) local_unnamed_addr #5

; Function Attrs: nounwind
declare dso_local i32 @printf(i8* nocapture readonly, ...) local_unnamed_addr #4

; Function Attrs: noreturn nounwind
declare dso_local void @__assert_fail(i8*, i8*, i32, i8*) local_unnamed_addr #6

attributes #0 = { norecurse nounwind readonly uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { noreturn nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #7 = { nounwind }
attributes #8 = { noreturn nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0}
!3 = !{!"double", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}

; CHECK: define internal {{(dso_local )?}}{ double } @differeadwriteread_helper(double* nocapture %x, double* nocapture %"x'", double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[augf:.+]] = call { { double }, double } @augmented_f_read(double* %x, double* %"x'")
; CHECK-NEXT:   %[[tapef:.+]] = extractvalue { { double }, double } %[[augf]], 0
; CHECK-NEXT:   %[[retf:.+]] = extractvalue { { double }, double } %[[augf]], 1
; CHECK-NEXT:   %[[gret:.+]] = call { double } @augmented_g_write(double* %x, double* %"x'", double %[[retf]])
; CHECK-NEXT:   %[[dhret:.+]] = call { double } @diffeh_read(double* %x, double* %"x'", double %differeturn)
; CHECK-NEXT:   %[[dg:.+]] = call { double } @diffeg_write(double* %x, double* %"x'", double %[[retf]], { double } %[[gret]])
; CHECK-NEXT:   %[[dgret:.+]] = extractvalue { double } %[[dg]], 0
; CHECK-NEXT:   call void @diffef_read(double* %x, double* %"x'", double %[[dgret]], { double } %[[tapef]])
; CHECK-NEXT:   ret { double } %[[dhret]]
; CHECK-NEXT: }

