; RUN: opt < %s %loadEnzyme -enzyme -enzyme_preopt=false -inline -mem2reg -instsimplify -adce -loop-deletion -correlated-propagation -simplifycfg -S | FileCheck %s

@main.x = private unnamed_addr constant [3 x double] [double 1.000000e+00, double 2.000000e+00, double 3.000000e+00], align 16
@main.yp = private unnamed_addr constant [1 x double] [double 1.000000e+00], align 8
@.str = private unnamed_addr constant [28 x i8] c"xp[0]=%f xp[1]=%f xp[2]=%f\0A\00", align 1
@.str.1 = private unnamed_addr constant [10 x i8] c"yp[0]=%f\0A\00", align 1
@.str.2 = private unnamed_addr constant [9 x i8] c"y[0]=%f\0A\00", align 1
@.str.3 = private unnamed_addr constant [25 x i8] c"x[0]=%f x[1]=%f x[2]=%f\0A\00", align 1

; Function Attrs: noinline nounwind uwtable
define dso_local double @f(double* nocapture readonly %x, i64 %n) #0 {
entry:
  br label %for.body

for.body:                                         ; preds = %if.end, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %if.end ]
  %data.016 = phi double [ 0.000000e+00, %entry ], [ %add5, %if.end ]
  %cmp2 = fcmp fast ogt double %data.016, 1.000000e+01
  br i1 %cmp2, label %if.then, label %if.end

if.then:                                          ; preds = %for.body
  %arrayidx = getelementptr inbounds double, double* %x, i64 %n
  %0 = load double, double* %arrayidx, align 8, !tbaa !2
  %add = fadd fast double %0, %data.016
  br label %cleanup

if.end:                                           ; preds = %for.body
  %arrayidx4 = getelementptr inbounds double, double* %x, i64 %indvars.iv
  %1 = load double, double* %arrayidx4, align 8, !tbaa !2
  %add5 = fadd fast double %1, %data.016
  %indvars.iv.next = add nuw i64 %indvars.iv, 1
  %cmp = icmp ult i64 %indvars.iv, %n
  br i1 %cmp, label %for.body, label %cleanup

cleanup:                                          ; preds = %if.end, %if.then
  %data.1 = phi double [ %add, %if.then ], [ %add5, %if.end ]
  ret double %data.1
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

; Function Attrs: noinline nounwind uwtable
define dso_local double @dsumsquare(double* %x, double* %xp, i64 %n) #0 {
entry:
  %x.addr = alloca double*, align 8
  %xp.addr = alloca double*, align 8
  %n.addr = alloca i64, align 8
  store double* %x, double** %x.addr, align 8, !tbaa !2
  store double* %xp, double** %xp.addr, align 8, !tbaa !2
  store i64 %n, i64* %n.addr, align 8, !tbaa !6
  %0 = load double*, double** %x.addr, align 8, !tbaa !2
  %1 = load double*, double** %xp.addr, align 8, !tbaa !2
  %2 = load i64, i64* %n.addr, align 8, !tbaa !6
  %call = call fast double @__enzyme_autodiff(i8* bitcast (double (double*, i64)* @f to i8*), double* %0, double* %1, i64 %2)
  ret double %call
}

declare dso_local double @__enzyme_autodiff(i8*, double*, double*, i64) #2

; Function Attrs: nounwind uwtable
define dso_local i32 @main() #3 {
entry:
  %x = alloca [3 x double], align 16
  %y = alloca [1 x double], align 8
  %xp = alloca [3 x double], align 16
  %yp = alloca [1 x double], align 8
  %0 = bitcast [3 x double]* %x to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* %0) #4
  %1 = bitcast [3 x double]* %x to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %1, i8* align 16 bitcast ([3 x double]* @main.x to i8*), i64 24, i1 false)
  %2 = bitcast [1 x double]* %y to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %2) #4
  %3 = bitcast [1 x double]* %y to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %3, i8 0, i64 8, i1 false)
  %4 = bitcast [3 x double]* %xp to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* %4) #4
  %5 = bitcast [3 x double]* %xp to i8*
  call void @llvm.memset.p0i8.i64(i8* align 16 %5, i8 0, i64 24, i1 false)
  %6 = bitcast [1 x double]* %yp to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %6) #4
  %7 = bitcast [1 x double]* %yp to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %7, i8* align 8 bitcast ([1 x double]* @main.yp to i8*), i64 8, i1 false)
  %arraydecay = getelementptr inbounds [3 x double], [3 x double]* %x, i32 0, i32 0
  %arraydecay1 = getelementptr inbounds [3 x double], [3 x double]* %xp, i32 0, i32 0
  %call = call fast double @dsumsquare(double* %arraydecay, double* %arraydecay1, i64 2)
  %arrayidx = getelementptr inbounds [3 x double], [3 x double]* %xp, i64 0, i64 0
  %8 = load double, double* %arrayidx, align 16, !tbaa !8
  %arrayidx2 = getelementptr inbounds [3 x double], [3 x double]* %xp, i64 0, i64 1
  %9 = load double, double* %arrayidx2, align 8, !tbaa !8
  %arrayidx3 = getelementptr inbounds [3 x double], [3 x double]* %xp, i64 0, i64 2
  %10 = load double, double* %arrayidx3, align 16, !tbaa !8
  %call4 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([28 x i8], [28 x i8]* @.str, i32 0, i32 0), double %8, double %9, double %10)
  %arrayidx5 = getelementptr inbounds [1 x double], [1 x double]* %yp, i64 0, i64 0
  %11 = load double, double* %arrayidx5, align 8, !tbaa !8
  %call6 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str.1, i32 0, i32 0), double %11)
  %arrayidx7 = getelementptr inbounds [1 x double], [1 x double]* %y, i64 0, i64 0
  %12 = load double, double* %arrayidx7, align 8, !tbaa !8
  %call8 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str.2, i32 0, i32 0), double %12)
  %arrayidx9 = getelementptr inbounds [3 x double], [3 x double]* %x, i64 0, i64 0
  %13 = load double, double* %arrayidx9, align 16, !tbaa !8
  %arrayidx10 = getelementptr inbounds [3 x double], [3 x double]* %x, i64 0, i64 1
  %14 = load double, double* %arrayidx10, align 8, !tbaa !8
  %arrayidx11 = getelementptr inbounds [3 x double], [3 x double]* %x, i64 0, i64 2
  %15 = load double, double* %arrayidx11, align 16, !tbaa !8
  %call12 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([25 x i8], [25 x i8]* @.str.3, i32 0, i32 0), double %13, double %14, double %15)
  %16 = bitcast [1 x double]* %yp to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %16) #4
  %17 = bitcast [3 x double]* %xp to i8*
  call void @llvm.lifetime.end.p0i8(i64 24, i8* %17) #4
  %18 = bitcast [1 x double]* %y to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %18) #4
  %19 = bitcast [3 x double]* %x to i8*
  call void @llvm.lifetime.end.p0i8(i64 24, i8* %19) #4
  ret i32 0
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1) #1

declare dso_local i32 @printf(i8*, ...) #2

attributes #0 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #3 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #4 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0}
!3 = !{!"any pointer", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
!6 = !{!7, !7, i64 0}
!7 = !{!"long long", !4, i64 0}
!8 = !{!9, !9, i64 0}
!9 = !{!"double", !4, i64 0}
!10 = !{!11, !11, i64 0}
!11 = !{!"int", !4, i64 0}
