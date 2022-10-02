; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -simplifycfg -adce -S | FileCheck %s
; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | %lli - | FileCheck %s --check-prefix=EVAL

; EVAL: reduce_max=2.000000
; EVAL: d_reduce_max(0)=0.000000
; EVAL: d_reduce_max(1)=0.500000
; EVAL: d_reduce_max(2)=0.000000
; EVAL: d_reduce_max(3)=0.500000
; EVAL: d_reduce_max(4)=0.000000

source_filename = "multivecmax.cpp"

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, %struct._IO_codecvt*, %struct._IO_wide_data*, %struct._IO_FILE*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type opaque
%struct._IO_codecvt = type opaque
%struct._IO_wide_data = type opaque

@.str = private unnamed_addr constant [15 x i8] c"reduce_max=%f\0A\00", align 1
@.str.1 = private unnamed_addr constant [21 x i8] c"d_reduce_max(%i)=%f\0A\00", align 1
@.str.2 = private unnamed_addr constant [68 x i8] c"Assertion Failed: fabs( [%s = %g] - [%s = %g] ) > %g at %s:%d (%s)\0A\00", align 1
@.str.3 = private unnamed_addr constant [9 x i8] c"d_vec[i]\00", align 1
@.str.4 = private unnamed_addr constant [7 x i8] c"ans[i]\00", align 1
@.str.5 = private unnamed_addr constant [16 x i8] c"multivecmax.cpp\00", align 1
@__PRETTY_FUNCTION__.main = private unnamed_addr constant [11 x i8] c"int main()\00", align 1

; Function Attrs: nounwind uwtable
define dso_local double @_Z10reduce_maxPdi(double* %vec, i32 %size) #0 {
entry:
  %arrayidx = getelementptr inbounds double, double* %vec, i64 1
  %_M_start.i.i = alloca double*, align 8
  %call5.i.i.i.i.i = call i8* @malloc(i64 8) #9
  %tmp1 = bitcast double* %arrayidx to i64*
  %tmp2 = load i64, i64* %tmp1, align 8, !tbaa !2
  %tmp3 = bitcast i8* %call5.i.i.i.i.i to i64*
  store i64 %tmp2, i64* %tmp3, align 8, !tbaa !2
  %tmp4 = bitcast double** %_M_start.i.i to i8**
  store i8* %call5.i.i.i.i.i, i8** %tmp4, align 8, !tbaa !6
  %arrayidx1 = getelementptr inbounds double, double* %vec, i64 3
  %rtmp2 = load double, double* %arrayidx1, align 8, !tbaa !2
  call void @_ZNSt6vectorIdSaIdEE9push_backERKd2(double** %_M_start.i.i, double %rtmp2)
  %tmp7 = load double*, double** %_M_start.i.i, align 8, !tbaa !6
  %tmp8 = load double, double* %tmp7, align 8, !tbaa !2
  %add.ptr.i = getelementptr inbounds double, double* %tmp7, i64 1
  %tmp9 = load double, double* %add.ptr.i, align 8, !tbaa !2
  %add = fadd fast double %tmp9, %tmp8
  %div = fmul fast double %add, 5.000000e-01
  %tmp10 = bitcast double* %tmp7 to i8*
  call void @free(i8* nonnull %tmp10) #9
  ret double %div
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

define linkonce_odr dso_local void @_ZNSt6vectorIdSaIdEE9push_backERKd2(double** %_M_start.i, double %tmp2) {
entry:
  %tmp = load double*, double** %_M_start.i, align 8, !tbaa !9
  %call5.i.i.i.i = call i8* @malloc(i64 16) #9
  %add.ptr.i = getelementptr inbounds i8, i8* %call5.i.i.i.i, i64 8
  %tmp3 = bitcast i8* %add.ptr.i to double*
  store double %tmp2, double* %tmp3, align 8, !tbaa !2
  %tmp4 = bitcast double* %tmp to i64*
  %tmp5 = bitcast i8* %call5.i.i.i.i to i64*
  %tmp6 = load i64, i64* %tmp4, align 8
  store i64 %tmp6, i64* %tmp5, align 8
  %tmp7 = bitcast double* %tmp to i8*
  call void @free(i8* nonnull %tmp7) #9
  %tmp8 = bitcast double** %_M_start.i to i8**
  store i8* %call5.i.i.i.i, i8** %tmp8, align 8, !tbaa !6
  ret void
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

; Function Attrs: norecurse nounwind uwtable
define dso_local i32 @main() local_unnamed_addr #2 {
entry:
  %vec = alloca [5 x double], align 16
  %d_vec = alloca [5 x double], align 16
  %tmp = bitcast [5 x double]* %vec to i8*
  call void @llvm.lifetime.start.p0i8(i64 40, i8* nonnull %tmp) #9
  call void @llvm.memset.p0i8.i64(i8* nonnull align 16 dereferenceable(40) %tmp, i8 0, i64 40, i1 false)
  %tmp1 = getelementptr inbounds [5 x double], [5 x double]* %vec, i64 0, i64 0
  store double -1.000000e+00, double* %tmp1, align 16
  %tmp2 = getelementptr inbounds [5 x double], [5 x double]* %vec, i64 0, i64 1
  store double 2.000000e+00, double* %tmp2, align 8
  %tmp3 = getelementptr inbounds [5 x double], [5 x double]* %vec, i64 0, i64 2
  store double -2.000000e-01, double* %tmp3, align 16
  %tmp4 = getelementptr inbounds [5 x double], [5 x double]* %vec, i64 0, i64 3
  store double 2.000000e+00, double* %tmp4, align 8
  %tmp5 = getelementptr inbounds [5 x double], [5 x double]* %vec, i64 0, i64 4
  store double 1.000000e+00, double* %tmp5, align 16
  %tmp6 = bitcast [5 x double]* %d_vec to i8*
  call void @llvm.lifetime.start.p0i8(i64 40, i8* nonnull %tmp6) #9
  call void @llvm.memset.p0i8.i64(i8* nonnull align 16 dereferenceable(40) %tmp6, i8 0, i64 40, i1 false)
  %call = call fast double @_Z10reduce_maxPdi(double* nonnull %tmp1, i32 undef)
  %call1 = call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([15 x i8], [15 x i8]* @.str, i64 0, i64 0), double %call)
  %arraydecay3 = getelementptr inbounds [5 x double], [5 x double]* %d_vec, i64 0, i64 0
  call void @_Z17__enzyme_autodiffPvPdS0_i(i8* bitcast (double (double*, i32)* @_Z10reduce_maxPdi to i8*), double* nonnull %tmp1, double* nonnull %arraydecay3, i32 5) #9
  %tmp7 = load double, double* %arraydecay3, align 16, !tbaa !2
  %call4 = call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([21 x i8], [21 x i8]* @.str.1, i64 0, i64 0), i32 0, double %tmp7)
  %arrayidx.1 = getelementptr inbounds [5 x double], [5 x double]* %d_vec, i64 0, i64 1
  %tmp8 = load double, double* %arrayidx.1, align 8, !tbaa !2
  %call4.1 = call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([21 x i8], [21 x i8]* @.str.1, i64 0, i64 0), i32 1, double %tmp8)
  %arrayidx.2 = getelementptr inbounds [5 x double], [5 x double]* %d_vec, i64 0, i64 2
  %tmp9 = load double, double* %arrayidx.2, align 16, !tbaa !2
  %call4.2 = call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([21 x i8], [21 x i8]* @.str.1, i64 0, i64 0), i32 2, double %tmp9)
  %arrayidx.3 = getelementptr inbounds [5 x double], [5 x double]* %d_vec, i64 0, i64 3
  %tmp10 = load double, double* %arrayidx.3, align 8, !tbaa !2
  %call4.3 = call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([21 x i8], [21 x i8]* @.str.1, i64 0, i64 0), i32 3, double %tmp10)
  %arrayidx.4 = getelementptr inbounds [5 x double], [5 x double]* %d_vec, i64 0, i64 4
  %tmp11 = load double, double* %arrayidx.4, align 16, !tbaa !2
  %call4.4 = call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([21 x i8], [21 x i8]* @.str.1, i64 0, i64 0), i32 4, double %tmp11)
  %call5 = call i32 @fflush(%struct._IO_FILE* null)
  %tmp12 = load double, double* %arraydecay3, align 16, !tbaa !2
  %tmp13 = call fast double @llvm.fabs.f64(double %tmp12)
  %cmp15 = fcmp fast ogt double %tmp13, 1.000000e-10
  br i1 %cmp15, label %if.then, label %for.cond7

for.cond7:                                        ; preds = %entry
  %tmp14 = load double, double* %arrayidx.1, align 8, !tbaa !2
  %sub.1 = fadd fast double %tmp14, -5.000000e-01
  %tmp15 = call fast double @llvm.fabs.f64(double %sub.1)
  %cmp15.1 = fcmp fast ogt double %tmp15, 1.000000e-10
  br i1 %cmp15.1, label %if.then, label %for.cond7.1

if.then:                                          ; preds = %for.cond7.3, %for.cond7.2, %for.cond7.1, %for.cond7, %entry
  %.lcssa5 = phi double [ %tmp12, %entry ], [ %tmp14, %for.cond7 ], [ %tmp17, %for.cond7.1 ], [ %tmp19, %for.cond7.2 ], [ %tmp21, %for.cond7.3 ]
  %.lcssa = phi double [ 0.000000e+00, %entry ], [ 5.000000e-01, %for.cond7 ], [ 0.000000e+00, %for.cond7.1 ], [ 5.000000e-01, %for.cond7.2 ], [ 0.000000e+00, %for.cond7.3 ]
  %call20 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([68 x i8], [68 x i8]* @.str.2, i64 0, i64 0), i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str.3, i64 0, i64 0), double %.lcssa5, i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.4, i64 0, i64 0), double %.lcssa, double 1.000000e-10, i8* getelementptr inbounds ([16 x i8], [16 x i8]* @.str.5, i64 0, i64 0), i32 48, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @__PRETTY_FUNCTION__.main, i64 0, i64 0)) #10
  call void @abort() #11
  unreachable

for.cond7.1:                                      ; preds = %for.cond7
  %tmp17 = load double, double* %arrayidx.2, align 16, !tbaa !2
  %tmp18 = call fast double @llvm.fabs.f64(double %tmp17)
  %cmp15.2 = fcmp fast ogt double %tmp18, 1.000000e-10
  br i1 %cmp15.2, label %if.then, label %for.cond7.2

for.cond7.2:                                      ; preds = %for.cond7.1
  %tmp19 = load double, double* %arrayidx.3, align 8, !tbaa !2
  %sub.3 = fadd fast double %tmp19, -5.000000e-01
  %tmp20 = call fast double @llvm.fabs.f64(double %sub.3)
  %cmp15.3 = fcmp fast ogt double %tmp20, 1.000000e-10
  br i1 %cmp15.3, label %if.then, label %for.cond7.3

for.cond7.3:                                      ; preds = %for.cond7.2
  %tmp21 = load double, double* %arrayidx.4, align 16, !tbaa !2
  %tmp22 = call fast double @llvm.fabs.f64(double %tmp21)
  %cmp15.4 = fcmp fast ogt double %tmp22, 1.000000e-10
  br i1 %cmp15.4, label %if.then, label %for.cond7.4

for.cond7.4:                                      ; preds = %for.cond7.3
  call void @llvm.lifetime.end.p0i8(i64 40, i8* nonnull %tmp6) #9
  call void @llvm.lifetime.end.p0i8(i64 40, i8* nonnull %tmp) #9
  ret i32 0
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1) #1

; Function Attrs: nofree nounwind
declare dso_local i32 @printf(i8* nocapture readonly, ...) local_unnamed_addr #3

declare dso_local void @_Z17__enzyme_autodiffPvPdS0_i(i8*, double*, double*, i32) local_unnamed_addr #4

; Function Attrs: nofree nounwind
declare dso_local i32 @fflush(%struct._IO_FILE* nocapture) local_unnamed_addr #3

; Function Attrs: nounwind readnone speculatable willreturn
declare double @llvm.fabs.f64(double) #5

; Function Attrs: nofree nounwind
declare dso_local i32 @fprintf(%struct._IO_FILE* nocapture, i8* nocapture readonly, ...) local_unnamed_addr #3

; Function Attrs: noreturn nounwind
declare dso_local void @abort() local_unnamed_addr #6

; Function Attrs: nobuiltin nounwind
declare dso_local void @free(i8*) local_unnamed_addr #7

; Function Attrs: nobuiltin nofree
declare dso_local noalias nonnull i8* @malloc(i64) local_unnamed_addr #8

attributes #0 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="true" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #3 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #4 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #5 = { nounwind readnone }
attributes #6 = { noreturn nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #7 = { nobuiltin nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #8 = { nobuiltin "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #9 = { nounwind }
attributes #10 = { cold }
attributes #11 = { noreturn nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 10.0.1 (git@github.com:llvm/llvm-project ef32c611aa214dea855364efd7ba451ec5ec3f74)"}
!2 = !{!3, !3, i64 0}
!3 = !{!"double", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C++ TBAA"}
!6 = !{!7, !8, i64 0}
!7 = !{!"_ZTSNSt12_Vector_baseIdSaIdEE17_Vector_impl_dataE", !8, i64 0, !8, i64 8, !8, i64 16}
!8 = !{!"any pointer", !4, i64 0}
!9 = !{!8, !8, i64 0}

; CHECK: define internal void @diffe_Z10reduce_maxPdi(double* %vec, double* %"vec'", i32 %size, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"arrayidx'ipg" = getelementptr inbounds double, double* %"vec'", i64 1
; CHECK-NEXT:   %arrayidx = getelementptr inbounds double, double* %vec, i64 1
; CHECK-NEXT:   %"_M_start.i.i'ipa" = alloca double*, align 8
; CHECK-NEXT:   store double* null, double** %"_M_start.i.i'ipa", align 8
; CHECK-NEXT:   %_M_start.i.i = alloca double*, align 8
; CHECK-NEXT:   %call5.i.i.i.i.i = call noalias nonnull dereferenceable(8) dereferenceable_or_null(8) i8* @malloc(i64 8)
; CHECK-NEXT:   %"call5.i.i.i.i.i'mi" = call noalias nonnull dereferenceable(8) dereferenceable_or_null(8) i8* @malloc(i64 8)
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull dereferenceable(8) dereferenceable_or_null(8) %"call5.i.i.i.i.i'mi", i8 0, i64 8, i1 false)
; CHECK-NEXT:   %"tmp1'ipc" = bitcast double* %"arrayidx'ipg" to i64*
; CHECK-NEXT:   %tmp1 = bitcast double* %arrayidx to i64*
; CHECK-NEXT:   %tmp2 = load i64, i64* %tmp1, align 8, !tbaa !2
; CHECK-NEXT:   %"tmp3'ipc" = bitcast i8* %"call5.i.i.i.i.i'mi" to i64*
; CHECK-NEXT:   %tmp3 = bitcast i8* %call5.i.i.i.i.i to i64*
; CHECK-NEXT:   store i64 %tmp2, i64* %tmp3, align 8, !tbaa !2
; CHECK-NEXT:   %"tmp4'ipc" = bitcast double** %"_M_start.i.i'ipa" to i8**
; CHECK-NEXT:   %tmp4 = bitcast double** %_M_start.i.i to i8**
; CHECK-NEXT:   store i8* %"call5.i.i.i.i.i'mi", i8** %"tmp4'ipc", align 8, !tbaa !6
; CHECK-NEXT:   store i8* %call5.i.i.i.i.i, i8** %tmp4, align 8, !tbaa !6
; CHECK-NEXT:   %"arrayidx1'ipg" = getelementptr inbounds double, double* %"vec'", i64 3
; CHECK-NEXT:   %arrayidx1 = getelementptr inbounds double, double* %vec, i64 3
; CHECK-NEXT:   %rtmp2 = load double, double* %arrayidx1, align 8, !tbaa !2
; CHECK-NEXT:   %_augmented = call { i8*, i8*, double* } @augmented__ZNSt6vectorIdSaIdEE9push_backERKd2(double** %_M_start.i.i, double** %"_M_start.i.i'ipa", double %rtmp2)
; CHECK-NEXT:   %"tmp7'ipl" = load double*, double** %"_M_start.i.i'ipa", align 8, !tbaa !6
; CHECK-NEXT:   %"add.ptr.i'ipg" = getelementptr inbounds double, double* %"tmp7'ipl", i64 1
; CHECK-NEXT:   %m0diffeadd = fmul fast double %differeturn, 5.000000e-01
; CHECK-NEXT:   %0 = fadd fast double 0.000000e+00, %m0diffeadd
; CHECK-NEXT:   %1 = fadd fast double 0.000000e+00, %0
; CHECK-NEXT:   %2 = fadd fast double 0.000000e+00, %0
; CHECK-NEXT:   %3 = load double, double* %"add.ptr.i'ipg", align 8, !tbaa !2
; CHECK-NEXT:   %4 = fadd fast double %3, %1
; CHECK-NEXT:   store double %4, double* %"add.ptr.i'ipg", align 8, !tbaa !2
; CHECK-NEXT:   %5 = load double, double* %"tmp7'ipl", align 8, !tbaa !2
; CHECK-NEXT:   %6 = fadd fast double %5, %2
; CHECK-NEXT:   store double %6, double* %"tmp7'ipl", align 8, !tbaa !2
; CHECK-NEXT:   %7 = call { double } @diffe_ZNSt6vectorIdSaIdEE9push_backERKd2(double** %_M_start.i.i, double** %"_M_start.i.i'ipa", double %rtmp2, { i8*, i8*, double* } %_augmented)
; CHECK-NEXT:   %8 = extractvalue { double } %7, 0
; CHECK-NEXT:   %9 = fadd fast double 0.000000e+00, %8
; CHECK-NEXT:   %10 = load double, double* %"arrayidx1'ipg", align 8, !tbaa !2
; CHECK-NEXT:   %11 = fadd fast double %10, %9
; CHECK-NEXT:   store double %11, double* %"arrayidx1'ipg", align 8, !tbaa !2
; CHECK-NEXT:   %12 = load i64, i64* %"tmp3'ipc", align 8
; CHECK-NEXT:   store i64 0, i64* %"tmp3'ipc", align 8, !tbaa !2
; CHECK-NEXT:   %13 = bitcast i64 0 to double
; CHECK-NEXT:   %14 = bitcast i64 %12 to double
; CHECK-NEXT:   %15 = fadd fast double %13, %14
; CHECK-NEXT:   %16 = bitcast double %15 to i64
; CHECK-NEXT:   %17 = bitcast i64* %"tmp1'ipc" to double*
; CHECK-NEXT:   %18 = bitcast i64 %16 to double
; CHECK-NEXT:   %19 = load double, double* %17, align 8, !tbaa !2
; CHECK-NEXT:   %20 = fadd fast double %19, %18
; CHECK-NEXT:   store double %20, double* %17, align 8, !tbaa !2
; CHECK-NEXT:   tail call void @free(i8* nonnull %"call5.i.i.i.i.i'mi")
; CHECK-NEXT:   tail call void @free(i8* nonnull %call5.i.i.i.i.i)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
