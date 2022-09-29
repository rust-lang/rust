; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S -gvn | FileCheck %s

; ModuleID = 'bout.ll'
source_filename = "/home/runner/work/Enzyme/Enzyme/enzyme/test/Integration/multivecmax.cpp"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, %struct._IO_codecvt*, %struct._IO_wide_data*, %struct._IO_FILE*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type opaque
%struct._IO_codecvt = type opaque
%struct._IO_wide_data = type opaque

@.str = private unnamed_addr constant [15 x i8] c"reduce_max=%f\0A\00", align 1
@.str.1 = private unnamed_addr constant [21 x i8] c"d_reduce_max(%i)=%f\0A\00", align 1
@stderr = external dso_local local_unnamed_addr global %struct._IO_FILE*, align 8
@.str.2 = private unnamed_addr constant [68 x i8] c"Assertion Failed: fabs( [%s = %g] - [%s = %g] ) > %g at %s:%d (%s)\0A\00", align 1
@.str.3 = private unnamed_addr constant [9 x i8] c"d_vec[i]\00", align 1
@.str.4 = private unnamed_addr constant [7 x i8] c"ans[i]\00", align 1
@.str.5 = private unnamed_addr constant [72 x i8] c"/home/runner/work/Enzyme/Enzyme/enzyme/test/Integration/multivecmax.cpp\00", align 1
@__PRETTY_FUNCTION__.main = private unnamed_addr constant [11 x i8] c"int main()\00", align 1
@.str.6 = private unnamed_addr constant [26 x i8] c"vector::_M_realloc_insert\00", align 1


; Function Attrs: nounwind uwtable
define internal double* @sub(double** %gep0, i64 %size, double %arg2) #0 {
bb:
  %ptr0 = load double*, double** %gep0, align 8, !tbaa !8
  %i80 = bitcast double* %ptr0 to i8*
  %ptrsize = shl i64 %size, 3
  %nptrsize = add i64 %ptrsize, 8
  
  %alloc = tail call i8* @malloc(i64 %nptrsize) #8
  %nptr = bitcast i8* %alloc to double*
  %insertptr = getelementptr inbounds double, double* %nptr, i64 %size
  store double %arg2, double* %insertptr, align 8, !tbaa !2
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 %alloc, i8* align 8 %i80, i64 %ptrsize, i1 false), !tbaa !2
  ret double* %nptr
}

; Function Attrs: nounwind uwtable
define dso_local double @_Z10reduce_maxPdi(double* %arg, i32 %arg1) #0 {
bb:
  %tmp3 = alloca double*

  %tmp12 = getelementptr inbounds double, double* %arg, i64 1
  %tmp13 = load double, double* %tmp12, align 8, !tbaa !2


  %alloc = tail call i8* @malloc(i64 8)
  %nptr = bitcast i8* %alloc to double*
  %insertptr = getelementptr inbounds double, double* %nptr, i64 0
  store double %tmp13, double* %insertptr, align 8, !tbaa !2

  store double* %nptr, double** %tmp3, align 8, !tbaa !8

  %tmp122 = getelementptr inbounds double, double* %arg, i64 3
  %tmp132 = load double, double* %tmp122, align 8, !tbaa !2
  %nptr2 = call double* @sub(double** %tmp3, i64 1, double %tmp132) #8

  ; if removed this errs
  store double* %nptr2, double** %tmp3, align 8, !tbaa !8

  
  %l1 = load double, double* %nptr2, align 8, !tbaa !2
  %gep1 = getelementptr inbounds double, double* %nptr2, i64 1
  %l2 = load double, double* %gep1, align 8, !tbaa !2
  %toret = fadd fast double %l1, %l2
  ret double %toret
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

; Function Attrs: norecurse nounwind uwtable
define dso_local i32 @main() local_unnamed_addr #2 {
bb:
  %tmp1 = alloca [5 x double], align 16
  %tmp2 = alloca [5 x double], align 16
  %tmp3 = alloca [5 x double], align 16
  %tmp4 = bitcast [5 x double]* %tmp1 to i8*
  call void @llvm.lifetime.start.p0i8(i64 40, i8* nonnull %tmp4) #8
  %tmp5 = getelementptr inbounds [5 x double], [5 x double]* %tmp1, i64 0, i64 0
  store double -1.000000e+00, double* %tmp5, align 16
  %tmp6 = getelementptr inbounds [5 x double], [5 x double]* %tmp1, i64 0, i64 1
  store double 2.000000e+00, double* %tmp6, align 8
  %tmp7 = getelementptr inbounds [5 x double], [5 x double]* %tmp1, i64 0, i64 2
  store double -2.000000e-01, double* %tmp7, align 16
  %tmp8 = getelementptr inbounds [5 x double], [5 x double]* %tmp1, i64 0, i64 3
  store double 2.000000e+00, double* %tmp8, align 8
  %tmp9 = getelementptr inbounds [5 x double], [5 x double]* %tmp1, i64 0, i64 4
  store double 1.000000e+00, double* %tmp9, align 16
  %tmp10 = bitcast [5 x double]* %tmp2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 40, i8* nonnull %tmp10) #8
  call void @llvm.memset.p0i8.i64(i8* nonnull align 16 %tmp10, i8 0, i64 40, i1 false)
  br label %bb21

bb15:                                             ; preds = %bb42
  %0 = inttoptr i64 %tmp.sroa.0.1 to double*
  %1 = inttoptr i64 %tmp.sroa.7.2 to double*
  %tmp20 = icmp eq double* %0, %1
  br i1 %tmp20, label %bb48, label %bb57

bb21:                                             ; preds = %bb45, %bb
  %tmp.sroa.14.0 = phi double* [ null, %bb ], [ %tmp.sroa.14.1, %bb45 ]
  %tmp.sroa.7.0 = phi i64 [ 0, %bb ], [ %tmp.sroa.7.2, %bb45 ]
  %tmp.sroa.0.0 = phi i64 [ 0, %bb ], [ %tmp.sroa.0.1, %bb45 ]
  %tmp22 = phi double [ -1.000000e+00, %bb ], [ %tmp47, %bb45 ]
  %tmp23 = phi i64 [ 0, %bb ], [ %tmp43, %bb45 ]
  %tmp24 = phi double [ 0xFFF0000000000000, %bb ], [ %tmp33, %bb45 ]
  %tmp25 = getelementptr inbounds [5 x double], [5 x double]* %tmp1, i64 0, i64 %tmp23
  %tmp26 = fcmp fast ogt double %tmp22, %tmp24
  %2 = inttoptr i64 %tmp.sroa.0.0 to double*
  %3 = inttoptr i64 %tmp.sroa.7.0 to double*
  %tmp30 = icmp eq double* %3, %2
  %spec.select = select i1 %tmp30, i64 %tmp.sroa.7.0, i64 %tmp.sroa.0.0
  %tmp.sroa.7.1 = select i1 %tmp26, i64 %spec.select, i64 %tmp.sroa.7.0
  %tmp33 = select i1 %tmp26, double %tmp22, double %tmp24
  %tmp34 = fcmp fast oeq double %tmp22, %tmp33
  br i1 %tmp34, label %bb35, label %bb42

bb35:                                             ; preds = %bb21
  %4 = inttoptr i64 %tmp.sroa.7.1 to double*
  %tmp38 = icmp eq double* %4, %tmp.sroa.14.0
  br i1 %tmp38, label %bb41, label %bb39

bb39:                                             ; preds = %bb35
  store double %tmp22, double* %4, align 8, !tbaa !2
  %tmp40 = getelementptr inbounds double, double* %4, i64 1
  %5 = ptrtoint double* %tmp40 to i64
  br label %bb42

bb41:                                             ; preds = %bb35
  %tmp8.i = sub i64 %tmp.sroa.7.1, %tmp.sroa.0.0
  %tmp9.i = ashr exact i64 %tmp8.i, 3
  %tmp10.i = add nsw i64 %tmp9.i, 1
  %tmp11.i = sub i64 %tmp.sroa.7.1, %tmp.sroa.0.0
  %tmp12.i = ashr exact i64 %tmp11.i, 3
  %tmp13.i = shl i64 %tmp10.i, 3
  %tmp14.i = call i8* @malloc(i64 %tmp13.i) #8
  %tmp15.i = bitcast i8* %tmp14.i to double*
  %tmp16.i = getelementptr inbounds double, double* %tmp15.i, i64 %tmp12.i
  %tmp17.i = bitcast double* %tmp25 to i64*
  %tmp18.i = load i64, i64* %tmp17.i, align 8, !tbaa !2
  %tmp19.i = bitcast double* %tmp16.i to i64*
  store i64 %tmp18.i, i64* %tmp19.i, align 8, !tbaa !2
  %tmp20.i = inttoptr i64 %tmp.sroa.0.0 to i8*
  call void @llvm.memmove.p0i8.p0i8.i64(i8* nonnull align 8 %tmp14.i, i8* align 8 %tmp20.i, i64 %tmp11.i, i1 false) #8, !tbaa !2
  %tmp21.i = getelementptr inbounds double, double* %tmp16.i, i64 1
  %tmp24.i = bitcast double* %tmp21.i to i8*
  %tmp25.i = bitcast double* %4 to i8*
  call void @llvm.memmove.p0i8.p0i8.i64(i8* nonnull align 8 %tmp24.i, i8* align 8 %tmp25.i, i64 0, i1 false) #8, !tbaa !2
  %6 = ptrtoint i8* %tmp14.i to i64
  %7 = ptrtoint double* %tmp21.i to i64
  %tmp29.i = getelementptr inbounds double, double* %tmp15.i, i64 %tmp10.i
  br label %bb42

bb42:                                             ; preds = %bb41, %bb39, %bb21
  %tmp.sroa.14.1 = phi double* [ %tmp29.i, %bb41 ], [ %tmp.sroa.14.0, %bb39 ], [ %tmp.sroa.14.0, %bb21 ]
  %tmp.sroa.7.2 = phi i64 [ %7, %bb41 ], [ %5, %bb39 ], [ %tmp.sroa.7.1, %bb21 ]
  %tmp.sroa.0.1 = phi i64 [ %6, %bb41 ], [ %tmp.sroa.0.0, %bb39 ], [ %tmp.sroa.0.0, %bb21 ]
  %tmp43 = add nuw nsw i64 %tmp23, 1
  %tmp44 = icmp eq i64 %tmp43, 5
  br i1 %tmp44, label %bb15, label %bb45

bb45:                                             ; preds = %bb42
  %tmp46 = getelementptr inbounds [5 x double], [5 x double]* %tmp1, i64 0, i64 %tmp43
  %tmp47 = load double, double* %tmp46, align 8, !tbaa !2
  br label %bb21

bb48:                                             ; preds = %bb57, %bb15
  %tmp49 = phi double [ 0.000000e+00, %bb15 ], [ %tmp61, %bb57 ]
  %tmp50 = sub i64 %tmp.sroa.7.2, %tmp.sroa.0.1
  %tmp51 = ashr exact i64 %tmp50, 3
  %tmp52 = uitofp i64 %tmp51 to double
  %tmp53 = fdiv fast double %tmp49, %tmp52
  %tmp54 = icmp eq double* %0, null
  br i1 %tmp54, label %bb64, label %bb55

bb55:                                             ; preds = %bb48
  %tmp56 = bitcast double* %0 to i8*
  call void @_ZdlPv(i8* %tmp56) #8
  br label %bb64

bb57:                                             ; preds = %bb57, %bb15
  %tmp58 = phi double [ %tmp61, %bb57 ], [ 0.000000e+00, %bb15 ]
  %tmp59 = phi double* [ %tmp62, %bb57 ], [ %0, %bb15 ]
  %tmp60 = load double, double* %tmp59, align 8, !tbaa !2
  %tmp61 = fadd fast double %tmp60, %tmp58
  %tmp62 = getelementptr inbounds double, double* %tmp59, i64 1
  %tmp63 = icmp eq double* %tmp62, %1
  br i1 %tmp63, label %bb48, label %bb57

bb64:                                             ; preds = %bb55, %bb48
  %tmp65 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([15 x i8], [15 x i8]* @.str, i64 0, i64 0), double %tmp53)
  %tmp66 = getelementptr inbounds [5 x double], [5 x double]* %tmp2, i64 0, i64 0
  call void @_Z17__enzyme_autodiffPvPdS0_i(i8* bitcast (double (double*, i32)* @_Z10reduce_maxPdi to i8*), double* nonnull %tmp5, double* nonnull %tmp66, i32 5) #8
  br label %bb72

bb67:                                             ; preds = %bb72
  %tmp68 = call i32 @fflush(%struct._IO_FILE* null)
  %tmp69 = bitcast [5 x double]* %tmp3 to i8*
  call void @llvm.lifetime.start.p0i8(i64 40, i8* nonnull %tmp69) #8
  call void @llvm.memset.p0i8.i64(i8* nonnull align 16 %tmp69, i8 0, i64 40, i1 false)
  %tmp70 = getelementptr inbounds [5 x double], [5 x double]* %tmp3, i64 0, i64 1
  store double 1.000000e+00, double* %tmp70, align 8
  %tmp71 = getelementptr inbounds [5 x double], [5 x double]* %tmp3, i64 0, i64 3
  store double 1.000000e+00, double* %tmp71, align 8
  br label %bb83

bb72:                                             ; preds = %bb72, %bb64
  %tmp73 = phi i64 [ 0, %bb64 ], [ %tmp78, %bb72 ]
  %tmp74 = getelementptr inbounds [5 x double], [5 x double]* %tmp2, i64 0, i64 %tmp73
  %tmp75 = load double, double* %tmp74, align 8, !tbaa !2
  %tmp76 = trunc i64 %tmp73 to i32
  %tmp77 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([21 x i8], [21 x i8]* @.str.1, i64 0, i64 0), i32 %tmp76, double %tmp75)
  %tmp78 = add nuw nsw i64 %tmp73, 1
  %tmp79 = icmp eq i64 %tmp78, 5
  br i1 %tmp79, label %bb67, label %bb72

bb80:                                             ; preds = %bb83
  %tmp81 = icmp ult i64 %tmp92, 5
  br i1 %tmp81, label %bb83, label %bb82

bb82:                                             ; preds = %bb80
  call void @llvm.lifetime.end.p0i8(i64 40, i8* nonnull %tmp69) #8
  call void @llvm.lifetime.end.p0i8(i64 40, i8* nonnull %tmp10) #8
  call void @llvm.lifetime.end.p0i8(i64 40, i8* nonnull %tmp4) #8
  ret i32 0

bb83:                                             ; preds = %bb80, %bb67
  %tmp84 = phi i64 [ 0, %bb67 ], [ %tmp92, %bb80 ]
  %tmp85 = getelementptr inbounds [5 x double], [5 x double]* %tmp2, i64 0, i64 %tmp84
  %tmp86 = load double, double* %tmp85, align 8, !tbaa !2
  %tmp87 = getelementptr inbounds [5 x double], [5 x double]* %tmp3, i64 0, i64 %tmp84
  %tmp88 = load double, double* %tmp87, align 8, !tbaa !2
  %tmp89 = fsub fast double %tmp86, %tmp88
  %tmp90 = call fast double @llvm.fabs.f64(double %tmp89)
  %tmp91 = fcmp fast ogt double %tmp90, 1.000000e-10
  %tmp92 = add nuw nsw i64 %tmp84, 1
  br i1 %tmp91, label %bb93, label %bb80

bb93:                                             ; preds = %bb83
  %tmp94 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !6
  %tmp95 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %tmp94, i8* getelementptr inbounds ([68 x i8], [68 x i8]* @.str.2, i64 0, i64 0), i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str.3, i64 0, i64 0), double %tmp86, i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.4, i64 0, i64 0), double %tmp88, double 1.000000e-10, i8* getelementptr inbounds ([72 x i8], [72 x i8]* @.str.5, i64 0, i64 0), i32 59, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @__PRETTY_FUNCTION__.main, i64 0, i64 0)) #9
  call void @abort() #10
  unreachable
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1) #1

; Function Attrs: nounwind
declare dso_local i32 @printf(i8* nocapture readonly, ...) local_unnamed_addr #3

declare dso_local void @_Z17__enzyme_autodiffPvPdS0_i(i8*, double*, double*, i32) local_unnamed_addr #4

; Function Attrs: nounwind
declare dso_local i32 @fflush(%struct._IO_FILE* nocapture) local_unnamed_addr #3

; Function Attrs: nounwind readnone speculatable
declare double @llvm.fabs.f64(double) #5

; Function Attrs: nounwind
declare dso_local i32 @fprintf(%struct._IO_FILE* nocapture, i8* nocapture readonly, ...) local_unnamed_addr #3

; Function Attrs: noreturn nounwind
declare dso_local void @abort() local_unnamed_addr #6

; Function Attrs: nobuiltin nounwind
declare dso_local void @_ZdlPv(i8*) local_unnamed_addr #7

declare noalias nonnull i8* @malloc(i64)

; Function Attrs: argmemonly nounwind
declare void @llvm.memmove.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i1) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1) #1

attributes #0 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #3 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #4 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #5 = { nounwind readnone speculatable }
attributes #6 = { noreturn nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #7 = { nobuiltin nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #8 = { nounwind }
attributes #9 = { cold }
attributes #10 = { noreturn nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.0.1-12 (tags/RELEASE_701/final)"}
!2 = !{!3, !3, i64 0}
!3 = !{!"double", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C++ TBAA"}
!6 = !{!7, !7, i64 0}
!7 = !{!"any pointer", !4, i64 0}
!8 = !{!9, !7, i64 0}
!9 = !{!"_ZTSNSt12_Vector_baseIdSaIdEE17_Vector_impl_dataE", !7, i64 0, !7, i64 8, !7, i64 16}
!10 = !{!9, !7, i64 8}

; CHECK: define internal { double } @diffesub(double** %gep0, double** %"gep0'", i64 %size, double %arg2, { i8*, i8*, double* } %tapeArg)
; CHECK-NEXT: bb:
; CHECK-NEXT:   %"ptr0'il_phi" = extractvalue { i8*, i8*, double* } %tapeArg, 2
; CHECK-NEXT:   %ptrsize = shl i64 %size, 3
; CHECK-NEXT:   %alloc = extractvalue { i8*, i8*, double* } %tapeArg, 1
; CHECK-NEXT:   %"alloc'mi" = extractvalue { i8*, i8*, double* } %tapeArg, 0
; CHECK-NEXT:   %"nptr'ipc" = bitcast i8* %"alloc'mi" to double*
; CHECK-NEXT:   %"insertptr'ipg" = getelementptr inbounds double, double* %"nptr'ipc", i64 %size
; CHECK-NEXT:   %0 = udiv i64 %ptrsize, 8
; CHECK-NEXT:   %1 = icmp eq i64 %0, 0
; CHECK-NEXT:   br i1 %1, label %__enzyme_memcpyadd_doubleda8sa8.exit, label %for.body.i

; CHECK: for.body.i:                                       ; preds = %for.body.i, %bb
; CHECK-NEXT:   %idx.i = phi i64 [ 0, %bb ], [ %idx.next.i, %for.body.i ]
; CHECK-NEXT:   %dst.i.i = getelementptr inbounds double, double* %"nptr'ipc", i64 %idx.i
; CHECK-NEXT:   %dst.i.l.i = load double, double* %dst.i.i
; CHECK-NEXT:   store double 0.000000e+00, double* %dst.i.i
; CHECK-NEXT:   %src.i.i = getelementptr inbounds double, double* %"ptr0'il_phi", i64 %idx.i
; CHECK-NEXT:   %src.i.l.i = load double, double* %src.i.i
; CHECK-NEXT:   %2 = fadd fast double %src.i.l.i, %dst.i.l.i
; CHECK-NEXT:   store double %2, double* %src.i.i
; CHECK-NEXT:   %idx.next.i = add nuw i64 %idx.i, 1
; CHECK-NEXT:   %3 = icmp eq i64 %0, %idx.next.i
; CHECK-NEXT:   br i1 %3, label %__enzyme_memcpyadd_doubleda8sa8.exit, label %for.body.i

; CHECK: __enzyme_memcpyadd_doubleda8sa8.exit:             ; preds = %bb, %for.body.i
; CHECK-NEXT:   %4 = load double, double* %"insertptr'ipg"
; CHECK-NEXT:   store double 0.000000e+00, double* %"insertptr'ipg", align 8
; CHECK-NEXT:   tail call void @free(i8* nonnull %"alloc'mi")
; CHECK-NEXT:   tail call void @free(i8* %alloc)
; CHECK-NEXT:   %5 = insertvalue { double } undef, double %4, 0
; CHECK-NEXT:   ret { double } %5
; CHECK-NEXT: }