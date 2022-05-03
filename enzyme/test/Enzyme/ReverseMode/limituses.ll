; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -adce -loop-deletion -correlated-propagation -simplifycfg -adce -S | FileCheck %s

; ModuleID = 'test.cpp'
source_filename = "test.cpp"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@.str = private unnamed_addr constant [14 x i8] c"reduce_max=%f\00", align 1
@.str.1 = private unnamed_addr constant [20 x i8] c"d_reduce_max(%i)=%f\00", align 1

; Function Attrs: nounwind uwtable
define dso_local double @_Z10reduce_maxPdi(double* nocapture readonly %vec, double* %end) #0 {
entry:
  br label %for.cond13

for.cond13:                                       ; preds = %for.cond13.preheader, %for.body15
  %ptr = phi double* [ %incdec.ptr.i445, %for.cond13 ], [ %vec, %entry ]
  %ret.2 = phi double [ %add, %for.cond13 ], [ 0.000000e+00, %entry ]
  %a13 = load double, double* %ptr, align 8
  %add = fadd fast double %ret.2, %a13
  %incdec.ptr.i445 = getelementptr inbounds double, double* %ptr, i32 1
  %cmp.i432 = icmp ne double* %ptr, %end
  br i1 %cmp.i432, label %for.cond13, label %endloop

endloop:                                        ; preds = %for.cond13
  %endi64 = ptrtoint double* %end to i64
  %veci64 = ptrtoint double* %vec to i64
  %sub.ptr = sub i64 %endi64, %veci64
  %num = sdiv exact i64 %sub.ptr, 8
  %conv = uitofp i64 %num to double
  %div = fdiv fast double %add, %conv
  %sub.ptr.rhs.cast.i480 = ptrtoint double* %vec to i64
  ret double %div
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

; Function Attrs: norecurse nounwind uwtable
define dso_local i32 @main() local_unnamed_addr #2 {
entry:
  %vec = alloca [5 x double], align 16
  %d_vec = alloca [5 x double], align 16
  %0 = bitcast [5 x double]* %vec to i8*
  call void @llvm.lifetime.start.p0i8(i64 40, i8* nonnull %0) #9
  %1 = getelementptr inbounds [5 x double], [5 x double]* %vec, i64 0, i64 0
  store double -1.000000e+00, double* %1, align 16
  %2 = getelementptr inbounds [5 x double], [5 x double]* %vec, i64 0, i64 1
  store double 2.000000e+00, double* %2, align 8
  %3 = getelementptr inbounds [5 x double], [5 x double]* %vec, i64 0, i64 2
  store double -2.000000e-01, double* %3, align 16
  %4 = getelementptr inbounds [5 x double], [5 x double]* %vec, i64 0, i64 3
  store double 2.000000e+00, double* %4, align 8
  %5 = getelementptr inbounds [5 x double], [5 x double]* %vec, i64 0, i64 4
  store double 1.000000e+00, double* %5, align 16
  %6 = bitcast [5 x double]* %d_vec to i8*
  call void @llvm.lifetime.start.p0i8(i64 40, i8* nonnull %6) #9
  call void @llvm.memset.p0i8.i64(i8* nonnull align 16 %6, i8 0, i64 40, i1 false)
  %gep = getelementptr double, double* %1, i32 5
  %call = call fast double @_Z10reduce_maxPdi(double* nonnull %1, double* %gep)
  %call1 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([14 x i8], [14 x i8]* @.str, i64 0, i64 0), double %call)
  %arraydecay3 = getelementptr inbounds [5 x double], [5 x double]* %d_vec, i64 0, i64 0
  call void @_Z17__enzyme_autodiffPvPdS0_i(i8* bitcast (double (double*, double*)* @_Z10reduce_maxPdi to i8*), double* nonnull %1, double* nonnull %arraydecay3, double* %gep, double* %gep) #9
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  call void @llvm.lifetime.end.p0i8(i64 40, i8* nonnull %6) #9
  call void @llvm.lifetime.end.p0i8(i64 40, i8* nonnull %0) #9
  ret i32 0

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds [5 x double], [5 x double]* %d_vec, i64 0, i64 %indvars.iv
  %7 = load double, double* %arrayidx, align 8, !tbaa !2
  %8 = trunc i64 %indvars.iv to i32
  %call4 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([20 x i8], [20 x i8]* @.str.1, i64 0, i64 0), i32 %8, double %7)
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 5
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1) #1

; Function Attrs: nounwind
declare dso_local i32 @printf(i8* nocapture readonly, ...) local_unnamed_addr #3

declare dso_local void @_Z17__enzyme_autodiffPvPdS0_i(i8*, double*, double*, double*, double*) local_unnamed_addr #4

; Function Attrs: nobuiltin nounwind
declare dso_local void @_ZdlPv(i8*) local_unnamed_addr #5

; Function Attrs: noreturn
declare dso_local void @_ZSt17__throw_bad_allocv() local_unnamed_addr #6

; Function Attrs: nobuiltin
declare dso_local noalias nonnull i8* @_Znwm(i64) local_unnamed_addr #7

; Function Attrs: argmemonly nounwind
declare void @llvm.memmove.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i1) #1

attributes #0 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #3 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #4 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #5 = { nobuiltin nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #6 = { noreturn "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #7 = { nobuiltin "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #8 = { noreturn nounwind }
attributes #9 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0}
!3 = !{!"double", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C++ TBAA"}


; CHECK: define internal void @diffe_Z10reduce_maxPdi(double* nocapture readonly %vec, double* nocapture %"vec'", double* %end, double* %"end'", double %differeturn) 
; CHECK-NEXT: entry:
; CHECK-NEXT:   br label %for.cond13

; CHECK: for.cond13:                                       ; preds = %for.cond13, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %for.cond13 ], [ 0, %entry ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %scevgep = getelementptr double, double* %vec, i64 %iv
; CHECK-NEXT:   %cmp.i432 = icmp ne double* %scevgep, %end
; CHECK-NEXT:   br i1 %cmp.i432, label %for.cond13, label %endloop

; CHECK: endloop:                                          ; preds = %for.cond13
; CHECK-NEXT:   %endi64 = ptrtoint double* %end to i64
; CHECK-NEXT:   %veci64 = ptrtoint double* %vec to i64
; CHECK-NEXT:   %sub.ptr = sub i64 %endi64, %veci64
; CHECK-NEXT:   %num = sdiv exact i64 %sub.ptr, 8
; CHECK-NEXT:   %conv = uitofp i64 %num to double
; CHECK-NEXT:   %d0diffeadd = fdiv fast double %differeturn, %conv
; CHECK-NEXT:   br label %invertfor.cond13

; CHECK: invertentry:                                      ; preds = %invertfor.cond13
; CHECK-NEXT:   ret void

; CHECK: invertfor.cond13:                                 ; preds = %endloop, %incinvertfor.cond13
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %iv, %endloop ], [ %3, %incinvertfor.cond13 ]
; CHECK-NEXT:   %"scevgep'ipg_unwrap" = getelementptr double, double* %"vec'", i64 %"iv'ac.0"
; CHECK-NEXT:   %0 = load double, double* %"scevgep'ipg_unwrap", align 8
; CHECK-NEXT:   %1 = fadd fast double %0, %d0diffeadd
; CHECK-NEXT:   store double %1, double* %"scevgep'ipg_unwrap", align 8
; CHECK-NEXT:   %2 = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %2, label %invertentry, label %incinvertfor.cond13

; CHECK: incinvertfor.cond13:                              ; preds = %invertfor.cond13
; CHECK-NEXT:   %3 = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertfor.cond13
; CHECK-NEXT: }