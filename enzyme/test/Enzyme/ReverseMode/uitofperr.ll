; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -simplifycfg -instsimplify -early-cse -correlated-propagation -adce -S | FileCheck %s

%"class.std::ios_base::Init" = type { i8 }
%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }
%"class.std::exception" = type { i32 (...)** }
%"class.boost::array.1" = type { [1 x double] }

@_ZStL8__ioinit = internal global %"class.std::ios_base::Init" zeroinitializer, align 1
@.str = private unnamed_addr constant [46 x i8] c"final result t=%f x(t)=%f, -0.2=%f, steps=%d\0A\00", align 1
@.str.3 = private unnamed_addr constant [48 x i8] c"t=%f d/dt(exp(-1.2*t))=%f, -1.2*exp(-1.2*t)=%f\0A\00", align 1
@stderr = external dso_local local_unnamed_addr global %struct._IO_FILE*, align 8
@.str.4 = private unnamed_addr constant [68 x i8] c"Assertion Failed: fabs( [%s = %g] - [%s = %g] ) > %g at %s:%d (%s)\0A\00", align 1
@.str.5 = private unnamed_addr constant [4 x i8] c"res\00", align 1
@.str.6 = private unnamed_addr constant [11 x i8] c"realanswer\00", align 1
@.str.7 = private unnamed_addr constant [17 x i8] c"integrateexp.cpp\00", align 1
@__PRETTY_FUNCTION__.main = private unnamed_addr constant [23 x i8] c"int main(int, char **)\00", align 1
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_integrateexp.cpp, i8* null }]

; Function Attrs: nounwind readnone uwtable
define dso_local zeroext i1 @_Z24approx_fp_equality_floatffd(float %f1, float %f2, double %threshold) local_unnamed_addr #0 {
entry:
  %sub = fsub fast float %f1, %f2
  %0 = tail call fast float @llvm.fabs.f32(float %sub) #4
  %conv = fpext float %0 to double
  %cmp = fcmp fast ule double %conv, %threshold
  ret i1 %cmp
}

declare dso_local void @_ZNSt8ios_base4InitC1Ev(%"class.std::ios_base::Init"*) unnamed_addr #1

; Function Attrs: nounwind
declare dso_local void @_ZNSt8ios_base4InitD1Ev(%"class.std::ios_base::Init"*) unnamed_addr #2

; Function Attrs: nounwind uwtable
define internal void @__dtor__ZStL8__ioinit() #3 section ".text.startup" {
entry:
  tail call void @_ZNSt8ios_base4InitD1Ev(%"class.std::ios_base::Init"* nonnull @_ZStL8__ioinit)
  ret void
}

; Function Attrs: nounwind
declare dso_local i32 @atexit(void ()*) local_unnamed_addr #4

; Function Attrs: norecurse nounwind readnone uwtable
define dso_local void @_ZN5boost15throw_exceptionERKSt9exception(%"class.std::exception"* nocapture dereferenceable(8) %e) local_unnamed_addr #5 {
entry:
  ret void
}

; Function Attrs: norecurse nounwind uwtable
define dso_local void @_Z6lorenzRKN5boost5arrayIdLm1EEERS1_d(%"class.boost::array.1"* nocapture readonly dereferenceable(8) %x, %"class.boost::array.1"* nocapture dereferenceable(8) %dxdt, double %t) local_unnamed_addr #6 {
entry:
  %arrayidx.i = getelementptr inbounds %"class.boost::array.1", %"class.boost::array.1"* %x, i64 0, i32 0, i64 0
  %0 = load double, double* %arrayidx.i, align 8, !tbaa !2
  %mul = fmul fast double %0, -1.200000e+00
  %arrayidx.i3 = getelementptr inbounds %"class.boost::array.1", %"class.boost::array.1"* %dxdt, i64 0, i32 0, i64 0
  store double %mul, double* %arrayidx.i3, align 8, !tbaa !2
  ret void
}

; Function Attrs: nounwind uwtable
define dso_local double @_Z6foobard(double %t) #3 {
entry:
  %mul = fmul fast double %t, -1.200000e+00
  br label %while

while:                              ; preds = %entry, %while.body.us.i.i.i
  %0 = phi double [ %add, %while ], [ 1.000000e+00, %entry ]
  %i = phi i32 [ %nexti, %while ], [ 0, %entry ]
  %mul2 = fmul fast double %mul, %0
  %add = fadd fast double %mul2, %0
  %nexti = add nuw nsw i32 %i, 1
  %conv = sitofp i32 %nexti to double
  %mul.us.i.i.i = fmul fast double %conv, %t
  %cmp = fcmp fast ugt double %mul.us.i.i.i, 0x3CB0000000000000
  br i1 %cmp, label %exit, label %while

exit: ; preds = %while.body.i.i.i, %while
  %a2 = phi double [ %add, %while ]
  %a3 = phi i32 [ %nexti, %while ]
  %a4 = zext i32 %a3 to i64
  %call2 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([46 x i8], [46 x i8]* @.str, i64 0, i64 0), double %t, double %a2, double -2.000000e-01, i64 %a4)
  ret double %a2
}

; Function Attrs: nounwind
declare dso_local i32 @printf(i8* nocapture readonly, ...) local_unnamed_addr #2

; Function Attrs: norecurse nounwind uwtable
define dso_local i32 @main(i32 %argc, i8** nocapture readnone %argv) local_unnamed_addr #6 {
entry:
  br label %for.body

for.cond:                                         ; preds = %for.body
  %cmp = icmp ult i32 %inc, 101
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  ret i32 0

for.body:                                         ; preds = %entry, %for.cond
  %i.033 = phi i32 [ 1, %entry ], [ %inc, %for.cond ]
  %conv = sitofp i32 %i.033 to double
  %div = fmul fast double %conv, 1.000000e-01
  %call = tail call fast double @__enzyme_autodiff(i8* bitcast (double (double)* @_Z6foobard to i8*), double %div) #4
  %call1 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([48 x i8], [48 x i8]* @.str.3, i64 0, i64 0), double %div, double %call, double -1.200000e+00)
  %sub = fadd fast double %call, 1.200000e+00
  %0 = tail call fast double @llvm.fabs.f64(double %sub)
  %cmp5 = fcmp fast ogt double %0, 1.200000e-01
  %inc = add nuw nsw i32 %i.033, 1
  br i1 %cmp5, label %if.then, label %for.cond

if.then:                                          ; preds = %for.body
  %1 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !6
  %call10 = tail call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %1, i8* getelementptr inbounds ([68 x i8], [68 x i8]* @.str.4, i64 0, i64 0), i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.5, i64 0, i64 0), double %call, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.6, i64 0, i64 0), double -1.200000e+00, double 1.200000e-01, i8* getelementptr inbounds ([17 x i8], [17 x i8]* @.str.7, i64 0, i64 0), i32 66, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @__PRETTY_FUNCTION__.main, i64 0, i64 0)) #9
  tail call void @abort() #10
  unreachable
}

declare dso_local double @__enzyme_autodiff(i8*, double) local_unnamed_addr #1

; Function Attrs: nounwind readnone speculatable
declare double @llvm.fabs.f64(double) #7

; Function Attrs: nounwind
declare dso_local i32 @fprintf(%struct._IO_FILE* nocapture, i8* nocapture readonly, ...) local_unnamed_addr #2

; Function Attrs: noreturn nounwind
declare dso_local void @abort() local_unnamed_addr #8

; Function Attrs: nounwind readnone speculatable
declare float @llvm.fabs.f32(float) #7

; Function Attrs: nounwind uwtable
define internal void @_GLOBAL__sub_I_integrateexp.cpp() #3 section ".text.startup" {
entry:
  tail call void @_ZNSt8ios_base4InitC1Ev(%"class.std::ios_base::Init"* nonnull @_ZStL8__ioinit) #4
  %0 = tail call i32 @atexit(void ()* nonnull @__dtor__ZStL8__ioinit) #4
  ret void
}

attributes #0 = { nounwind readnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #2 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #3 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #4 = { nounwind }
attributes #5 = { norecurse nounwind readnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #6 = { norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #7 = { nounwind readnone speculatable }
attributes #8 = { noreturn nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #9 = { cold }
attributes #10 = { noreturn nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0}
!3 = !{!"double", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C++ TBAA"}
!6 = !{!7, !7, i64 0}
!7 = !{!"any pointer", !4, i64 0}

; CHECK: define internal { double } @diffe_Z6foobard(double %t, double %differeturn) #3 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %mul = fmul fast double %t, -1.200000e+00
; CHECK-NEXT:   br label %while

; CHECK: while: 
; CHECK-NEXT:   %[[phiload:.+]] = phi double* [ null, %entry ], [ %[[_realloccast:.+]], %[[mergeblk:.+]] ]
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %[[mergeblk]] ], [ 0, %entry ] 
; CHECK-NEXT:   %[[phi1:.+]] = phi double [ %add, %[[mergeblk]] ], [ 1.000000e+00, %entry ] 
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %[[phibc:.+]] = bitcast double* %[[phiload]] to i8*
; CHECK-NEXT:   %[[nexttrunc0:.+]] = and i64 %iv.next, 1
; CHECK-NEXT:   %[[nexttrunc:.+]] = icmp ne i64 %[[nexttrunc0]], 0
; CHECK-NEXT:   %[[popcnt:.+]] = call i64 @llvm.ctpop.i64(i64 %iv.next)
; CHECK-NEXT:   %[[le2:.+]] = icmp ult i64 %[[popcnt:.+]], 3
; CHECK-NEXT:   %[[shouldgrow:.+]] = and i1 %[[le2]], %[[nexttrunc]]
; CHECK-NEXT:   br i1 %[[shouldgrow]], label %grow.i, label %[[mergeblk]]

; CHECK: grow.i:
; CHECK-NEXT:   %[[ctlz:.+]] = call i64 @llvm.ctlz.i64(i64 %iv.next, i1 true)
; CHECK-NEXT:   %[[maxbit:.+]] = sub nuw nsw i64 64, %[[ctlz]]
; CHECK-NEXT:   %[[numbytes:.+]] = shl i64 8, %[[maxbit]]
; CHECK-NEXT:   %[[growalloc:.+]] = call i8* @realloc(i8* %[[phibc]], i64 %[[numbytes]])
; CHECK-NEXT:   br label %[[mergeblk]]

; CHECK: [[mergeblk]]:
; CHECK-NEXT:   %[[gphi:.+]] = phi i8* [ %[[growalloc]], %grow.i ], [ %[[phibc]], %while ]
; CHECK-NEXT:   %[[_realloccast:.+]] = bitcast i8* %[[gphi]] to double*
; CHECK-NEXT:   %[[gep:.+]] = getelementptr inbounds double, double* %[[_realloccast]], i64 %iv
; CHECK-NEXT:   store double %[[phi1:.+]], double* %[[gep]], align 8, !invariant.group !8
; CHECK-NEXT:   %[[trunc:.+]] = trunc i64 %iv to i32
; CHECK-NEXT:   %mul2 = fmul fast double %mul, %[[phi1]]
; CHECK-NEXT:   %add = fadd fast double %mul2, %[[phi1]]
; CHECK-NEXT:   %nexti = add nuw nsw i32 %[[trunc]], 1
; CHECK-NEXT:   %conv = sitofp i32 %nexti to double
; CHECK-NEXT:   %mul.us.i.i.i = fmul fast double %conv, %t
; CHECK-NEXT:   %cmp = fcmp fast ugt double %mul.us.i.i.i, 0x3CB0000000000000
; CHECK-NEXT:   br i1 %cmp, label %exit, label %while

; CHECK: exit:
; CHECK-NEXT:   %a4 = zext i32 %nexti to i64
; CHECK-NEXT:   %call2 = tail call i32 (i8*, ...) @printf
; CHECK-NEXT:   br label %invertwhile

; CHECK: invertentry:                                      ; preds = %invertwhile
; CHECK-NEXT:   %m0diffet = fmul fast double %[[fadd:.+]], -1.200000e+00
; CHECK-NEXT:   %[[toret:.+]] = insertvalue { double } undef, double %m0diffet, 0
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[gphi]])
; CHECK-NEXT:   ret { double } %[[toret]]

; CHECK: invertwhile:                                      ; preds = %exit, %incinvertwhile
; CHECK-NEXT:   %"mul'de.0" = phi double [ 0.000000e+00, %exit ], [ %[[fadd:.+]], %incinvertwhile ]
; CHECK-NEXT:   %"add'de.0" = phi double [ %differeturn, %exit ], [ %[[dad:.+]], %incinvertwhile ]
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %iv, %exit ], [ %[[sub:.+]], %incinvertwhile ]
; CHECK-NEXT:   %[[igep:.+]] = getelementptr inbounds double, double* %[[_realloccast]], i64 %"iv'ac.0"
; CHECK-NEXT:   %[[il:.+]] = load double, double* %[[igep]], align 8, !invariant.group !8
; CHECK-NEXT:   %m0diffemul = fmul fast double %"add'de.0", %[[il]]
; CHECK-NEXT:   %m1diffe = fmul fast double %"add'de.0", %mul
; CHECK-NEXT:   %[[fadd]] = fadd fast double %"mul'de.0", %m0diffemul
; CHECK-NEXT:   %[[dad]] = fadd fast double %"add'de.0", %m1diffe
; CHECK-NEXT:   %[[ieq:.+]] = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %[[ieq]], label %invertentry, label %incinvertwhile

; CHECK: incinvertwhile:                                   ; preds = %invertwhile
; CHECK-NEXT:   %[[sub]] = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertwhile
; CHECK-NEXT: }
