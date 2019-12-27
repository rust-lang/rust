; RUN: %opt < %s %loadEnzyme -enzyme -enzyme_preopt=false -mem2reg -sroa -simplifycfg -instcombine -adce -S | FileCheck %s

source_filename = "./integrateexp.cpp"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%"class.std::ios_base::Init" = type { i8 }
%"class.std::exception" = type { i32 (...)** }
%"class.boost::array.1" = type { [1 x double] }
%"class.boost::numeric::odeint::step_adjustment_error" = type { %"class.boost::numeric::odeint::odeint_error" }
%"class.boost::numeric::odeint::odeint_error" = type { %"class.std::runtime_error" }
%"class.std::runtime_error" = type { %"class.std::exception", %"struct.std::__cow_string" }
%"struct.std::__cow_string" = type { %union.anon }
%union.anon = type { i8* }
%"struct.boost::exception_detail::error_info_injector" = type <{ %"class.boost::numeric::odeint::step_adjustment_error", %"class.boost::exception.base", [4 x i8] }>
%"class.boost::exception.base" = type <{ i32 (...)**, %"class.boost::exception_detail::refcount_ptr", i8*, i8*, i32 }>
%"class.boost::exception_detail::refcount_ptr" = type { %"struct.boost::exception_detail::error_info_container"* }
%"struct.boost::exception_detail::error_info_container" = type { i32 (...)** }
%"class.std::__cxx11::basic_string" = type { %"struct.std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Alloc_hider", i64, %union.anon.24 }
%"struct.std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Alloc_hider" = type { i8* }
%union.anon.24 = type { i64, [8 x i8] }
%"class.std::allocator" = type { i8 }

; Function Attrs: nounwind readnone uwtable
define dso_local zeroext i1 @_Z24approx_fp_equality_floatffd(float %f1, float %f2, double %threshold) local_unnamed_addr #0 {
entry:
  ret i1 false
}

declare dso_local void @_ZNSt8ios_base4InitC1Ev(%"class.std::ios_base::Init"*) unnamed_addr #1

declare dso_local void @_ZNSt8ios_base4InitD1Ev(%"class.std::ios_base::Init"*) unnamed_addr #2

; Function Attrs: nounwind uwtable
define internal void @__dtor__ZStL8__ioinit() #3 section ".text.startup" {
entry:
  ret void
}

declare noalias i8* @malloc(i64)

; Function Attrs: nounwind uwtable
define dso_local double @_Z6foobard(double %t) #3 {
entry:
  %malloccall = tail call i8* @malloc(i64 8) #4
  %x = bitcast i8* %malloccall to double*
  %0 = bitcast i8* %malloccall to i64*
  store i64 4607182418800017408, i64* %0, align 8
  %div = fmul fast double %t, 1.000000e-02
  %x.promoted = load double, double* %x, align 8
  br label %while.body.i.i.i

while.body.i.i.i:                                 ; preds = %while.body.i.i.i, %entry
  %load.i1 = phi double [ %x.promoted, %entry ], [ %add10.i.i.i, %while.body.i.i.i ]
  %step.029.i.i.i = phi i32 [ 0, %entry ], [ %inc.i.i.i, %while.body.i.i.i ]
  %1 = fmul fast double %load.i1, 0xBFF3333333333332
  %reass.mul325.i = fmul fast double %1, %div
  %add10.i.i.i = fadd fast double %reass.mul325.i, %load.i1
  %inc.i.i.i = add nuw nsw i32 %step.029.i.i.i, 1
  %conv8.i.i.i = sitofp i32 %inc.i.i.i to double
  %mul.i.i.i = fmul fast double %div, %conv8.i.i.i
  %add.i.i.i = fadd fast double %mul.i.i.i, %div
  %sub.i.i.i.i = fsub fast double %add.i.i.i, %t
  %cmp2.i.i.i.i = fcmp fast ugt double %sub.i.i.i.i, 0x3CB0000000000000
  br i1 %cmp2.i.i.i.i, label %loopexit, label %while.body.i.i.i

loopexit:                                         ; preds = %while.body.i.i.i
  store double %add10.i.i.i, double* %x, align 8
  ret double %add10.i.i.i
}

; CHECK: define internal { double } @diffe_Z6foobard(double %t, double %differeturn)

; CHECK-NEXT: entry:
; CHECK-NEXT:   %div = fmul fast double %t, 1.000000e-02
; CHECK-NEXT:   br label %while.body.i.i.i

; CHECK: while.body.i.i.i:
; CHECK-NEXT:   %0 = phi i8* [ null, %entry ], [ %_realloccache, %while.body.i.i.i ]
; CHECK-NEXT:   %iv = phi i64 [ 0, %entry ], [ %iv.next, %while.body.i.i.i ]
; CHECK-NEXT:   %load.i1 = phi double [ 1.000000e+00, %entry ], [ %add10.i.i.i, %while.body.i.i.i ]
; CHECK-NEXT:   %1 = trunc i64 %iv to i32
; CHECK-NEXT:   %iv.next = add nuw i64 %iv, 1
; CHECK-NEXT:   %2 = shl nuw i64 %iv.next, 3
; CHECK-NEXT:   %_realloccache = call i8* @realloc(i8* %0, i64 %2) #4
; CHECK-NEXT:   %_realloccast = bitcast i8* %_realloccache to double*
; CHECK-NEXT:   %3 = fmul fast double %load.i1, 0xBFF3333333333332
; CHECK-NEXT:   %4 = getelementptr double, double* %_realloccast, i64 %iv
; CHECK-NEXT:   store double %3, double* %4, align 8, !invariant.group ![[igs:.+]]
; CHECK-NEXT:   %reass.mul325.i = fmul fast double %3, %div
; CHECK-NEXT:   %add10.i.i.i = fadd fast double %reass.mul325.i, %load.i1
; CHECK-NEXT:   %inc.i.i.i = add nuw nsw i32 %1, 1
; CHECK-NEXT:   %conv8.i.i.i = sitofp i32 %inc.i.i.i to double
; CHECK-NEXT:   %mul.i.i.i = fmul fast double %div, %conv8.i.i.i
; CHECK-NEXT:   %add.i.i.i = fadd fast double %mul.i.i.i, %div
; CHECK-NEXT:   %sub.i.i.i.i = fsub fast double %add.i.i.i, %t
; CHECK-NEXT:   %cmp2.i.i.i.i = fcmp fast ugt double %sub.i.i.i.i, 0x3CB0000000000000
; CHECK-NEXT:   br i1 %cmp2.i.i.i.i, label %invertwhile.body.i.i.i, label %while.body.i.i.i

; CHECK: invertentry:
; CHECK-NEXT:   tail call void @free(i8* nonnull %_realloccache)
; CHECK-NEXT:   %m0diffet = fmul fast double %8, 1.000000e-02
; CHECK-NEXT:   %5 = insertvalue { double } undef, double %m0diffet, 0
; CHECK-NEXT:   ret { double } %5

; CHECK: invertwhile.body.i.i.i:
; CHECK-NEXT:   %"div'de.0" = phi double [ %8, %incinvertwhile.body.i.i.i ], [ 0.000000e+00, %while.body.i.i.i ]
; CHECK-NEXT:   %"add10.i.i.i'de.0" = phi double [ %10, %incinvertwhile.body.i.i.i ], [ %differeturn, %while.body.i.i.i ]
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %11, %incinvertwhile.body.i.i.i ], [ %iv, %while.body.i.i.i ]
; CHECK-NEXT:   %6 = getelementptr double, double* %_realloccast, i64 %"iv'ac.0"
; CHECK-NEXT:   %7 = load double, double* %6, align 8, !invariant.group ![[igs]]
; CHECK-NEXT:   %m1diffediv = fmul fast double %"add10.i.i.i'de.0", %7
; CHECK-NEXT:   %8 = fadd fast double %"div'de.0", %m1diffediv
; CHECK-NEXT:   %9 = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %9, label %invertentry, label %incinvertwhile.body.i.i.i

; CHECK: incinvertwhile.body.i.i.i:
; CHECK-NEXT:   %div_unwrap = fmul fast double %t, 1.000000e-02
; CHECK-NEXT:   %m0diffe = fmul fast double %"add10.i.i.i'de.0", %div_unwrap
; CHECK-NEXT:   %m0diffeload.i1 = fmul fast double %m0diffe, 0xBFF3333333333332
; CHECK-NEXT:   %10 = fadd fast double %"add10.i.i.i'de.0", %m0diffeload.i1
; CHECK-NEXT:   %11 = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertwhile.body.i.i.i
; CHECK-NEXT: }

; Function Attrs: norecurse nounwind uwtable
define double @caller(double %inp) {
entry:
  %call = tail call fast double @__enzyme_autodiff(i8* bitcast (double (double)* @_Z6foobard to i8*), double %inp)
  ret double %call
}

declare dso_local double @__enzyme_autodiff(i8*, double) local_unnamed_addr #1


attributes #0 = { nounwind readnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #2 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #3 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #4 = { nounwind }
attributes #5 = { norecurse nounwind readnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #6 = { alwaysinline norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #7 = { argmemonly nounwind }
attributes #8 = { norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #9 = { nounwind readnone speculatable }
attributes #10 = { noreturn nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #11 = { inlinehint nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #12 = { nobuiltin nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #13 = { builtin nounwind }

!2 = !{!3, !3, i64 0}
!3 = !{!"double", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C++ TBAA"}
!6 = !{!7, !9, i64 16}
!7 = !{!"_ZTSN5boost9exceptionE", !8, i64 8, !9, i64 16, !9, i64 24, !10, i64 32}
!8 = !{!"_ZTSN5boost16exception_detail12refcount_ptrINS0_20error_info_containerEEE", !9, i64 0}
!9 = !{!"any pointer", !4, i64 0}
!10 = !{!"int", !4, i64 0}
!11 = !{!7, !9, i64 24}
!12 = !{!7, !10, i64 32}
!13 = !{!14, !14, i64 0}
!14 = !{!"vtable pointer", !5, i64 0}
!15 = !{!8, !9, i64 0}
