; RUN: %opt < %s %loadEnzyme -enzyme -enzyme_preopt=false -mem2reg -early-cse -instcombine -simplifycfg -S | FileCheck %s
source_filename = "julia"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-linux-gnu"

%jl_value_t = type opaque

@__stack_chk_guard = external constant %jl_value_t*
@jl_true = external constant %jl_value_t*
@jl_false = external constant %jl_value_t*
@jl_emptysvec = external constant %jl_value_t*
@jl_emptytuple = external constant %jl_value_t*
@jl_diverror_exception = external constant %jl_value_t*
@jl_undefref_exception = external constant %jl_value_t*
@jl_RTLD_DEFAULT_handle = external constant i8*
@jl_world_counter = external global i64

; Function Attrs: nounwind readnone speculatable
declare double @llvm.pow.f64(double, double) #11

define dso_local double @julia_num2num_3(double) {
top:
  %1 = fadd double %0, %0
  %2 = call double @llvm.pow.f64(double 1.031000e+01, double %1)
  %ret = fsub double %2, %0
  ret double %ret
}

define internal nonnull %jl_value_t addrspace(10)* @julia_overdub_1411(double) #9 {
entry:
  unreachable
}

; Function Attrs: alwaysinline
define double @enzyme_entry(double) #12 {
entry:
  %z = call double (...) @__enzyme_autodiff.Float64(double (double)* @julia_num2num_3, double %0)
  ret double %z
}

declare double @__enzyme_autodiff.Float64(...)

attributes #0 = { noreturn }
attributes #1 = { "thunk" }
attributes #2 = { returns_twice }
attributes #3 = { argmemonly nounwind readonly }
attributes #4 = { nounwind readnone }
attributes #5 = { nounwind readonly }
attributes #6 = { noinline }
attributes #7 = { allocsize(1) }
attributes #8 = { argmemonly nounwind }
attributes #9 = { noinline noreturn }
attributes #10 = { cold noreturn nounwind }
attributes #11 = { nounwind readnone speculatable }
attributes #12 = { alwaysinline }
attributes #13 = { inaccessiblemem_or_argmemonly }
attributes #14 = { nounwind }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 1, !"Debug Info Version", i32 3}
!2 = !{!3, !3, i64 0}
!3 = !{!"jtbaa_gcframe", !4, i64 0}
!4 = !{!"jtbaa", !5, i64 0}
!5 = !{!"jtbaa"}
!6 = !{!7, !7, i64 0}
!7 = !{!"jtbaa_tag", !8, i64 0}
!8 = !{!"jtbaa_data", !4, i64 0}
!9 = !{!10, !10, i64 0}
!10 = !{!"jtbaa_immut", !11, i64 0}
!11 = !{!"jtbaa_value", !8, i64 0}
!12 = !{}
!13 = !{i64 8}
!14 = !{!4, !4, i64 0}
!15 = !{!16, !16, i64 0}
!16 = !{!"jtbaa_mutab", !11, i64 0}
!17 = distinct !{}

; CHECK: define internal { double } @diffejulia_num2num_3(double, double %differeturn)
; CHECK-NEXT: top:
; CHECK-NEXT:   %[[x2:.+]] = fadd double %0, %0
; CHECK-NEXT:   %[[pow:.+]] = call double @llvm.pow.f64(double 1.031000e+01, double %[[x2]])
; CHECK-NEXT:   %[[dmul:.+]] = fmul fast double %[[pow]], %differeturn
; CHECK-NEXT:   %[[cmul:.+]] = fmul fast double %[[dmul]], 0x4002AA37D43EE973
; CHECK-NEXT:   %[[sub:.+]] = fsub fast double %[[cmul]], %differeturn
; CHECK-NEXT:   %[[add:.+]] = fadd fast double %[[sub]], %[[cmul]]
; CHECK-NEXT:   %[[res:.+]] = insertvalue { double } undef, double %[[add]], 0
; CHECK-NEXT:   ret { double } %[[res]]
; CHECK-NEXT: }
