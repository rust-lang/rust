; RUN: if [ %llvmver -ge 9 ]; then %opt < %s %loadEnzyme -enzyme -enzyme_preopt=false -mem2reg -adce -correlated-propagation  -instsimplify -simplifycfg -S | FileCheck %s; fi
source_filename = "julia"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

%jl_value_t = type opaque

define dso_local double @julia_overdub_1414(double) local_unnamed_addr {
top:
  br label %L21.lr.ph

L21.lr.ph:                                        ; preds = %L23, %top
  %value_phi227 = phi i64 [ 3, %top ], [ %8, %L23 ]
  %value_phi126 = phi i64 [ 10, %top ], [ %9, %L23 ]
  %value_phi25 = phi i64 [ 10, %top ], [ %5, %L23 ]
  %1 = call i64 @llvm.cttz.i64(i64 %value_phi227, i1 false), !range !2
  %2 = add nuw nsw i64 %1, 1
  %3 = icmp ugt i64 %1, 62
  %.v = select i1 %3, i64 63, i64 %2
  br label %L21

L21:                                              ; preds = %L21.lr.ph, %L21
  %4 = phi i64 [ %1, %L21.lr.ph ], [ %6, %L21 ]
  %value_phi324 = phi i64 [ %value_phi25, %L21.lr.ph ], [ %5, %L21 ]
  %5 = mul i64 %value_phi324, %value_phi324
  %6 = add nsw i64 %4, -1
  %7 = icmp slt i64 %4, 1
  br i1 %7, label %L23, label %L21

L23:                                              ; preds = %L21
  %8 = ashr i64 %value_phi227, %.v
  %9 = mul i64 %5, %value_phi126
  %10 = icmp slt i64 %8, 1
  br i1 %10, label %L28, label %L21.lr.ph

L28:                                              ; preds = %L23
  %11 = icmp sgt i64 %9, 0
  %12 = select i1 %11, i64 %9, i64 0
  br i1 %11, label %L39.preheader, label %L79

L39.preheader:                                    ; preds = %L28
  %13 = fmul double %0, %0
  %14 = fmul double %13, %0
  %15 = fdiv double 1.000000e+00, %0
  br label %L39

L39:                                              ; preds = %L39.preheader, %L64
  %value_phi8 = phi double [ %20, %L64 ], [ 0.000000e+00, %L39.preheader ]
  %value_phi9 = phi i64 [ %22, %L64 ], [ 1, %L39.preheader ]
  switch i64 %value_phi9, label %L61 [
    i64 -1, label %L44
    i64 0, label %L64
    i64 1, label %L51
    i64 2, label %L54
    i64 3, label %L58
  ]

L44:                                              ; preds = %L39
  br label %L64

L51:                                              ; preds = %L39
  br label %L64

L54:                                              ; preds = %L39
  br label %L64

L58:                                              ; preds = %L39
  br label %L64

L61:                                              ; preds = %L39
  %16 = sitofp i64 %value_phi9 to double
  %17 = call double @llvm.pow.f64(double %0, double %16)
  br label %L64

L64:                                              ; preds = %L39, %L61, %L58, %L54, %L51, %L44
  %value_phi11 = phi double [ %15, %L44 ], [ %0, %L51 ], [ %13, %L54 ], [ %14, %L58 ], [ %17, %L61 ], [ 1.000000e+00, %L39 ]
  %18 = sitofp i64 %value_phi9 to double
  %19 = fdiv double %value_phi11, %18
  %20 = fadd double %value_phi8, %19
  %21 = icmp eq i64 %value_phi9, %12
  %22 = add nuw i64 %value_phi9, 1
  br i1 %21, label %L79, label %L39

L79:                                              ; preds = %L64, %L28
  %value_phi15 = phi double [ 0.000000e+00, %L28 ], [ %20, %L64 ]
  ret double %value_phi15
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #0

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1 immarg) #1

; Function Attrs: nounwind readnone speculatable
declare i64 @llvm.cttz.i64(i64, i1 immarg) #0

; Function Attrs: nounwind readnone speculatable
declare double @llvm.pow.f64(double, double) #0

; Function Attrs: alwaysinline
define double @enzyme_entry(double) #2 {
entry:
  %1 = call double (i8*, ...) @__enzyme_autodiff.Float64(i8* bitcast (double (double)* @julia_overdub_1414 to i8*), metadata !"diffe_out", double %0)
  ret double %1
}

declare double @__enzyme_autodiff.Float64(i8*, ...)

; Function Attrs: inaccessiblemem_or_argmemonly
declare void @jl_gc_queue_root(%jl_value_t addrspace(10)*) #3

; Function Attrs: allocsize(1)
declare noalias nonnull %jl_value_t addrspace(10)* @jl_gc_pool_alloc(i8*, i32, i32) #4

; Function Attrs: allocsize(1)
declare noalias nonnull %jl_value_t addrspace(10)* @jl_gc_big_alloc(i8*, i64) #4

attributes #0 = { nounwind readnone speculatable }
attributes #1 = { argmemonly nounwind }
attributes #2 = { alwaysinline }
attributes #3 = { inaccessiblemem_or_argmemonly }
attributes #4 = { allocsize(1) }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 1, !"Debug Info Version", i32 3}
!2 = !{i64 0, i64 65}

; CHECK: define internal { double } @diffejulia_overdub_1414(double{{( %0)?}}, double %differeturn)
