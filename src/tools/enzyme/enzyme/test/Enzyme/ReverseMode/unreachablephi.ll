; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -simplifycfg -instsimplify -gvn -adce -S | FileCheck %s

source_filename = "text"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:10:11:12:13"
target triple = "x86_64-pc-linux-gnu"


define internal fastcc i64 @julia_ht_keyindex_1432({} addrspace(10)* nocapture nonnull readonly align 8 dereferenceable(64) %arg, {} addrspace(10)* nonnull %arg1) {
top:
  %i = call {}*** @julia.ptls_states()
  %i2 = bitcast {} addrspace(10)* %arg to i8 addrspace(10)*
  %i3 = addrspacecast i8 addrspace(10)* %i2 to i8 addrspace(11)*
  %i4 = getelementptr inbounds i8, i8 addrspace(11)* %i3, i64 8
  %i5 = bitcast i8 addrspace(11)* %i4 to { i8 addrspace(13)*, i64, i16, i16, i32 } addrspace(10)* addrspace(11)*
  %i6 = load atomic { i8 addrspace(13)*, i64, i16, i16, i32 } addrspace(10)*, { i8 addrspace(13)*, i64, i16, i16, i32 } addrspace(10)* addrspace(11)* %i5 unordered, align 8
  %i7 = addrspacecast { i8 addrspace(13)*, i64, i16, i16, i32 } addrspace(10)* %i6 to { i8 addrspace(13)*, i64, i16, i16, i32 } addrspace(11)*
  %i8 = getelementptr inbounds { i8 addrspace(13)*, i64, i16, i16, i32 }, { i8 addrspace(13)*, i64, i16, i16, i32 } addrspace(11)* %i7, i64 0, i32 1
  %i9 = load i64, i64 addrspace(11)* %i8, align 8
  %i10 = getelementptr inbounds i8, i8 addrspace(11)* %i3, i64 56
  %i11 = bitcast i8 addrspace(11)* %i10 to i64 addrspace(11)*
  %i12 = load i64, i64 addrspace(11)* %i11, align 8
  %i13 = call i64 @jl_object_id({} addrspace(10)* nonnull %arg1) "nofree"
  %i14 = shl i64 %i13, 21
  %i15 = xor i64 %i14, -1
  %i16 = add i64 %i13, %i15
  %i17 = lshr i64 %i16, 24
  %i18 = xor i64 %i17, %i16
  %i19 = mul i64 %i18, 265
  %i20 = lshr i64 %i19, 14
  %i21 = xor i64 %i20, %i19
  %i22 = mul i64 %i21, 21
  %i23 = lshr i64 %i22, 28
  %i24 = xor i64 %i23, %i22
  %i25 = mul i64 %i24, 2147483649
  %i26 = add nsw i64 %i9, -1
  %i27 = bitcast { i8 addrspace(13)*, i64, i16, i16, i32 } addrspace(10)* %i6 to {} addrspace(10)* addrspace(13)* addrspace(10)*
  %i28 = bitcast {} addrspace(10)* %arg to { i8 addrspace(13)*, i64, i16, i16, i32 } addrspace(10)* addrspace(10)*
  %i29 = addrspacecast { i8 addrspace(13)*, i64, i16, i16, i32 } addrspace(10)* addrspace(10)* %i28 to { i8 addrspace(13)*, i64, i16, i16, i32 } addrspace(10)* addrspace(11)*
  %i30 = load atomic { i8 addrspace(13)*, i64, i16, i16, i32 } addrspace(10)*, { i8 addrspace(13)*, i64, i16, i16, i32 } addrspace(10)* addrspace(11)* %i29 unordered, align 8
  %i31 = addrspacecast { i8 addrspace(13)*, i64, i16, i16, i32 } addrspace(10)* %i30 to { i8 addrspace(13)*, i64, i16, i16, i32 } addrspace(11)*
  %i32 = getelementptr inbounds { i8 addrspace(13)*, i64, i16, i16, i32 }, { i8 addrspace(13)*, i64, i16, i16, i32 } addrspace(11)* %i31, i64 0, i32 0
  %i33 = load i8 addrspace(13)*, i8 addrspace(13)* addrspace(11)* %i32, align 16
  %i34 = addrspacecast {} addrspace(10)* addrspace(13)* addrspace(10)* %i27 to {} addrspace(10)* addrspace(13)* addrspace(11)*
  %i35 = load {} addrspace(10)* addrspace(13)*, {} addrspace(10)* addrspace(13)* addrspace(11)* %i34, align 16
  br label %L84

L84:                                              ; preds = %L106, %top
  %.pn = phi i64 [ %i25, %top ], [ %value_phi, %L106 ]
  %value_phi1 = phi i64 [ 0, %top ], [ %i40, %L106 ]
  %value_phi.in = and i64 %.pn, %i26
  %value_phi = add i64 %value_phi.in, 1
  %i36 = getelementptr inbounds i8, i8 addrspace(13)* %i33, i64 %value_phi.in
  %i37 = load i8, i8 addrspace(13)* %i36, align 1
  switch i8 %i37, label %L97 [
    i8 0, label %L105
    i8 2, label %L106
  ]

L97:                                              ; preds = %L84
  %i38 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(13)* %i35, i64 %value_phi.in
  %i39 = load atomic {} addrspace(10)*, {} addrspace(10)* addrspace(13)* %i38 unordered, align 8
  %.not12 = icmp eq {} addrspace(10)* %i39, null
  br i1 %.not12, label %fail, label %pass

L105:                                             ; preds = %pass, %L106, %L84
  %merge.ph = phi i64 [ %value_phi, %pass ], [ -1, %L84 ], [ -1, %L106 ]
  ret i64 %merge.ph

L106:                                             ; preds = %pass, %L84
  %i40 = add i64 %value_phi1, 1
  %.not13 = icmp slt i64 %i12, %i40
  br i1 %.not13, label %L105, label %L84

fail:                                             ; preds = %L97
  call void @jl_throw({} addrspace(10)* addrspacecast ({}* inttoptr (i64 140161201230928 to {}*) to {} addrspace(10)*)) #1
  unreachable

pass:                                             ; preds = %L97
  %i41 = icmp eq {} addrspace(10)* %i39, %arg1
  br i1 %i41, label %L105, label %L106
}

; Function Attrs: readnone
declare {}*** @julia.ptls_states() local_unnamed_addr #0

; Function Attrs: noreturn
declare void @jl_throw({} addrspace(10)*) local_unnamed_addr #1

; Function Attrs: inaccessiblememonly allocsize(1)
declare noalias nonnull {} addrspace(12)* @julia.gc_alloc_obj(i8*, i64, {} addrspace(10)*) local_unnamed_addr #2

; Function Attrs: readonly
declare i64 @jl_object_id({} addrspace(10)*) local_unnamed_addr #3

declare double @__enzyme_autodiff(...)

define double @dsquare({} addrspace(10)* nocapture nonnull readonly align 8 dereferenceable(64) %arg, {} addrspace(10)* nocapture nonnull readonly align 8 dereferenceable(64) %arg1) local_unnamed_addr {
entry:
  %call = tail call double (...) @__enzyme_autodiff(i8* bitcast (double ({} addrspace(10)*)* @julia_sum_rec_1428.inner.1 to i8*), metadata !"enzyme_dup", {} addrspace(10)* nocapture nonnull readonly align 8 dereferenceable(64) %arg, {} addrspace(10)* nocapture nonnull readonly align 8 dereferenceable(64) %arg1)
  ret double %call
}

define double @julia_sum_rec_1428.inner.1({} addrspace(10)* nocapture nonnull readonly align 8 dereferenceable(64) %arg) local_unnamed_addr {
entry:
  %i1 = bitcast {} addrspace(10)* %arg to i8 addrspace(10)*
  %i2 = addrspacecast i8 addrspace(10)* %i1 to i8 addrspace(11)*

  %i31 = bitcast i8 addrspace(11)* %i2 to {} addrspace(10)* addrspace(13)* addrspace(10)* addrspace(11)*
  %i47 = load atomic {} addrspace(10)* addrspace(13)* addrspace(10)*, {} addrspace(10)* addrspace(13)* addrspace(10)* addrspace(11)* %i31 unordered, align 8


  %i3 = getelementptr inbounds i8, i8 addrspace(11)* %i2, i64 48
  %i4 = bitcast i8 addrspace(11)* %i3 to i64 addrspace(11)*
  %i5 = load i64, i64 addrspace(11)* %i4, align 8
  %i6 = bitcast {} addrspace(10)* %arg to { i8 addrspace(13)*, i64, i16, i16, i32 } addrspace(10)* addrspace(10)*
  %i7 = addrspacecast { i8 addrspace(13)*, i64, i16, i16, i32 } addrspace(10)* addrspace(10)* %i6 to { i8 addrspace(13)*, i64, i16, i16, i32 } addrspace(10)* addrspace(11)*
  %i8 = load atomic { i8 addrspace(13)*, i64, i16, i16, i32 } addrspace(10)*, { i8 addrspace(13)*, i64, i16, i16, i32 } addrspace(10)* addrspace(11)* %i7 unordered, align 8
  %i9 = addrspacecast { i8 addrspace(13)*, i64, i16, i16, i32 } addrspace(10)* %i8 to { i8 addrspace(13)*, i64, i16, i16, i32 } addrspace(11)*
  %i14 = getelementptr inbounds { i8 addrspace(13)*, i64, i16, i16, i32 }, { i8 addrspace(13)*, i64, i16, i16, i32 } addrspace(11)* %i9, i64 0, i32 0
  %i15 = load i8 addrspace(13)*, i8 addrspace(13)* addrspace(11)* %i14, align 16
  %i16 = add i64 %i5, -1
  %i17 = getelementptr inbounds i8, i8 addrspace(13)* %i15, i64 %i16
  %i18 = load i8, i8 addrspace(13)* %i17, align 1
  %.not26 = icmp eq i8 %i18, 1
  
  %i21 = bitcast i8 addrspace(11)* %i2 to {} addrspace(10)* addrspace(13)* addrspace(10)* addrspace(11)*
  %i22 = load atomic {} addrspace(10)* addrspace(13)* addrspace(10)*, {} addrspace(10)* addrspace(13)* addrspace(10)* addrspace(11)* %i21 unordered, align 8
  %i23 = addrspacecast {} addrspace(10)* addrspace(13)* addrspace(10)* %i22 to {} addrspace(10)* addrspace(13)* addrspace(11)*

  %i24 = load {} addrspace(10)* addrspace(13)*, {} addrspace(10)* addrspace(13)* addrspace(11)* %i23, align 16

  %i26 = load atomic {} addrspace(10)*, {} addrspace(10)* addrspace(13)* %i24 unordered, align 8

  br i1 %.not26, label %L42.i, label %julia_sum_rec_1428.inner.exit

L42.i:                                            ; preds = %L38.i
  %i28 = icmp sgt i64 %i5, -1
  br i1 %i28, label %L66.i, label %L62.i

L62.i:                                            ; preds = %L132.i, %L50.i
  %value_phi3.i.lcssa = phi {} addrspace(10)* [ %i26, %L42.i ], [ null, %L101.i ]
  call void @jl_throw({} addrspace(10)* %value_phi3.i.lcssa) 
  unreachable

L66.i:                                            ; preds = %L132.i, %L66.i.lr.ph
  %i77 = call fastcc i64 @julia_ht_keyindex_1432({} addrspace(10)* null, {} addrspace(10)* nonnull %i26)
  %i49 = addrspacecast {} addrspace(10)* addrspace(13)* addrspace(10)* %i47 to {} addrspace(10)* addrspace(13)* addrspace(11)*
  %i50 = load {} addrspace(10)* addrspace(13)*, {} addrspace(10)* addrspace(13)* addrspace(11)* %i49, align 16
  %i52 = load atomic {} addrspace(10)*, {} addrspace(10)* addrspace(13)* %i50 unordered, align 8
  %i84 = call double @julia_sum_rec_1428.inner.1({} addrspace(10)* nocapture nonnull readonly align 8 dereferenceable(64) %i52)
  %.not17 = icmp sgt i64 %i5, 0
  %i72 = icmp sgt i64 %i5, -1
  br i1 %.not17, label %julia_sum_rec_1428.inner.exit, label %L101.i

L101.i:                                           ; preds = %L108.i, %L101.i.preheader
  br i1 %i72, label %L66.i, label %L62.i

julia_sum_rec_1428.inner.exit:                    ; preds = %pass9.i, %L120.i, %L108.i, %L87.i, %L38.i, %L26.i, %L5.i, %entry
  ret double 1.000000e+00
}

attributes #0 = { readnone "enzyme_inactive" }
attributes #1 = { noreturn }
attributes #2 = { inaccessiblememonly allocsize(1) }
attributes #3 = { readonly "enzyme_inactive" }
attributes #4 = { argmemonly nounwind }
attributes #5 = { readonly }
attributes #6 = { allocsize(1) }

; CHECK: define internal void @diffejulia_sum_rec_1428.inner.1
