; RUN: if [ %llvmver -ge 12 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -gvn -early-cse -instsimplify -simplifycfg -adce -S | FileCheck %s; fi
; RUN: if [ %llvmver -lt 12 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -gvn -early-cse -instsimplify -simplifycfg -adce -S | FileCheck %s --check-prefix=BEFORE; fi

define void @foo(float* noalias %out, float* noalias %in, i64* %x2.i.i, i1 %a9) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body.i.i.preheader.i.i.i.i.i.i59, %for.cond.cleanup4
  %s1.0235 = phi i64 [ 0, %entry ], [ %inc16, %for.cond.cleanup4 ]
  br label %for.body2

for.body2:                                       ; preds = %for.body, %_ZNK11OuterStruct11InnerStruct5sizexEv.exit.i
  %j = phi i64 [ %nextj, %merge ], [ 0, %for.body ]
  br i1 %a9, label %cond.false, label %merge

cond.false:                                   ; preds = %for.body.i
  %a14 = load i64, i64* %x2.i.i, align 8
  store i64 1, i64* %x2.i.i
  br label %merge

merge:    ; preds = %cond.false.i.i, %cond.true.i.i
  %cond = phi i64 [ 2, %for.body2 ], [ %a14, %cond.false ]
  %nextj = add i64 %j, 1
  %cmp = icmp eq i64 %nextj, 7
  br i1 %cmp, label %_ZNK11OuterStruct4sizeEv.exit, label %for.body2

_ZNK11OuterStruct4sizeEv.exit:                    ; preds = %_ZNK11OuterStruct11InnerStruct5sizexEv.exit.i, %for.body
  %s.0.lcssa.i = phi i64 [ %cond, %merge ]
  %cmp3.not233 = icmp eq i64 %s.0.lcssa.i, 0
  br i1 %cmp3.not233, label %for.cond.cleanup4, label %for.cond6.preheader

for.cond6.preheader:                              ; preds = %_ZNK11OuterStruct4sizeEv.exit, %for.cond6.preheader.split.split
  %i = phi i64 [ %nexti, %for.cond6.preheader ], [ 0, %_ZNK11OuterStruct4sizeEv.exit ]
  %a17 = load float, float* %in, align 8
  %sq = fmul float %a17, %a17
  store float %sq, float* %out, align 8
  %nexti = add nuw i64 %i, 1
  %cmp3.not = icmp eq i64 %nexti, %s.0.lcssa.i
  br i1 %cmp3.not, label %for.cond.cleanup4, label %for.cond6.preheader

for.cond.cleanup4:                                ; preds = %for.cond6.preheader.split.split, %_ZNK11OuterStruct4sizeEv.exit
  %inc16 = add nuw nsw i64 %s1.0235, 1
  %cmp.not = icmp eq i64 %inc16, 10
  br i1 %cmp.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.cond.cleanup4
  store float 0.000000e+00, float* %in, align 8
  ret void
}

; Function Attrs: nounwind uwtable
define void @caller(float* %in, float* %din, float* %outstr, float* %d_outstr, i64* %l, i1 %s) local_unnamed_addr {
entry:
  call void @__enzyme_autodiff(i8* bitcast (void (float*, float*, i64*, i1)* @foo to i8*), float* %in, float* %din, float* %outstr, float* %d_outstr, i64* %l, i1 %s) #11
  ret void
}

declare void @__enzyme_autodiff(i8*, float*, float*, float*, float*, i64* %l, i1)

; CHECK: define internal void @diffefoo(float* noalias %out, float* %"out'", float* noalias %in, float* %"in'", i64* %x2.i.i, i1 %a9)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %malloccall = tail call noalias nonnull dereferenceable(80) dereferenceable_or_null(80) i8* @malloc(i64 80)
; CHECK-NEXT:   %a17_malloccache = bitcast i8* %malloccall to float**
; CHECK-NEXT:   %[[malloccall9:.+]] = tail call noalias nonnull dereferenceable(80) dereferenceable_or_null(80) i8* @malloc(i64 80)
; CHECK-NEXT:   %cond.lcssa_malloccache = bitcast i8* %[[malloccall9]] to i64*
; CHECK-NEXT:   br label %for.body

; CHECK: for.body:                                         ; preds = %for.cond.cleanup4, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %for.cond.cleanup4 ], [ 0, %entry ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   br label %for.body2

; CHECK: for.body2:                                        ; preds = %merge, %for.body
; CHECK-NEXT:   %iv1 = phi i64 [ %iv.next2, %merge ], [ 0, %for.body ]
; CHECK-NEXT:   %iv.next2 = add nuw nsw i64 %iv1, 1
; CHECK-NEXT:   br i1 %a9, label %cond.false, label %merge

; CHECK: cond.false:                                       ; preds = %for.body2
; CHECK-NEXT:   %a14 = load i64, i64* %x2.i.i, align 8
; CHECK-NEXT:   store i64 1, i64* %x2.i.i, align 4
; CHECK-NEXT:   br label %merge

; CHECK: merge:                                            ; preds = %cond.false, %for.body2
; CHECK-NEXT:   %cond = phi i64 [ 2, %for.body2 ], [ %a14, %cond.false ]
; CHECK-NEXT:   %cmp = icmp eq i64 %iv.next2, 7
; CHECK-NEXT:   br i1 %cmp, label %_ZNK11OuterStruct4sizeEv.exit, label %for.body2

; CHECK: _ZNK11OuterStruct4sizeEv.exit:                    ; preds = %merge
; CHECK-NEXT:   %[[i0:.+]] = getelementptr inbounds i64, i64* %cond.lcssa_malloccache, i64 %iv
; CHECK-NEXT:   store i64 %cond, i64* %[[i0]], align 8, !invariant.group !0
; CHECK-NEXT:   %cmp3.not233 = icmp eq i64 %cond, 0
; CHECK-NEXT:   br i1 %cmp3.not233, label %for.cond.cleanup4, label %for.cond6.preheader.preheader

; CHECK: for.cond6.preheader.preheader:                    ; preds = %_ZNK11OuterStruct4sizeEv.exit
; CHECK-NEXT:   %[[i2:.+]] = getelementptr inbounds float*, float** %a17_malloccache, i64 %iv
; CHECK-NEXT:   %mallocsize = mul nuw nsw i64 %cond, 4
; CHECK-NEXT:   %[[malloccall5:.+]] = tail call noalias nonnull i8* @malloc(i64 %mallocsize)
; CHECK-NEXT:   %[[a17_malloccache6:.+]] = bitcast i8* %[[malloccall5]] to float*
; CHECK-NEXT:   store float* %[[a17_malloccache6]], float** %[[i2]], align 4, !invariant.group ![[g1:[0-9]+]]
; CHECK-NEXT:   %a17.pre = load float, float* %in, align 8
; CHECK-NEXT:   br label %for.cond6.preheader

; CHECK: for.cond6.preheader:                              ; preds = %for.cond6.preheader, %for.cond6.preheader.preheader
; CHECK-NEXT:   %iv3 = phi i64 [ %iv.next4, %for.cond6.preheader ], [ 0, %for.cond6.preheader.preheader ]
; CHECK-NEXT:   %iv.next4 = add nuw nsw i64 %iv3, 1
; CHECK-NEXT:   %sq = fmul float %a17.pre, %a17.pre
; CHECK-NEXT:   store float %sq, float* %out, align 8
; CHECK-NEXT:   %[[i3:.+]] = getelementptr inbounds float, float* %[[a17_malloccache6]], i64 %iv3
; CHECK-NEXT:   store float %a17.pre, float* %[[i3]], align 4, !invariant.group ![[g2:[0-9]+]]
; CHECK-NEXT:   %cmp3.not = icmp eq i64 %iv.next4, %cond
; CHECK-NEXT:   br i1 %cmp3.not, label %for.cond.cleanup4, label %for.cond6.preheader

; CHECK: for.cond.cleanup4:                                ; preds = %for.cond6.preheader, %_ZNK11OuterStruct4sizeEv.exit
; CHECK-NEXT:   %cmp.not = icmp eq i64 %iv.next, 10
; CHECK-NEXT:   br i1 %cmp.not, label %for.cond.cleanup, label %for.body

; CHECK: for.cond.cleanup:                                 ; preds = %for.cond.cleanup4
; CHECK-NEXT:   store float 0.000000e+00, float* %in, align 8
; CHECK-NEXT:   store float 0.000000e+00, float* %"in'", align 8
; CHECK-NEXT:   br label %invertfor.cond.cleanup4

; CHECK: invertentry:                                      ; preds = %invertfor.body
; CHECK-NEXT:   tail call void @free(i8* nonnull %malloccall)
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[malloccall9]])
; CHECK-NEXT:   ret void

; CHECK: invertfor.body:                                   ; preds = %invertmerge
; CHECK-NEXT:   %[[i4:.+]] = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %[[i4]], label %invertentry, label %incinvertfor.body

; CHECK: incinvertfor.body:                                ; preds = %invertfor.body
; CHECK-NEXT:   %[[i5:.+]] = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertfor.cond.cleanup4

; CHECK: incinvertfor.body2:                               ; preds = %invertmerge
; CHECK-NEXT:   %[[i6:.+]] = add nsw i64 %"iv1'ac.0", -1
; CHECK-NEXT:   br label %invertmerge

; CHECK: invertmerge:                                      ; preds = %invert_ZNK11OuterStruct4sizeEv.exit, %incinvertfor.body2
; CHECK-NEXT:   %"iv1'ac.0" = phi i64 [ 6, %invert_ZNK11OuterStruct4sizeEv.exit ], [ %[[i6]], %incinvertfor.body2 ]
; CHECK-NEXT:   %[[i7:.+]] = icmp eq i64 %"iv1'ac.0", 0
; CHECK-NEXT:   br i1 %[[i7]], label %invertfor.body, label %incinvertfor.body2

; CHECK: invert_ZNK11OuterStruct4sizeEv.exit:              ; preds = %invertfor.cond.cleanup4, %invertfor.cond6.preheader.preheader
; CHECK-NEXT:   %"a17'de.0" = phi float [ %"a17'de.2", %invertfor.cond.cleanup4 ], [ 0.000000e+00, %invertfor.cond6.preheader.preheader ]
; CHECK-NEXT:   %"sq'de.0" = phi float [ %"sq'de.2", %invertfor.cond.cleanup4 ], [ 0.000000e+00, %invertfor.cond6.preheader.preheader ]
; CHECK-NEXT:   br label %invertmerge

; CHECK: invertfor.cond6.preheader.preheader:              ; preds = %invertfor.cond6.preheader
; CHECK-NEXT:   %[[i8:.+]] = bitcast float* %.pre to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[i8]])
; CHECK-NEXT:   br label %invert_ZNK11OuterStruct4sizeEv.exit

; CHECK: invertfor.cond6.preheader:                        ; preds = %invertfor.cond.cleanup4.loopexit, %incinvertfor.cond6.preheader
; CHECK-NEXT:   %"a17'de.1" = phi float [ %"a17'de.2", %invertfor.cond.cleanup4.loopexit ], [ 0.000000e+00, %incinvertfor.cond6.preheader ]
; CHECK-NEXT:   %"sq'de.1" = phi float [ %"sq'de.2", %invertfor.cond.cleanup4.loopexit ], [ 0.000000e+00, %incinvertfor.cond6.preheader ]
; CHECK-NEXT:   %"iv3'ac.0" = phi i64 [ %[[_unwrap12:.+]], %invertfor.cond.cleanup4.loopexit ], [ %[[i18:.+]], %incinvertfor.cond6.preheader ]
; CHECK-NEXT:   %[[i9:.+]] = load float, float* %"out'", align 8
; CHECK-NEXT:   store float 0.000000e+00, float* %"out'", align 8
; CHECK-NEXT:   %[[i10:.+]] = fadd fast float %"sq'de.1", %[[i9]]
; CHECK-NEXT:   %[[i11:.+]] = getelementptr inbounds float, float* %.pre, i64 %"iv3'ac.0"
; CHECK-NEXT:   %[[i12:.+]] = load float, float* %[[i11]], align 4, !invariant.group ![[g2]]
; CHECK-NEXT:   %m0diffea17 = fmul fast float %[[i10]], %[[i12]]
; CHECK-NEXT:   %[[i13:.+]] = fadd fast float %"a17'de.1", %m0diffea17
; CHECK-NEXT:   %[[i14:.+]] = fadd fast float %[[i13]], %m0diffea17
; CHECK-NEXT:   %[[i15:.+]] = load float, float* %"in'", align 8
; CHECK-NEXT:   %[[i16:.+]] = fadd fast float %[[i15]], %[[i14]]
; CHECK-NEXT:   store float %[[i16:.+]], float* %"in'", align 8
; CHECK-NEXT:   %[[i17:.+]] = icmp eq i64 %"iv3'ac.0", 0
; CHECK-NEXT:   br i1 %[[i17]], label %invertfor.cond6.preheader.preheader, label %incinvertfor.cond6.preheader

; CHECK: incinvertfor.cond6.preheader:                     ; preds = %invertfor.cond6.preheader
; CHECK-NEXT:   %[[i18]] = add nsw i64 %"iv3'ac.0", -1
; CHECK-NEXT:   br label %invertfor.cond6.preheader

; CHECK: invertfor.cond.cleanup4.loopexit:                 ; preds = %invertfor.cond.cleanup4
; CHECK-NEXT:   %[[_unwrap12]] = add i64 %[[i22:.+]], -1
; CHECK-NEXT:   %.phi.trans.insert = getelementptr inbounds float*, float** %a17_malloccache, i64 %"iv'ac.0"
; CHECK-NEXT:   %.pre = load float*, float** %.phi.trans.insert, align 8, !invariant.group ![[g1]]
; CHECK-NEXT:   br label %invertfor.cond6.preheader

; CHECK: invertfor.cond.cleanup4:                          ; preds = %for.cond.cleanup, %incinvertfor.body
; CHECK-NEXT:   %"a17'de.2" = phi float [ 0.000000e+00, %for.cond.cleanup ], [ %"a17'de.0", %incinvertfor.body ]
; CHECK-NEXT:   %"sq'de.2" = phi float [ 0.000000e+00, %for.cond.cleanup ], [ %"sq'de.0", %incinvertfor.body ]
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ 9, %for.cond.cleanup ], [ %[[i5]], %incinvertfor.body ]
; CHECK-NEXT:   %[[i19:.+]] = getelementptr inbounds i64, i64* %cond.lcssa_malloccache, i64 %"iv'ac.0"
; CHECK-NEXT:   %[[i22]] = load i64, i64* %[[i19]], align 8, !invariant.group !0
; CHECK-NEXT:   %cmp3.not233_unwrap = icmp eq i64 %[[i22]], 0
; CHECK-NEXT:   br i1 %cmp3.not233_unwrap, label %invert_ZNK11OuterStruct4sizeEv.exit, label %invertfor.cond.cleanup4.loopexit
; CHECK-NEXT: }

; BEFORE: define internal void @diffefoo(float* noalias %out, float* %"out'", float* noalias %in, float* %"in'", i64* %x2.i.i, i1 %a9)
; BEFORE-NEXT: entry:
; BEFORE-NEXT:   %malloccall = tail call noalias nonnull dereferenceable(80) dereferenceable_or_null(80) i8* @malloc(i64 80)
; BEFORE-NEXT:   %a17_malloccache = bitcast i8* %malloccall to float**
; BEFORE-NEXT:   %[[malloccall12:.+]] = tail call noalias nonnull dereferenceable(80) dereferenceable_or_null(80) i8* @malloc(i64 80)
; BEFORE-NEXT:   %[[a14manual_lcssa10_malloccache:.+]] = bitcast i8* %[[malloccall12]] to i64*
; BEFORE-NEXT:   %[[malloccall17:.+]] = tail call noalias nonnull dereferenceable(80) dereferenceable_or_null(80) i8* @malloc(i64 80)
; BEFORE-NEXT:   %[[condmanual_lcssa16_malloccache:.+]] = bitcast i8* %[[malloccall17]] to i64*
; BEFORE-NEXT:   %[[malloccall20:.+]] = tail call noalias nonnull dereferenceable(80) dereferenceable_or_null(80) i8* @malloc(i64 80)
; BEFORE-NEXT:   %[[s0lcssa:.+]] = bitcast i8* %[[malloccall20]] to i64*
; BEFORE-NEXT:   br label %for.body

; BEFORE: for.body:                                         ; preds = %for.cond.cleanup4, %entry
; BEFORE-NEXT:   %iv = phi i64 [ %iv.next, %for.cond.cleanup4 ], [ 0, %entry ]
; BEFORE-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; BEFORE-NEXT:   br label %for.body2

; BEFORE: for.body2:                                        ; preds = %merge, %for.body
; BEFORE-NEXT:   %iv1 = phi i64 [ %iv.next2, %merge ], [ 0, %for.body ]
; BEFORE-NEXT:   %iv.next2 = add nuw nsw i64 %iv1, 1
; BEFORE-NEXT:   br i1 %a9, label %cond.false, label %merge

; BEFORE: cond.false:                                       ; preds = %for.body2
; BEFORE-NEXT:   %a14 = load i64, i64* %x2.i.i, align 8
; BEFORE-NEXT:   store i64 1, i64* %x2.i.i
; BEFORE-NEXT:   br label %merge

; BEFORE: merge:                                            ; preds = %cond.false, %for.body2
; BEFORE-NEXT:   %[[a14manual_lcssa11:.+]] = phi i64 [ %a14, %cond.false ], [ undef, %for.body2 ]
; BEFORE-NEXT:   %cond = phi i64 [ 2, %for.body2 ], [ %a14, %cond.false ]
; BEFORE-NEXT:   %cmp = icmp eq i64 %iv.next2, 7
; BEFORE-NEXT:   br i1 %cmp, label %_ZNK11OuterStruct4sizeEv.exit, label %for.body2

; BEFORE: _ZNK11OuterStruct4sizeEv.exit:                    ; preds = %merge
; BEFORE-NEXT:   %0 = getelementptr inbounds i64, i64* %[[a14manual_lcssa10_malloccache]], i64 %iv
; BEFORE-NEXT:   store i64 %[[a14manual_lcssa11]], i64* %0, align 8, !invariant.group !0
; BEFORE-NEXT:   %cmp3.not233 = icmp eq i64 %cond, 0
; BEFORE-NEXT:   br i1 %cmp3.not233, label %for.cond.cleanup4, label %for.cond6.preheader.preheader

; BEFORE: for.cond6.preheader.preheader:                    ; preds = %_ZNK11OuterStruct4sizeEv.exit
; BEFORE-NEXT:   %[[i2:.+]] = getelementptr inbounds float*, float** %a17_malloccache, i64 %iv
; BEFORE-NEXT:   %mallocsize = mul nuw nsw i64 %cond, 4
; BEFORE-NEXT:   %[[malloccall5:.+]] = tail call noalias nonnull i8* @malloc(i64 %mallocsize)
; BEFORE-NEXT:   %[[a17_malloccache6:.+]] = bitcast i8* %[[malloccall5]] to float*
; BEFORE-NEXT:   store float* %[[a17_malloccache6]], float** %[[i2]], align 4, !invariant.group ![[g1:[0-9]+]]
; BEFORE-NEXT:   %a17.pre = load float, float* %in, align 8
; BEFORE-NEXT:   br label %for.cond6.preheader

; BEFORE: for.cond6.preheader:                              ; preds = %for.cond6.preheader, %for.cond6.preheader.preheader
; BEFORE-NEXT:   %iv3 = phi i64 [ %iv.next4, %for.cond6.preheader ], [ 0, %for.cond6.preheader.preheader ] 
; BEFORE-NEXT:   %iv.next4 = add nuw nsw i64 %iv3, 1
; BEFORE-NEXT:   %sq = fmul float %a17.pre, %a17.pre
; BEFORE-NEXT:   store float %sq, float* %out, align 8
; BEFORE-NEXT:   %[[i3:.+]] = getelementptr inbounds float, float* %[[a17_malloccache6]], i64 %iv3
; BEFORE-NEXT:   store float %a17.pre, float* %[[i3]], align 4, !invariant.group ![[g2:[0-9]+]]
; BEFORE-NEXT:   %cmp3.not = icmp eq i64 %iv.next4, %cond
; BEFORE-NEXT:   br i1 %cmp3.not, label %for.cond.cleanup4.loopexit, label %for.cond6.preheader

; BEFORE: for.cond.cleanup4.loopexit:                       ; preds = %for.cond6.preheader
; BEFORE-NEXT:   %[[i4:.+]] = getelementptr inbounds i64, i64* %[[condmanual_lcssa16_malloccache]], i64 %iv
; BEFORE-NEXT:   store i64 %cond, i64* %[[i4]], align 8, !invariant.group ![[g3:[0-9]+]]
; BEFORE-NEXT:   br label %for.cond.cleanup4

; BEFORE: for.cond.cleanup4:                                ; preds = %for.cond.cleanup4.loopexit, %_ZNK11OuterStruct4sizeEv.exit
; BEFORE-NEXT:   %[[cond_lcssa17:.+]] = phi i64 [ %cond, %for.cond.cleanup4.loopexit ], [ 0, %_ZNK11OuterStruct4sizeEv.exit ]
; BEFORE-NEXT:   %[[ii1:.+]] = getelementptr inbounds i64, i64* %[[s0lcssa]], i64 %iv
; BEFORE-NEXT:   store i64 %[[cond_lcssa17]], i64* %[[ii1]], align 8, !invariant.group ![[g4:[0-9]+]]
; BEFORE-NEXT:   %cmp.not = icmp eq i64 %iv.next, 10
; BEFORE-NEXT:   br i1 %cmp.not, label %for.cond.cleanup, label %for.body

; BEFORE: for.cond.cleanup:                                 ; preds = %for.cond.cleanup4
; BEFORE-NEXT:   store float 0.000000e+00, float* %in, align 8
; BEFORE-NEXT:   store float 0.000000e+00, float* %"in'", align 8
; BEFORE-NEXT:   br label %invertfor.cond.cleanup4

; BEFORE: invertentry:                                      ; preds = %invertfor.body
; BEFORE-NEXT:   tail call void @free(i8* nonnull %malloccall)
; BEFORE-NEXT:   tail call void @free(i8* nonnull %[[malloccall12]])
; BEFORE-NEXT:   tail call void @free(i8* nonnull %[[malloccall17]])
; BEFORE-NEXT:   tail call void @free(i8* nonnull %[[malloccall20]])
; BEFORE-NEXT:   ret void

; BEFORE: invertfor.body:                                   ; preds = %invertmerge
; BEFORE-NEXT:   %5 = icmp eq i64 %"iv'ac.0", 0
; BEFORE-NEXT:   br i1 %5, label %invertentry, label %incinvertfor.body

; BEFORE: incinvertfor.body:                                ; preds = %invertfor.body
; BEFORE-NEXT:   %6 = add nsw i64 %"iv'ac.0", -1
; BEFORE-NEXT:   br label %invertfor.cond.cleanup4

; BEFORE: incinvertfor.body2:                               ; preds = %invertmerge
; BEFORE-NEXT:   %7 = add nsw i64 %"iv1'ac.0", -1
; BEFORE-NEXT:   br label %invertmerge

; BEFORE: invertmerge:                                      ; preds = %invert_ZNK11OuterStruct4sizeEv.exit, %incinvertfor.body2
; BEFORE-NEXT:   %"iv1'ac.0" = phi i64 [ 6, %invert_ZNK11OuterStruct4sizeEv.exit ], [ %7, %incinvertfor.body2 ]
; BEFORE-NEXT:   %8 = icmp eq i64 %"iv1'ac.0", 0
; BEFORE-NEXT:   br i1 %8, label %invertfor.body, label %incinvertfor.body2

; BEFORE: invert_ZNK11OuterStruct4sizeEv.exit:              ; preds = %invertfor.cond.cleanup4, %invertfor.cond6.preheader.preheader
; BEFORE-NEXT:   %"a17'de.0" = phi float [ %"a17'de.2", %invertfor.cond.cleanup4 ], [ 0.000000e+00, %invertfor.cond6.preheader.preheader ]
; BEFORE-NEXT:   %"sq'de.0" = phi float [ %"sq'de.2", %invertfor.cond.cleanup4 ], [ 0.000000e+00, %invertfor.cond6.preheader.preheader ]
; BEFORE-NEXT:   br label %invertmerge

; BEFORE: invertfor.cond6.preheader.preheader:              ; preds = %invertfor.cond6.preheader
; BEFORE-NEXT:   %9 = bitcast float* %13 to i8*
; BEFORE-NEXT:   tail call void @free(i8* nonnull %9)
; BEFORE-NEXT:   br label %invert_ZNK11OuterStruct4sizeEv.exit

; BEFORE: invertfor.cond6.preheader:                        ; preds = %invertfor.cond.cleanup4.loopexit, %incinvertfor.cond6.preheader
; BEFORE-NEXT:   %"a17'de.1" = phi float [ %"a17'de.2", %invertfor.cond.cleanup4.loopexit ], [ 0.000000e+00, %incinvertfor.cond6.preheader ]
; BEFORE-NEXT:   %"sq'de.1" = phi float [ %"sq'de.2", %invertfor.cond.cleanup4.loopexit ], [ 0.000000e+00, %incinvertfor.cond6.preheader ]
; BEFORE-NEXT:   %"iv3'ac.0" = phi i64 [ %[[_unwrap19:.+]], %invertfor.cond.cleanup4.loopexit ], [ %21, %incinvertfor.cond6.preheader ]
; BEFORE-NEXT:   %10 = load float, float* %"out'", align 8
; BEFORE-NEXT:   store float 0.000000e+00, float* %"out'", align 8
; BEFORE-NEXT:   %11 = fadd fast float %"sq'de.1", %10
; BEFORE-NEXT:   %12 = getelementptr inbounds float*, float** %a17_malloccache, i64 %"iv'ac.0"
; BEFORE-NEXT:   %13 = load float*, float** %12, align 8, !dereferenceable !{{[0-9]+}}, !invariant.group ![[g1]]
; BEFORE-NEXT:   %14 = getelementptr inbounds float, float* %13, i64 %"iv3'ac.0"
; BEFORE-NEXT:   %15 = load float, float* %14, align 4, !invariant.group ![[g2]]
; BEFORE-NEXT:   %m0diffea17 = fmul fast float %11, %15
; BEFORE-NEXT:   %16 = fadd fast float %"a17'de.1", %m0diffea17
; BEFORE-NEXT:   %17 = fadd fast float %16, %m0diffea17
; BEFORE-NEXT:   %18 = load float, float* %"in'", align 8
; BEFORE-NEXT:   %19 = fadd fast float %18, %17
; BEFORE-NEXT:   store float %19, float* %"in'", align 8
; BEFORE-NEXT:   %20 = icmp eq i64 %"iv3'ac.0", 0
; BEFORE-NEXT:   br i1 %20, label %invertfor.cond6.preheader.preheader, label %incinvertfor.cond6.preheader

; BEFORE: incinvertfor.cond6.preheader:                     ; preds = %invertfor.cond6.preheader
; BEFORE-NEXT:   %21 = add nsw i64 %"iv3'ac.0", -1
; BEFORE-NEXT:   br label %invertfor.cond6.preheader

; BEFORE: invertfor.cond.cleanup4.loopexit:                 ; preds = %invertfor.cond.cleanup4
; BEFORE-NEXT:   %22 = getelementptr inbounds i64, i64* %[[condmanual_lcssa16_malloccache]], i64 %"iv'ac.0"
; BEFORE-NEXT:   %23 = load i64, i64* %22, align 8, !invariant.group ![[g3]]
; BEFORE-NEXT:   %[[_unwrap19]] = add i64 %23, -1
; BEFORE-NEXT:   br label %invertfor.cond6.preheader

; BEFORE: invertfor.cond.cleanup4:                          ; preds = %for.cond.cleanup, %incinvertfor.body
; BEFORE-NEXT:   %"a17'de.2" = phi float [ 0.000000e+00, %for.cond.cleanup ], [ %"a17'de.0", %incinvertfor.body ]
; BEFORE-NEXT:   %"sq'de.2" = phi float [ 0.000000e+00, %for.cond.cleanup ], [ %"sq'de.0", %incinvertfor.body ]
; BEFORE-NEXT:   %"iv'ac.0" = phi i64 [ 9, %for.cond.cleanup ], [ %6, %incinvertfor.body ]
; BEFORE-NEXT:   %24 = getelementptr inbounds i64, i64* %[[s0lcssa]], i64 %"iv'ac.0"
; BEFORE-NEXT:   %25 = load i64, i64* %24, align 8, !invariant.group ![[g4]]
; BEFORE-NEXT:   %cmp3.not233_unwrap = icmp eq i64 %25, 0
; BEFORE-NEXT:   br i1 %cmp3.not233_unwrap, label %invert_ZNK11OuterStruct4sizeEv.exit, label %invertfor.cond.cleanup4.loopexit
; BEFORE-NEXT: }
