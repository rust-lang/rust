; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -inline -mem2reg -instsimplify -adce -loop-deletion -correlated-propagation -simplifycfg -adce -S | FileCheck %s

source_filename = "/home/wmoses/git/Enzyme/enzyme/lulesh/RAJAProxies/lulesh-v2.0/RAJA/lulesh.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%"struct.lulesh2::MemoryPool" = type { [32 x double*], [32 x i32] }
%struct.ident_t = type { i32, i32, i32, i32, i8* }
%"class.RAJA::TypedIndexSet" = type { %"class.RAJA::TypedIndexSet.0", %"class.RAJA::RAJAVec.11", %"class.RAJA::RAJAVec", %"class.RAJA::RAJAVec", %"class.RAJA::RAJAVec" }
%"class.RAJA::TypedIndexSet.0" = type { %"class.RAJA::TypedIndexSet.1", %"class.RAJA::RAJAVec.7", %"class.RAJA::RAJAVec", %"class.RAJA::RAJAVec", %"class.RAJA::RAJAVec" }
%"class.RAJA::TypedIndexSet.1" = type { %"class.RAJA::TypedIndexSet.2", %"class.RAJA::RAJAVec.3", %"class.RAJA::RAJAVec", %"class.RAJA::RAJAVec", %"class.RAJA::RAJAVec" }
%"class.RAJA::TypedIndexSet.2" = type { %"class.RAJA::RAJAVec", %"class.RAJA::RAJAVec", %"class.RAJA::RAJAVec", i64 }
%"class.RAJA::RAJAVec.3" = type { %"struct.RAJA::TypedRangeStrideSegment"**, %"class.std::allocator.4", i64, i64 }
%"struct.RAJA::TypedRangeStrideSegment" = type { %"class.RAJA::Iterators::strided_numeric_iterator", %"class.RAJA::Iterators::strided_numeric_iterator", i64 }
%"class.RAJA::Iterators::strided_numeric_iterator" = type { i64, i64 }
%"class.std::allocator.4" = type { i8 }
%"class.RAJA::RAJAVec.7" = type { %"class.RAJA::TypedListSegment"**, %"class.std::allocator.8", i64, i64 }
%"class.RAJA::TypedListSegment" = type { %"class.camp::resources::v1::Resource", i8, i32, i64*, i64 }
%"class.camp::resources::v1::Resource" = type { %"class.std::shared_ptr" }
%"class.std::shared_ptr" = type { %"class.std::__shared_ptr" }
%"class.std::__shared_ptr" = type { %"class.camp::resources::v1::Resource::ContextInterface"*, %"class.std::__shared_count" }
%"class.camp::resources::v1::Resource::ContextInterface" = type { i32 (...)** }
%"class.std::__shared_count" = type { %"class.std::_Sp_counted_base"* }
%"class.std::_Sp_counted_base" = type { i32 (...)**, i32, i32 }
%"class.std::allocator.8" = type { i8 }
%"class.RAJA::RAJAVec.11" = type { %"struct.RAJA::TypedRangeSegment"**, %"class.std::allocator.12", i64, i64 }
%"struct.RAJA::TypedRangeSegment" = type { %"class.RAJA::Iterators::numeric_iterator", %"class.RAJA::Iterators::numeric_iterator" }
%"class.RAJA::Iterators::numeric_iterator" = type { i64 }
%"class.std::allocator.12" = type { i8 }
%"class.RAJA::RAJAVec" = type { i64*, %"class.std::allocator", i64, i64 }
%"class.std::allocator" = type { i8 }

@elemMemPool = dso_local local_unnamed_addr global %"struct.lulesh2::MemoryPool" zeroinitializer, align 8
@0 = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
@1 = private unnamed_addr constant %struct.ident_t { i32 0, i32 514, i32 0, i32 0, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @0, i32 0, i32 0) }, align 8
@2 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @0, i32 0, i32 0) }, align 8

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture)

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture)

define void @caller(i8* %in, i8* %din) local_unnamed_addr {
entry:
  call void (i8*, ...) @_Z17__enzyme_autodiffPvS_S_(i8* bitcast (void (i8*)* @_ZL16LagrangeLeapFrogRN4RAJA13TypedIndexSetIJNS_17TypedRangeSegmentIllEENS_16TypedListSegmentIlEENS_23TypedRangeStrideSegmentIllEEEEE to i8*), metadata !"enzyme_dup", i8* %in, i8* nonnull %din) #5
  ret void
}

declare void @_Z17__enzyme_autodiffPvS_S_(i8*, ...) 

; Function Attrs: inlinehint nounwind uwtable mustprogress
define internal void @_ZL16LagrangeLeapFrogRN4RAJA13TypedIndexSetIJNS_17TypedRangeSegmentIllEENS_16TypedListSegmentIlEENS_23TypedRangeStrideSegmentIllEEEEE(i8* %iset) {
entry:
  tail call fastcc void @_ZL28CalcHourglassControlForElemsRN4RAJA13TypedIndexSetIJNS_17TypedRangeSegmentIllEENS_16TypedListSegmentIlEENS_23TypedRangeStrideSegmentIllEEEEE(i8* %iset)
  br label %for.body.i

for.cond.i:                                       ; preds = %for.body.i
  %indvars.iv.next.i = add nuw nsw i64 %indvars.iv.i, 1
  %exitcond.not.i = icmp eq i64 %indvars.iv.next.i, 32
  br i1 %exitcond.not.i, label %_ZN7lulesh210MemoryPoolIdE7releaseEPPd.exit, label %for.body.i, !llvm.loop !4

for.body.i:                                       ; preds = %for.cond.i, %entry
  %indvars.iv.i = phi i64 [ 0, %entry ], [ %indvars.iv.next.i, %for.cond.i ]
  %arrayidx.i = getelementptr inbounds %"struct.lulesh2::MemoryPool", %"struct.lulesh2::MemoryPool"* @elemMemPool, i64 0, i32 0, i64 %indvars.iv.i
  %0 = load double*, double** %arrayidx.i, align 8, !tbaa !7
  %cmp2.i = icmp eq double* %0, null
  br i1 %cmp2.i, label %if.then.i, label %for.cond.i

if.then.i:                                        ; preds = %for.body.i
  %idxprom.le.i = and i64 %indvars.iv.i, 4294967295
  %arrayidx4.i = getelementptr inbounds %"struct.lulesh2::MemoryPool", %"struct.lulesh2::MemoryPool"* @elemMemPool, i64 0, i32 1, i64 %idxprom.le.i
  %1 = load i32, i32* %arrayidx4.i, align 4, !tbaa !11
  %sub.i = sub nsw i32 0, %1
  store i32 %sub.i, i32* %arrayidx4.i, align 4, !tbaa !11
  br label %_ZN7lulesh210MemoryPoolIdE7releaseEPPd.exit

_ZN7lulesh210MemoryPoolIdE7releaseEPPd.exit:      ; preds = %if.then.i, %for.cond.i
  ret void
}

; Function Attrs: noinline nounwind uwtable mustprogress
define internal fastcc void @_ZL28CalcHourglassControlForElemsRN4RAJA13TypedIndexSetIJNS_17TypedRangeSegmentIllEENS_16TypedListSegmentIlEENS_23TypedRangeStrideSegmentIllEEEEE(i8* %i2) unnamed_addr #3 {
entry:
  %distance_it = alloca i64, align 8
  %CONTAINER.sroa.0.0..sroa_cast14 = bitcast i8* %i2 to i64*
  %sub.i.i.i = load i64, i64* %CONTAINER.sroa.0.0..sroa_cast14, align 8, !tbaa.struct !17
  %i3 = bitcast i64* %distance_it to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %i3) #5
  store i64 %sub.i.i.i, i64* %distance_it, align 8, !tbaa !18
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %i3) #5
  ret void

for.body:                                         ; preds = %for.body, %entry
  %segid.018 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%struct.ident_t* nonnull @2, i32 1, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, i64*)* @.omp_outlined. to void (i32*, i32*, ...)*), i64* nonnull %distance_it)
  %inc = add nuw nsw i32 %segid.018, 1
  %exitcond.not = icmp eq i32 %inc, 20
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body, !llvm.loop !20
}

; Function Attrs: norecurse nounwind uwtable
define internal void @.omp_outlined.(i32* noalias nocapture readonly %.global_tid., i32* noalias nocapture readnone %.bound_tid., i64* nocapture nonnull readonly align 8 dereferenceable(8) %distance_it) #4 {
entry:
  %.omp.lb = alloca i64, align 8
  %.omp.ub = alloca i64, align 8
  %.omp.stride = alloca i64, align 8
  %.omp.is_last = alloca i32, align 4
  %0 = load i64, i64* %distance_it, align 8, !tbaa !18
  %sub2 = add nsw i64 %0, -1
  %1 = bitcast i64* %.omp.lb to i8*
  store i64 0, i64* %.omp.lb, align 8, !tbaa !18
  %2 = bitcast i64* %.omp.ub to i8*
  store i64 %sub2, i64* %.omp.ub, align 8, !tbaa !18
  %3 = bitcast i64* %.omp.stride to i8*
  store i64 1, i64* %.omp.stride, align 8, !tbaa !18
  %4 = bitcast i32* %.omp.is_last to i8*
  store i32 0, i32* %.omp.is_last, align 4, !tbaa !11
  %5 = load i32, i32* %.global_tid., align 4, !tbaa !11
  call void @__kmpc_for_static_init_8(%struct.ident_t* nonnull @1, i32 %5, i32 34, i32* nonnull %.omp.is_last, i64* nonnull %.omp.lb, i64* nonnull %.omp.ub, i64* nonnull %.omp.stride, i64 1, i64 1) #5
  %6 = load i64, i64* %.omp.ub, align 8, !tbaa !18
  %cmp4.not = icmp slt i64 %6, %0
  %cond = select i1 %cmp4.not, i64 %6, i64 %sub2
  store i64 %cond, i64* %.omp.ub, align 8, !tbaa !18
  %7 = load i64, i64* %.omp.lb, align 8, !tbaa !18
  br label %omp.inner.for.cond

omp.inner.for.cond:                               ; preds = %omp.inner.for.cond, %omp.precond.then
  %.omp.iv.0 = phi i64 [ %7, %entry ], [ %add6, %omp.inner.for.cond ]
  %cmp5.not = icmp sgt i64 %.omp.iv.0, %cond
  %add6 = add nsw i64 %.omp.iv.0, 1
  br i1 %cmp5.not, label %omp.loop.exit, label %omp.inner.for.cond

omp.loop.exit:                                    ; preds = %omp.inner.for.cond
  call void @__kmpc_for_static_fini(%struct.ident_t* nonnull @1, i32 %5)
  ret void
}

; Function Attrs: nounwind
declare void @__kmpc_for_static_init_8(%struct.ident_t*, i32, i32, i32*, i64*, i64*, i64*, i64, i64) local_unnamed_addr #5

; Function Attrs: nounwind
declare void @__kmpc_for_static_fini(%struct.ident_t*, i32) local_unnamed_addr #5

; Function Attrs: nounwind
declare void @__kmpc_fork_call(%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) local_unnamed_addr #5

attributes #3 = { noinline }
attributes #4 = { norecurse nounwind uwtable }
attributes #5 = { nounwind }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}
!nvvm.annotations = !{}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{!"clang version 12.0.1 (git@github.com:llvm/llvm-project 4973ce53ca8abfc14233a3d8b3045673e0e8543c)"}
!4 = distinct !{!4, !5, !6}
!5 = !{!"llvm.loop.mustprogress"}
!6 = !{!"llvm.loop.unroll.disable"}
!7 = !{!8, !8, i64 0}
!8 = !{!"any pointer", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C++ TBAA"}
!11 = !{!12, !12, i64 0}
!12 = !{!"int", !9, i64 0}
!13 = !{!14, !8, i64 0}
!14 = !{!"_ZTSN4RAJA7RAJAVecIPNS_17TypedRangeSegmentIllEESaIS3_EEE", !8, i64 0, !15, i64 8, !16, i64 16, !16, i64 24}
!15 = !{!"_ZTSSaIPN4RAJA17TypedRangeSegmentIllEEE"}
!16 = !{!"long", !9, i64 0}
!17 = !{i64 0, i64 8, !18, i64 8, i64 8, !18}
!18 = !{!16, !16, i64 0}
!19 = !{i64 0, i64 8, !18}
!20 = distinct !{!20, !5, !6}
!21 = !{!22}
!22 = !{i64 2, i64 -1, i64 -1, i1 true}

; CHECK: define internal void @diffe.omp_outlined
