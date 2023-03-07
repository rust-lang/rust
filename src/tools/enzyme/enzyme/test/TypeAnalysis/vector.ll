; RUN: %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=reduce_max -o /dev/null | FileCheck %s

; ModuleID = 'test.c'
source_filename = "test.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@.str = private unnamed_addr constant [14 x i8] c"reduce_max=%f\00", align 1
@.str.1 = private unnamed_addr constant [20 x i8] c"d_reduce_max(%i)=%f\00", align 1

; Function Attrs: norecurse nounwind readonly uwtable
define dso_local void @reduce_max(double* nocapture readonly %vec, i32 %size) #0 {
entry:
  %cmp12 = icmp sgt i32 %size, 0
  br i1 %cmp12, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %size to i64
  %min.iters.check = icmp ult i32 %size, 4
  br i1 %min.iters.check, label %for.body.preheader20, label %vector.ph

for.body.preheader20:                             ; preds = %middle.block, %for.body.preheader
  %indvars.iv.ph = phi i64 [ 0, %for.body.preheader ], [ %n.vec, %middle.block ]
  %ret.013.ph = phi double [ 0xFFF0000000000000, %for.body.preheader ], [ %28, %middle.block ]
  br label %for.body

vector.ph:                                        ; preds = %for.body.preheader
  %n.vec = and i64 %wide.trip.count, 4294967292
  %0 = add nsw i64 %n.vec, -4
  %1 = lshr exact i64 %0, 2
  %2 = add nuw nsw i64 %1, 1
  %xtraiter = and i64 %2, 1
  %3 = icmp eq i64 %0, 0
  br i1 %3, label %middle.block.unr-lcssa, label %vector.ph.new

vector.ph.new:                                    ; preds = %vector.ph
  %unroll_iter = sub nsw i64 %2, %xtraiter
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph.new
  %index = phi i64 [ 0, %vector.ph.new ], [ %index.next.1, %vector.body ]
  %vec.phi = phi <2 x double> [ <double 0xFFF0000000000000, double 0xFFF0000000000000>, %vector.ph.new ], [ %18, %vector.body ]
  %vec.phi16 = phi <2 x double> [ <double 0xFFF0000000000000, double 0xFFF0000000000000>, %vector.ph.new ], [ %19, %vector.body ]
  %niter = phi i64 [ %unroll_iter, %vector.ph.new ], [ %niter.nsub.1, %vector.body ]
  %4 = getelementptr inbounds double, double* %vec, i64 %index
  %5 = bitcast double* %4 to <2 x double>*
  %wide.load = load <2 x double>, <2 x double>* %5, align 8, !tbaa !2
  %6 = getelementptr inbounds double, double* %4, i64 2
  %7 = bitcast double* %6 to <2 x double>*
  %wide.load17 = load <2 x double>, <2 x double>* %7, align 8, !tbaa !2
  %8 = fcmp fast ogt <2 x double> %vec.phi, %wide.load
  %9 = fcmp fast ogt <2 x double> %vec.phi16, %wide.load17
  %10 = select <2 x i1> %8, <2 x double> %vec.phi, <2 x double> %wide.load
  %11 = select <2 x i1> %9, <2 x double> %vec.phi16, <2 x double> %wide.load17
  %index.next = or i64 %index, 4
  %12 = getelementptr inbounds double, double* %vec, i64 %index.next
  %13 = bitcast double* %12 to <2 x double>*
  %wide.load.1 = load <2 x double>, <2 x double>* %13, align 8, !tbaa !2
  %14 = getelementptr inbounds double, double* %12, i64 2
  %15 = bitcast double* %14 to <2 x double>*
  %wide.load17.1 = load <2 x double>, <2 x double>* %15, align 8, !tbaa !2
  %16 = fcmp fast ogt <2 x double> %10, %wide.load.1
  %17 = fcmp fast ogt <2 x double> %11, %wide.load17.1
  %18 = select <2 x i1> %16, <2 x double> %10, <2 x double> %wide.load.1
  %19 = select <2 x i1> %17, <2 x double> %11, <2 x double> %wide.load17.1
  %index.next.1 = add i64 %index, 8
  %niter.nsub.1 = add i64 %niter, -2
  %niter.ncmp.1 = icmp eq i64 %niter.nsub.1, 0
  br i1 %niter.ncmp.1, label %middle.block.unr-lcssa, label %vector.body, !llvm.loop !6

middle.block.unr-lcssa:                           ; preds = %vector.body, %vector.ph
  %.lcssa21.ph = phi <2 x double> [ undef, %vector.ph ], [ %18, %vector.body ]
  %.lcssa.ph = phi <2 x double> [ undef, %vector.ph ], [ %19, %vector.body ]
  %index.unr = phi i64 [ 0, %vector.ph ], [ %index.next.1, %vector.body ]
  %vec.phi.unr = phi <2 x double> [ <double 0xFFF0000000000000, double 0xFFF0000000000000>, %vector.ph ], [ %18, %vector.body ]
  %vec.phi16.unr = phi <2 x double> [ <double 0xFFF0000000000000, double 0xFFF0000000000000>, %vector.ph ], [ %19, %vector.body ]
  %lcmp.mod = icmp eq i64 %xtraiter, 0
  br i1 %lcmp.mod, label %middle.block, label %vector.body.epil

vector.body.epil:                                 ; preds = %middle.block.unr-lcssa
  %20 = getelementptr inbounds double, double* %vec, i64 %index.unr
  %21 = bitcast double* %20 to <2 x double>*
  %wide.load.epil = load <2 x double>, <2 x double>* %21, align 8, !tbaa !2
  %22 = getelementptr inbounds double, double* %20, i64 2
  %23 = bitcast double* %22 to <2 x double>*
  %wide.load17.epil = load <2 x double>, <2 x double>* %23, align 8, !tbaa !2
  %24 = fcmp fast ogt <2 x double> %vec.phi16.unr, %wide.load17.epil
  %25 = select <2 x i1> %24, <2 x double> %vec.phi16.unr, <2 x double> %wide.load17.epil
  %26 = fcmp fast ogt <2 x double> %vec.phi.unr, %wide.load.epil
  %27 = select <2 x i1> %26, <2 x double> %vec.phi.unr, <2 x double> %wide.load.epil
  br label %middle.block

middle.block:                                     ; preds = %middle.block.unr-lcssa, %vector.body.epil
  %.lcssa21 = phi <2 x double> [ %.lcssa21.ph, %middle.block.unr-lcssa ], [ %27, %vector.body.epil ]
  %.lcssa = phi <2 x double> [ %.lcssa.ph, %middle.block.unr-lcssa ], [ %25, %vector.body.epil ]
  %rdx.minmax.cmp = fcmp fast ogt <2 x double> %.lcssa21, %.lcssa
  %rdx.minmax.select = select <2 x i1> %rdx.minmax.cmp, <2 x double> %.lcssa21, <2 x double> %.lcssa
  %rdx.shuf = shufflevector <2 x double> %rdx.minmax.select, <2 x double> undef, <2 x i32> <i32 1, i32 undef>
  %rdx.minmax.cmp18 = fcmp fast ogt <2 x double> %rdx.minmax.select, %rdx.shuf
  %rdx.minmax.select19 = select <2 x i1> %rdx.minmax.cmp18, <2 x double> %rdx.minmax.select, <2 x double> %rdx.shuf
  %28 = extractelement <2 x double> %rdx.minmax.select19, i32 0
  %cmp.n = icmp eq i64 %n.vec, %wide.trip.count
  br i1 %cmp.n, label %for.cond.cleanup, label %for.body.preheader20

for.cond.cleanup:                                 ; preds = %for.body, %middle.block, %entry
  %ret.0.lcssa = phi double [ 0xFFF0000000000000, %entry ], [ %28, %middle.block ], [ %ret.0., %for.body ]
  ret void

for.body:                                         ; preds = %for.body.preheader20, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ %indvars.iv.ph, %for.body.preheader20 ]
  %ret.013 = phi double [ %ret.0., %for.body ], [ %ret.013.ph, %for.body.preheader20 ]
  %arrayidx = getelementptr inbounds double, double* %vec, i64 %indvars.iv
  %29 = load double, double* %arrayidx, align 8, !tbaa !2
  %cmp1 = fcmp fast ogt double %ret.013, %29
  %ret.0. = select i1 %cmp1, double %ret.013, double %29
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %for.cond.cleanup, label %for.body, !llvm.loop !8
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1) #1

; Function Attrs: nounwind
declare dso_local i32 @printf(i8* nocapture readonly, ...) local_unnamed_addr #3

attributes #0 = { norecurse nounwind readonly uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #3 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #4 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #5 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0}
!3 = !{!"double", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.isvectorized", i32 1}
!8 = distinct !{!8, !9, !7}
!9 = !{!"llvm.loop.unroll.runtime.disable"}


; CHECK: reduce_max - {} |{[-1]:Pointer, [-1,-1]:Float@double}:{} {[-1]:Integer}:{} 
; CHECK-NEXT: double* %vec: {[-1]:Pointer, [-1,-1]:Float@double}
; CHECK-NEXT: i32 %size: {[-1]:Integer}
; CHECK-NEXT: entry
; CHECK-NEXT:   %cmp12 = icmp sgt i32 %size, 0: {[-1]:Integer}
; CHECK-NEXT:   br i1 %cmp12, label %for.body.preheader, label %for.cond.cleanup: {}
; CHECK-NEXT: for.body.preheader
; CHECK-NEXT:   %wide.trip.count = zext i32 %size to i64: {[-1]:Integer}
; CHECK-NEXT:   %min.iters.check = icmp ult i32 %size, 4: {[-1]:Integer}
; CHECK-NEXT:   br i1 %min.iters.check, label %for.body.preheader20, label %vector.ph: {}
; CHECK-NEXT: for.body.preheader20
; CHECK-NEXT:   %indvars.iv.ph = phi i64 [ 0, %for.body.preheader ], [ %n.vec, %middle.block ]: {[-1]:Integer}
; CHECK-NEXT:   %ret.013.ph = phi double [ 0xFFF0000000000000, %for.body.preheader ], [ %28, %middle.block ]: {[-1]:Float@double}
; CHECK-NEXT:   br label %for.body: {}
; CHECK-NEXT: vector.ph
; CHECK-NEXT:   %n.vec = and i64 %wide.trip.count, 4294967292: {[-1]:Integer}
; CHECK-NEXT:   %0 = add nsw i64 %n.vec, -4: {[-1]:Integer}
; CHECK-NEXT:   %1 = lshr exact i64 %0, 2: {[-1]:Integer}
; CHECK-NEXT:   %2 = add nuw nsw i64 %1, 1: {[-1]:Integer}
; CHECK-NEXT:   %xtraiter = and i64 %2, 1: {[-1]:Integer}
; CHECK-NEXT:   %3 = icmp eq i64 %0, 0: {[-1]:Integer}
; CHECK-NEXT:   br i1 %3, label %middle.block.unr-lcssa, label %vector.ph.new: {}
; CHECK-NEXT: vector.ph.new
; CHECK-NEXT:   %unroll_iter = sub nsw i64 %2, %xtraiter: {[-1]:Integer}
; CHECK-NEXT:   br label %vector.body: {}
; CHECK-NEXT: vector.body
; CHECK-NEXT:   %index = phi i64 [ 0, %vector.ph.new ], [ %index.next.1, %vector.body ]: {[-1]:Integer}
; CHECK-NEXT:   %vec.phi = phi <2 x double> [ <double 0xFFF0000000000000, double 0xFFF0000000000000>, %vector.ph.new ], [ %18, %vector.body ]: {[-1]:Float@double}
; CHECK-NEXT:   %vec.phi16 = phi <2 x double> [ <double 0xFFF0000000000000, double 0xFFF0000000000000>, %vector.ph.new ], [ %19, %vector.body ]: {[-1]:Float@double}
; CHECK-NEXT:   %niter = phi i64 [ %unroll_iter, %vector.ph.new ], [ %niter.nsub.1, %vector.body ]: {[-1]:Integer}
; CHECK-NEXT:   %4 = getelementptr inbounds double, double* %vec, i64 %index: {[-1]:Pointer, [-1,-1]:Float@double}
; CHECK-NEXT:   %5 = bitcast double* %4 to <2 x double>*: {[-1]:Pointer, [-1,-1]:Float@double}
; CHECK-NEXT:   %wide.load = load <2 x double>, <2 x double>* %5, align 8, !tbaa !2: {[-1]:Float@double}
; CHECK-NEXT:   %6 = getelementptr inbounds double, double* %4, i64 2: {[-1]:Pointer, [-1,-1]:Float@double}
; CHECK-NEXT:   %7 = bitcast double* %6 to <2 x double>*: {[-1]:Pointer, [-1,-1]:Float@double}
; CHECK-NEXT:   %wide.load17 = load <2 x double>, <2 x double>* %7, align 8, !tbaa !2: {[-1]:Float@double}
; CHECK-NEXT:   %8 = fcmp fast ogt <2 x double> %vec.phi, %wide.load: {[-1]:Integer}
; CHECK-NEXT:   %9 = fcmp fast ogt <2 x double> %vec.phi16, %wide.load17: {[-1]:Integer}
; CHECK-NEXT:   %10 = select <2 x i1> %8, <2 x double> %vec.phi, <2 x double> %wide.load: {[-1]:Float@double}
; CHECK-NEXT:   %11 = select <2 x i1> %9, <2 x double> %vec.phi16, <2 x double> %wide.load17: {[-1]:Float@double}
; CHECK-NEXT:   %index.next = or i64 %index, 4: {[-1]:Integer}
; CHECK-NEXT:   %12 = getelementptr inbounds double, double* %vec, i64 %index.next: {[-1]:Pointer, [-1,-1]:Float@double}
; CHECK-NEXT:   %13 = bitcast double* %12 to <2 x double>*: {[-1]:Pointer, [-1,-1]:Float@double}
; CHECK-NEXT:   %wide.load.1 = load <2 x double>, <2 x double>* %13, align 8, !tbaa !2: {[-1]:Float@double}
; CHECK-NEXT:   %14 = getelementptr inbounds double, double* %12, i64 2: {[-1]:Pointer, [-1,-1]:Float@double}
; CHECK-NEXT:   %15 = bitcast double* %14 to <2 x double>*: {[-1]:Pointer, [-1,-1]:Float@double}
; CHECK-NEXT:   %wide.load17.1 = load <2 x double>, <2 x double>* %15, align 8, !tbaa !2: {[-1]:Float@double}
; CHECK-NEXT:   %16 = fcmp fast ogt <2 x double> %10, %wide.load.1: {[-1]:Integer}
; CHECK-NEXT:   %17 = fcmp fast ogt <2 x double> %11, %wide.load17.1: {[-1]:Integer}
; CHECK-NEXT:   %18 = select <2 x i1> %16, <2 x double> %10, <2 x double> %wide.load.1: {[-1]:Float@double}
; CHECK-NEXT:   %19 = select <2 x i1> %17, <2 x double> %11, <2 x double> %wide.load17.1: {[-1]:Float@double}
; CHECK-NEXT:   %index.next.1 = add i64 %index, 8: {[-1]:Integer}
; CHECK-NEXT:   %niter.nsub.1 = add i64 %niter, -2: {[-1]:Integer}
; CHECK-NEXT:   %niter.ncmp.1 = icmp eq i64 %niter.nsub.1, 0: {[-1]:Integer}
; CHECK-NEXT:   br i1 %niter.ncmp.1, label %middle.block.unr-lcssa, label %vector.body, !llvm.loop !6: {}
; CHECK-NEXT: middle.block.unr-lcssa
; CHECK-NEXT:   %.lcssa21.ph = phi <2 x double> [ undef, %vector.ph ], [ %18, %vector.body ]: {[-1]:Float@double}
; CHECK-NEXT:   %.lcssa.ph = phi <2 x double> [ undef, %vector.ph ], [ %19, %vector.body ]: {[-1]:Float@double}
; CHECK-NEXT:   %index.unr = phi i64 [ 0, %vector.ph ], [ %index.next.1, %vector.body ]: {[-1]:Integer}
; CHECK-NEXT:   %vec.phi.unr = phi <2 x double> [ <double 0xFFF0000000000000, double 0xFFF0000000000000>, %vector.ph ], [ %18, %vector.body ]: {[-1]:Float@double}
; CHECK-NEXT:   %vec.phi16.unr = phi <2 x double> [ <double 0xFFF0000000000000, double 0xFFF0000000000000>, %vector.ph ], [ %19, %vector.body ]: {[-1]:Float@double}
; CHECK-NEXT:   %lcmp.mod = icmp eq i64 %xtraiter, 0: {[-1]:Integer}
; CHECK-NEXT:   br i1 %lcmp.mod, label %middle.block, label %vector.body.epil: {}
; CHECK-NEXT: vector.body.epil
; CHECK-NEXT:   %20 = getelementptr inbounds double, double* %vec, i64 %index.unr: {[-1]:Pointer, [-1,-1]:Float@double}
; CHECK-NEXT:   %21 = bitcast double* %20 to <2 x double>*: {[-1]:Pointer, [-1,-1]:Float@double}
; CHECK-NEXT:   %wide.load.epil = load <2 x double>, <2 x double>* %21, align 8, !tbaa !2: {[-1]:Float@double}
; CHECK-NEXT:   %22 = getelementptr inbounds double, double* %20, i64 2: {[-1]:Pointer, [-1,-1]:Float@double}
; CHECK-NEXT:   %23 = bitcast double* %22 to <2 x double>*: {[-1]:Pointer, [-1,-1]:Float@double}
; CHECK-NEXT:   %wide.load17.epil = load <2 x double>, <2 x double>* %23, align 8, !tbaa !2: {[-1]:Float@double}
; CHECK-NEXT:   %24 = fcmp fast ogt <2 x double> %vec.phi16.unr, %wide.load17.epil: {[-1]:Integer}
; CHECK-NEXT:   %25 = select <2 x i1> %24, <2 x double> %vec.phi16.unr, <2 x double> %wide.load17.epil: {[-1]:Float@double}
; CHECK-NEXT:   %26 = fcmp fast ogt <2 x double> %vec.phi.unr, %wide.load.epil: {[-1]:Integer}
; CHECK-NEXT:   %27 = select <2 x i1> %26, <2 x double> %vec.phi.unr, <2 x double> %wide.load.epil: {[-1]:Float@double}
; CHECK-NEXT:   br label %middle.block: {}
; CHECK-NEXT: middle.block
; CHECK-NEXT:   %.lcssa21 = phi <2 x double> [ %.lcssa21.ph, %middle.block.unr-lcssa ], [ %27, %vector.body.epil ]: {[-1]:Float@double}
; CHECK-NEXT:   %.lcssa = phi <2 x double> [ %.lcssa.ph, %middle.block.unr-lcssa ], [ %25, %vector.body.epil ]: {[-1]:Float@double}
; CHECK-NEXT:   %rdx.minmax.cmp = fcmp fast ogt <2 x double> %.lcssa21, %.lcssa: {[-1]:Integer}
; CHECK-NEXT:   %rdx.minmax.select = select <2 x i1> %rdx.minmax.cmp, <2 x double> %.lcssa21, <2 x double> %.lcssa: {[-1]:Float@double}
; CHECK-NEXT:   %rdx.shuf = shufflevector <2 x double> %rdx.minmax.select, <2 x double> undef, <2 x i32> <i32 1, i32 undef>: {[-1]:Float@double, [8]:Anything, [9]:Anything, [10]:Anything, [11]:Anything, [12]:Anything, [13]:Anything, [14]:Anything, [15]:Anything}
; CHECK-NEXT:   %rdx.minmax.cmp18 = fcmp fast ogt <2 x double> %rdx.minmax.select, %rdx.shuf: {[-1]:Integer}
; CHECK-NEXT:   %rdx.minmax.select19 = select <2 x i1> %rdx.minmax.cmp18, <2 x double> %rdx.minmax.select, <2 x double> %rdx.shuf: {[-1]:Float@double}
; CHECK-NEXT:   %28 = extractelement <2 x double> %rdx.minmax.select19, i32 0: {[-1]:Float@double}
; CHECK-NEXT:   %cmp.n = icmp eq i64 %n.vec, %wide.trip.count: {[-1]:Integer}
; CHECK-NEXT:   br i1 %cmp.n, label %for.cond.cleanup, label %for.body.preheader20: {}
; CHECK-NEXT: for.cond.cleanup
; CHECK-NEXT:   %ret.0.lcssa = phi double [ 0xFFF0000000000000, %entry ], [ %28, %middle.block ], [ %ret.0., %for.body ]: {[-1]:Float@double}
; CHECK-NEXT:   ret void: {}
; CHECK-NEXT: for.body
; CHECK-NEXT:   %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ %indvars.iv.ph, %for.body.preheader20 ]: {[-1]:Integer}
; CHECK-NEXT:   %ret.013 = phi double [ %ret.0., %for.body ], [ %ret.013.ph, %for.body.preheader20 ]: {[-1]:Float@double}
; CHECK-NEXT:   %arrayidx = getelementptr inbounds double, double* %vec, i64 %indvars.iv: {[-1]:Pointer, [-1,-1]:Float@double}
; CHECK-NEXT:   %29 = load double, double* %arrayidx, align 8, !tbaa !2: {[-1]:Float@double}
; CHECK-NEXT:   %cmp1 = fcmp fast ogt double %ret.013, %29: {[-1]:Integer}
; CHECK-NEXT:   %ret.0. = select i1 %cmp1, double %ret.013, double %29: {[-1]:Float@double}
; CHECK-NEXT:   %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1: {[-1]:Integer}
; CHECK-NEXT:   %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count: {[-1]:Integer}
; CHECK-NEXT:   br i1 %exitcond, label %for.cond.cleanup, label %for.body, !llvm.loop !8: {}
