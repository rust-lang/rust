; RUN: if [ %llvmver -ge 9 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -enzyme-shared-forward -mem2reg -sroa -instsimplify -early-cse -simplifycfg -S | FileCheck %s; fi

; TODO the code generated here may cause illegal reads as the phi recopmutation does not check that a load is legal to recompute for
; bounds reasons, only aliasing reasons. This code illustrates an example where it is incorrect and should be remedied. Now, the value
; will never be used and not present a correctness error, but may present a segfault where one need not occur.

target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

%struct.cudaFuncAttributes = type { i64, i64, i64, i32, i32, i32, i32, i32, i32, i32 }

@_ZZ22gpu_square_matrix_multPfS_S_mE6tile_a = internal unnamed_addr addrspace(3) global [16 x [16 x float]] undef, align 4
@_ZZ22gpu_square_matrix_multPfS_S_mE6tile_b = internal unnamed_addr addrspace(3) global [16 x [16 x float]] undef, align 4

; Function Attrs: convergent nounwind
declare void @llvm.nvvm.barrier0() #2

; Function Attrs: convergent nounwind mustprogress
define void @_Z22gpu_square_matrix_multPfS_S_m(float* noalias nocapture readonly %d_a, float* noalias nocapture readonly %d_b, float* noalias nocapture %d_result, i64 %n) #1 {
entry:
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #6
  %mul = shl i32 %0, 4
  %1 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y() #6
  %add = add i32 %mul, %1
  %conv = zext i32 %add to i64
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #6
  %mul3 = shl i32 %2, 4
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #6
  %add5 = add i32 %mul3, %3
  %conv6 = zext i32 %add5 to i64
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x() #6
  %conv8 = zext i32 %4 to i64
  %cmp104.not = icmp eq i32 %4, 0
  br i1 %cmp104.not, label %for.cond.cleanup, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %entry
  %mul9 = mul i64 %conv, %n
  %conv13 = zext i32 %3 to i64
  %add11 = add i64 %mul9, %conv13
  %mul15 = mul i64 %n, %n
  %idxprom = zext i32 %1 to i64
  %arrayidx2195 = getelementptr inbounds [16 x [16 x float]], [16 x [16 x float]] addrspace(3)* @_ZZ22gpu_square_matrix_multPfS_S_mE6tile_a, i64 0, i64 %idxprom, i64 %conv13
  %arrayidx21 = addrspacecast float addrspace(3)* %arrayidx2195 to float*
  %arrayidx4097 = getelementptr inbounds [16 x [16 x float]], [16 x [16 x float]] addrspace(3)* @_ZZ22gpu_square_matrix_multPfS_S_mE6tile_b, i64 0, i64 %idxprom, i64 %conv13
  %arrayidx40 = addrspacecast float addrspace(3)* %arrayidx4097 to float*
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.cond.cleanup43, %entry
  %tmp.0.lcssa = phi float [ 0.000000e+00, %entry ], [ %add56, %for.cond.cleanup43 ]
  %cmp60 = icmp ult i64 %conv, %n
  %cmp61 = icmp ult i64 %conv6, %n
  %or.cond = and i1 %cmp60, %cmp61
  br i1 %or.cond, label %if.then, label %if.end

for.body:                                         ; preds = %for.body.lr.ph, %for.cond.cleanup43
  %sub.0106 = phi i64 [ 0, %for.body.lr.ph ], [ %inc58, %for.cond.cleanup43 ]
  %tmp.0105 = phi float [ 0.000000e+00, %for.body.lr.ph ], [ %add56, %for.cond.cleanup43 ]
  %mul10 = shl i64 %sub.0106, 4
  %add14 = add i64 %add11, %mul10
  %cmp16.not = icmp ult i64 %add14, %mul15
  br i1 %cmp16.not, label %cond.false, label %cond.end

cond.false:                                       ; preds = %for.body
  %arrayidx = getelementptr inbounds float, float* %d_a, i64 %add14
  %5 = load float, float* %arrayidx, align 4, !tbaa !6
  br label %cond.end

cond.end:                                         ; preds = %for.body, %cond.false
  %cond = phi float [ %5, %cond.false ], [ 0.000000e+00, %for.body ]
  store float %cond, float* %arrayidx21, align 4, !tbaa !6
  %add25 = add nuw nsw i64 %mul10, %idxprom
  %mul26 = mul i64 %add25, %n
  %add27 = add i64 %mul26, %conv6
  %cmp29.not = icmp ult i64 %add27, %mul15
  br i1 %cmp29.not, label %cond.false31, label %cond.end33

cond.false31:                                     ; preds = %cond.end
  %arrayidx32 = getelementptr inbounds float, float* %d_b, i64 %add27
  %6 = load float, float* %arrayidx32, align 4, !tbaa !6
  br label %cond.end33

cond.end33:                                       ; preds = %cond.end, %cond.false31
  %cond34 = phi float [ %6, %cond.false31 ], [ 0.000000e+00, %cond.end ]
  store float %cond34, float* %arrayidx40, align 4, !tbaa !6
  call void @llvm.nvvm.barrier0()
  br label %for.body44

for.cond.cleanup43:                               ; preds = %for.body44
  call void @llvm.nvvm.barrier0()
  %inc58 = add nuw nsw i64 %sub.0106, 1
  %exitcond107.not = icmp eq i64 %inc58, %conv8
  br i1 %exitcond107.not, label %for.cond.cleanup, label %for.body, !llvm.loop !10

for.body44:                                       ; preds = %cond.end33, %for.body44
  %k.0103 = phi i32 [ 0, %cond.end33 ], [ %inc, %for.body44 ]
  %tmp.1102 = phi float [ %tmp.0105, %cond.end33 ], [ %add56, %for.body44 ]
  %idxprom48 = zext i32 %k.0103 to i64
  %arrayidx4999 = getelementptr inbounds [16 x [16 x float]], [16 x [16 x float]] addrspace(3)* @_ZZ22gpu_square_matrix_multPfS_S_mE6tile_a, i64 0, i64 %idxprom, i64 %idxprom48
  %arrayidx49 = addrspacecast float addrspace(3)* %arrayidx4999 to float*
  %7 = load float, float* %arrayidx49, align 4, !tbaa !6
  %arrayidx54101 = getelementptr inbounds [16 x [16 x float]], [16 x [16 x float]] addrspace(3)* @_ZZ22gpu_square_matrix_multPfS_S_mE6tile_b, i64 0, i64 %idxprom48, i64 %conv13
  %arrayidx54 = addrspacecast float addrspace(3)* %arrayidx54101 to float*
  %8 = load float, float* %arrayidx54, align 4, !tbaa !6
  %mul55 = fmul contract float %7, %8
  %add56 = fadd contract float %tmp.1102, %mul55
  %inc = add nuw nsw i32 %k.0103, 1
  %exitcond.not = icmp eq i32 %inc, 16
  br i1 %exitcond.not, label %for.cond.cleanup43, label %for.body44, !llvm.loop !13

if.then:                                          ; preds = %for.cond.cleanup
  %mul62 = mul i64 %conv, %n
  %add63 = add i64 %mul62, %conv6
  %arrayidx64 = getelementptr inbounds float, float* %d_result, i64 %add63
  store float %tmp.0.lcssa, float* %arrayidx64, align 4, !tbaa !6
  br label %if.end

if.end:                                           ; preds = %if.then, %for.cond.cleanup
  call void @llvm.nvvm.barrier0()
  ret void
}

; Function Attrs: convergent norecurse nounwind mustprogress
define void @_Z4axpyPfS_S_S_S_S_m(float* %x, float* %xp, float* %y, float* %yp, float* %z, float* %zp, i64 %sz) local_unnamed_addr #3 {
entry:
  %0 = bitcast float* %x to i8*
  %1 = bitcast float* %xp to i8*
  %2 = bitcast float* %y to i8*
  %3 = bitcast float* %yp to i8*
  %4 = bitcast float* %z to i8*
  %5 = bitcast float* %zp to i8*
  call void @_Z17__enzyme_autodiffPvS_S_S_S_S_S_m(i8* bitcast (void (float*, float*, float*, i64)* @_Z22gpu_square_matrix_multPfS_S_m to i8*), i8* %0, i8* %1, i8* %2, i8* %3, i8* %4, i8* %5, i64 %sz) #2
  ret void
}

; Function Attrs: convergent nounwind
declare void @_Z17__enzyme_autodiffPvS_S_S_S_S_S_m(i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64) local_unnamed_addr #4

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #5

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.y() #5

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #5

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() #5

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.x() #5

attributes #0 = { nounwind willreturn "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_60" "target-features"="+ptx72,+sm_60" }
attributes #1 = { convergent nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_60" "target-features"="+ptx72,+sm_60" }
attributes #2 = { convergent nounwind }
attributes #3 = { convergent norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_60" "target-features"="+ptx72,+sm_60" }
attributes #4 = { convergent nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_60" "target-features"="+ptx72,+sm_60" }
attributes #5 = { nounwind readnone }
attributes #6 = { nounwind }

!llvm.module.flags = !{!0, !1, !2}
!nvvm.annotations = !{!3}
!llvm.ident = !{!4, !5}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 11, i32 2]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!3 = !{void (float*, float*, float*, float*, float*, float*, i64)* @_Z4axpyPfS_S_S_S_S_m, !"kernel", i32 1}
!4 = !{!"clang version 13.0.0 (git@github.com:llvm/llvm-project 8987216a14a126f315cca1d49b3aa526509c9d4c)"}
!5 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"float", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C++ TBAA"}
!10 = distinct !{!10, !11, !12}
!11 = !{!"llvm.loop.mustprogress"}
!12 = !{!"llvm.loop.unroll.disable"}
!13 = distinct !{!13, !11, !12}

; CHECK: define internal void @diffe_Z22gpu_square_matrix_multPfS_S_m(float* noalias nocapture readonly %d_a, float* nocapture %"d_a'", float* noalias nocapture readonly %d_b, float* nocapture %"d_b'", float* noalias nocapture %d_result, float* nocapture %"d_result'", i64 %n)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
; CHECK-NEXT:   %1 = icmp eq i32 %0, 0
; CHECK-NEXT:   %2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y()
; CHECK-NEXT:   %3 = icmp eq i32 %2, 0
; CHECK-NEXT:   %4 = call i32 @llvm.nvvm.read.ptx.sreg.tid.z()
; CHECK-NEXT:   %5 = icmp eq i32 %4, 0
; CHECK-NEXT:   %6 = and i1 %1, %3
; CHECK-NEXT:   %7 = and i1 %6, %5
; CHECK-NEXT:   br i1 %7, label %shblock, label %8

; CHECK: shblock:                                          ; preds = %entry
; CHECK-NEXT:   store [16 x [16 x float]] zeroinitializer, [16 x [16 x float]] addrspace(3)* @_ZZ22gpu_square_matrix_multPfS_S_mE6tile_a_shadow, align 4
; CHECK-NEXT:   store [16 x [16 x float]] zeroinitializer, [16 x [16 x float]] addrspace(3)* @_ZZ22gpu_square_matrix_multPfS_S_mE6tile_b_shadow, align 4
; CHECK-NEXT:   br label %8

; CHECK: 8:                                                ; preds = %shblock, %entry
; CHECK-NEXT:   call void @llvm.nvvm.barrier0()
; CHECK-NEXT:   %9 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
; CHECK-NEXT:   %mul = shl i32 %9, 4
; CHECK-NEXT:   %10 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y()
; CHECK-NEXT:   %add = add i32 %mul, %10
; CHECK-NEXT:   %conv = zext i32 %add to i64
; CHECK-NEXT:   %11 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
; CHECK-NEXT:   %mul3 = shl i32 %11, 4
; CHECK-NEXT:   %12 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
; CHECK-NEXT:   %add5 = add i32 %mul3, %12
; CHECK-NEXT:   %conv6 = zext i32 %add5 to i64
; CHECK-NEXT:   %13 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
; CHECK-NEXT:   %conv8 = zext i32 %13 to i64
; CHECK-NEXT:   %cmp104.not = icmp eq i32 %13, 0
; CHECK-NEXT:   br i1 %cmp104.not, label %for.cond.cleanup, label %for.body.lr.ph

; CHECK: for.body.lr.ph:                                   ; preds = %8
; CHECK-NEXT:   %mul9 = mul i64 %conv, %n
; CHECK-NEXT:   %conv13 = zext i32 %12 to i64
; CHECK-NEXT:   %add11 = add i64 %mul9, %conv13
; CHECK-NEXT:   %mul15 = mul i64 %n, %n
; CHECK-NEXT:   %idxprom = zext i32 %10 to i64
; CHECK-NEXT:   %arrayidx2195 = getelementptr inbounds [16 x [16 x float]], [16 x [16 x float]] addrspace(3)* @_ZZ22gpu_square_matrix_multPfS_S_mE6tile_a, i64 0, i64 %idxprom, i64 %conv13
; CHECK-NEXT:   %arrayidx21 = addrspacecast float addrspace(3)* %arrayidx2195 to float*
; CHECK-NEXT:   %arrayidx4097 = getelementptr inbounds [16 x [16 x float]], [16 x [16 x float]] addrspace(3)* @_ZZ22gpu_square_matrix_multPfS_S_mE6tile_b, i64 0, i64 %idxprom, i64 %conv13
; CHECK-NEXT:   %arrayidx40 = addrspacecast float addrspace(3)* %arrayidx4097 to float*
; CHECK-NEXT:   br label %for.body

; CHECK: for.cond.cleanup:                                 ; preds = %for.cond.cleanup43, %8
; CHECK-NEXT:   %tmp.0.lcssa = phi float [ 0.000000e+00, %8 ], [ %add56, %for.cond.cleanup43 ]
; CHECK-NEXT:   %cmp60 = icmp ult i64 %conv, %n
; CHECK-NEXT:   %cmp61 = icmp ult i64 %conv6, %n
; CHECK-NEXT:   %or.cond = and i1 %cmp60, %cmp61
; CHECK-NEXT:   br i1 %or.cond, label %if.then, label %if.end

; CHECK: for.body:                                         ; preds = %for.cond.cleanup43, %for.body.lr.ph
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %for.cond.cleanup43 ], [ 0, %for.body.lr.ph ]
; CHECK-NEXT:   %tmp.0105 = phi float [ 0.000000e+00, %for.body.lr.ph ], [ %add56, %for.cond.cleanup43 ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %mul10 = shl i64 %iv, 4
; CHECK-NEXT:   %add14 = add i64 %add11, %mul10
; CHECK-NEXT:   %cmp16.not = icmp ult i64 %add14, %mul15
; CHECK-NEXT:   br i1 %cmp16.not, label %cond.false, label %cond.end

; CHECK: cond.false:                                       ; preds = %for.body
; CHECK-NEXT:   %arrayidx = getelementptr inbounds float, float* %d_a, i64 %add14
; CHECK-NEXT:   %14 = load float, float* %arrayidx, align 4, !tbaa !9
; CHECK-NEXT:   br label %cond.end

; CHECK: cond.end:                                         ; preds = %cond.false, %for.body
; CHECK-NEXT:   %cond = phi {{(contract )?}}float [ %14, %cond.false ], [ 0.000000e+00, %for.body ]
; CHECK-NEXT:   store float %cond, float* %arrayidx21, align 4, !tbaa !9
; CHECK-NEXT:   %add25 = add nuw nsw i64 %mul10, %idxprom
; CHECK-NEXT:   %mul26 = mul i64 %add25, %n
; CHECK-NEXT:   %add27 = add i64 %mul26, %conv6
; CHECK-NEXT:   %cmp29.not = icmp ult i64 %add27, %mul15
; CHECK-NEXT:   br i1 %cmp29.not, label %cond.false31, label %cond.end33

; CHECK: cond.false31:                                     ; preds = %cond.end
; CHECK-NEXT:   %arrayidx32 = getelementptr inbounds float, float* %d_b, i64 %add27
; CHECK-NEXT:   %15 = load float, float* %arrayidx32, align 4, !tbaa !9
; CHECK-NEXT:   br label %cond.end33

; CHECK: cond.end33:                                       ; preds = %cond.false31, %cond.end
; CHECK-NEXT:   %cond34 = phi {{(contract )?}}float [ %15, %cond.false31 ], [ 0.000000e+00, %cond.end ]
; CHECK-NEXT:   store float %cond34, float* %arrayidx40, align 4, !tbaa !9
; CHECK-NEXT:   call void @llvm.nvvm.barrier0()
; CHECK-NEXT:   br label %for.body44

; CHECK: for.cond.cleanup43:                               ; preds = %for.body44
; CHECK-NEXT:   call void @llvm.nvvm.barrier0()
; CHECK-NEXT:   %exitcond107.not = icmp eq i64 %iv.next, %conv8
; CHECK-NEXT:   br i1 %exitcond107.not, label %for.cond.cleanup, label %for.body, !llvm.loop !13

; CHECK: for.body44:                                       ; preds = %for.body44, %cond.end33
; CHECK-NEXT:   %iv1 = phi i64 [ %iv.next2, %for.body44 ], [ 0, %cond.end33 ]
; CHECK-NEXT:   %tmp.1102 = phi float [ %tmp.0105, %cond.end33 ], [ %add56, %for.body44 ]
; CHECK-NEXT:   %16 = trunc i64 %iv1 to i32
; CHECK-NEXT:   %iv.next2 = add nuw nsw i64 %iv1, 1
; CHECK-NEXT:   %idxprom48 = zext i32 %16 to i64
; CHECK-NEXT:   %arrayidx4999 = getelementptr inbounds [16 x [16 x float]], [16 x [16 x float]] addrspace(3)* @_ZZ22gpu_square_matrix_multPfS_S_mE6tile_a, i64 0, i64 %idxprom, i64 %idxprom48
; CHECK-NEXT:   %arrayidx49 = addrspacecast float addrspace(3)* %arrayidx4999 to float*
; CHECK-NEXT:   %17 = load float, float* %arrayidx49, align 4, !tbaa !9
; CHECK-NEXT:   %arrayidx54101 = getelementptr inbounds [16 x [16 x float]], [16 x [16 x float]] addrspace(3)* @_ZZ22gpu_square_matrix_multPfS_S_mE6tile_b, i64 0, i64 %idxprom48, i64 %conv13
; CHECK-NEXT:   %arrayidx54 = addrspacecast float addrspace(3)* %arrayidx54101 to float*
; CHECK-NEXT:   %18 = load float, float* %arrayidx54, align 4, !tbaa !9
; CHECK-NEXT:   %mul55 = fmul contract float %17, %18
; CHECK-NEXT:   %add56 = fadd contract float %tmp.1102, %mul55
; CHECK-NEXT:   %inc = add nuw nsw i32 %16, 1
; CHECK-NEXT:   %exitcond.not = icmp eq i32 %inc, 16
; CHECK-NEXT:   br i1 %exitcond.not, label %for.cond.cleanup43, label %for.body44, !llvm.loop !16

; CHECK: if.then:                                          ; preds = %for.cond.cleanup
; CHECK-NEXT:   %mul62 = mul i64 %conv, %n
; CHECK-NEXT:   %add63 = add i64 %mul62, %conv6
; CHECK-NEXT:   %arrayidx64 = getelementptr inbounds float, float* %d_result, i64 %add63
; CHECK-NEXT:   store float %tmp.0.lcssa, float* %arrayidx64, align 4, !tbaa !9
; CHECK-NEXT:   br label %if.end

; CHECK: if.end:                                           ; preds = %if.then, %for.cond.cleanup
; CHECK-NEXT:   %or.cond.pr = phi i1 [ %or.cond, %if.then ], [ false, %for.cond.cleanup ]
; CHECK-NEXT:   call void @llvm.nvvm.barrier0()
; CHECK-NEXT:   call void @llvm.nvvm.barrier0()
; CHECK-NEXT:   br i1 %or.cond.pr, label %invertif.then, label %invertfor.cond.cleanup

; CHECK: invertentry:                                      ; preds = %invertfor.body, %invertfor.cond.cleanup
; CHECK-NEXT:   ret void

; CHECK: invertfor.cond.cleanup.loopexit:                  ; preds = %invertfor.cond.cleanup
; CHECK-NEXT:   %_unwrap = add nsw i64 %conv8, -1
; CHECK-NEXT:   br label %invertfor.cond.cleanup43

; CHECK: invertfor.cond.cleanup:                           ; preds = %if.end, %invertif.then
; CHECK-NEXT:   %"tmp.0.lcssa'de.0" = phi float [ %41, %invertif.then ], [ 0.000000e+00, %if.end ]
; CHECK-NEXT:   %19 = select fast i1 %cmp104.not, float 0.000000e+00, float %"tmp.0.lcssa'de.0"
; CHECK-NEXT:   br i1 %cmp104.not, label %invertentry, label %invertfor.cond.cleanup.loopexit

; CHECK: invertfor.body:                                   ; preds = %invertcond.end, %invertcond.false
; CHECK-NEXT:   %"'de.0" = phi float [ 0.000000e+00, %invertcond.false ], [ %27, %invertcond.end ]
; CHECK-NEXT:   %20 = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   %21 = fadd fast float %37, %39
; CHECK-NEXT:   %22 = select fast i1 %20, float %37, float %21
; CHECK-NEXT:   br i1 %20, label %invertentry, label %incinvertfor.body

; CHECK: incinvertfor.body:                                ; preds = %invertfor.body
; CHECK-NEXT:   %23 = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertfor.cond.cleanup43

; CHECK: invertcond.false:                                 ; preds = %invertcond.end
; CHECK-NEXT:   %"arrayidx'ipg_unwrap" = getelementptr inbounds float, float* %"d_a'", i64 %add14_unwrap7
; CHECK-NEXT:   %24 = atomicrmw fadd float* %"arrayidx'ipg_unwrap", float %27 monotonic
; CHECK-NEXT:   br label %invertfor.body

; CHECK: invertcond.end:                                   ; preds = %invertcond.end33, %invertcond.false31
; CHECK-NEXT:   %"'de8.0" = phi float [ 0.000000e+00, %invertcond.false31 ], [ %31, %invertcond.end33 ]
; CHECK-NEXT:   %"arrayidx2195'ipg_unwrap" = getelementptr inbounds [16 x [16 x float]], [16 x [16 x float]] addrspace(3)* @_ZZ22gpu_square_matrix_multPfS_S_mE6tile_a_shadow, i64 0, i64 %idxprom_unwrap54, i64 %conv13_unwrap53
; CHECK-NEXT:   %"arrayidx21'ipc_unwrap" = addrspacecast float addrspace(3)* %"arrayidx2195'ipg_unwrap" to float*
; CHECK-NEXT:   %25 = load float, float* %"arrayidx21'ipc_unwrap", align 4
; CHECK-NEXT:   store float 0.000000e+00, float* %"arrayidx21'ipc_unwrap", align 4
; CHECK-NEXT:   %add11_unwrap5 = add i64 %mul9_unwrap35, %conv13_unwrap53
; CHECK-NEXT:   %add14_unwrap7 = add i64 %add11_unwrap5, %mul10_unwrap19
; CHECK-NEXT:   %cmp16.not_unwrap = icmp ult i64 %add14_unwrap7, %mul15_unwrap24
; CHECK-NEXT:   %26 = fadd fast float %"'de.1", %25
; CHECK-NEXT:   %27 = select fast i1 %cmp16.not_unwrap, float %26, float %"'de.1"
; CHECK-NEXT:   br i1 %cmp16.not_unwrap, label %invertcond.false, label %invertfor.body

; CHECK: invertcond.false31:                               ; preds = %invertcond.end33
; CHECK-NEXT:   %"arrayidx32'ipg_unwrap" = getelementptr inbounds float, float* %"d_b'", i64 %add27_unwrap16
; CHECK-NEXT:   %28 = atomicrmw fadd float* %"arrayidx32'ipg_unwrap", float %31 monotonic
; CHECK-NEXT:   br label %invertcond.end

; CHECK: invertcond.end33:                                 ; preds = %invertfor.body44
; CHECK-NEXT:   call void @llvm.nvvm.barrier0()
; CHECK-NEXT:   %"arrayidx4097'ipg_unwrap" = getelementptr inbounds [16 x [16 x float]], [16 x [16 x float]] addrspace(3)* @_ZZ22gpu_square_matrix_multPfS_S_mE6tile_b_shadow, i64 0, i64 %idxprom_unwrap54, i64 %conv13_unwrap53
; CHECK-NEXT:   %"arrayidx40'ipc_unwrap" = addrspacecast float addrspace(3)* %"arrayidx4097'ipg_unwrap" to float*
; CHECK-NEXT:   %29 = load float, float* %"arrayidx40'ipc_unwrap", align 4
; CHECK-NEXT:   store float 0.000000e+00, float* %"arrayidx40'ipc_unwrap", align 4
; CHECK-NEXT:   %add25_unwrap14 = add nuw nsw i64 %mul10_unwrap19, %idxprom_unwrap54
; CHECK-NEXT:   %mul26_unwrap15 = mul i64 %add25_unwrap14, %n
; CHECK-NEXT:   %add27_unwrap16 = add i64 %mul26_unwrap15, %conv6
; CHECK-NEXT:   %cmp29.not_unwrap = icmp ult i64 %add27_unwrap16, %mul15_unwrap24
; CHECK-NEXT:   %30 = fadd fast float %"'de8.1", %29
; CHECK-NEXT:   %31 = select fast i1 %cmp29.not_unwrap, float %30, float %"'de8.1"
; CHECK-NEXT:   br i1 %cmp29.not_unwrap, label %invertcond.false31, label %invertcond.end

; CHECK: invertfor.cond.cleanup43:                         ; preds = %incinvertfor.body, %invertfor.cond.cleanup.loopexit
; CHECK-NEXT:   %"'de8.1" = phi float [ 0.000000e+00, %invertfor.cond.cleanup.loopexit ], [ %"'de8.0", %incinvertfor.body ]
; CHECK-NEXT:   %"'de.1" = phi float [ 0.000000e+00, %invertfor.cond.cleanup.loopexit ], [ %"'de.0", %incinvertfor.body ]
; CHECK-NEXT:   %"add56'de.0" = phi float [ %19, %invertfor.cond.cleanup.loopexit ], [ %22, %incinvertfor.body ]
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %_unwrap, %invertfor.cond.cleanup.loopexit ], [ %23, %incinvertfor.body ]
; CHECK-NEXT:   call void @llvm.nvvm.barrier0()
; CHECK-NEXT:   br label %invertfor.body44

; CHECK: invertfor.body44:                                 ; preds = %incinvertfor.body44, %invertfor.cond.cleanup43
; CHECK-NEXT:   %"tmp.0105'de.1" = phi float [ 0.000000e+00, %invertfor.cond.cleanup43 ], [ %39, %incinvertfor.body44 ]
; CHECK-NEXT:   %"add56'de.1" = phi float [ %"add56'de.0", %invertfor.cond.cleanup43 ], [ %37, %incinvertfor.body44 ]
; CHECK-NEXT:   %"iv1'ac.0" = phi i64 [ 15, %invertfor.cond.cleanup43 ], [ %40, %incinvertfor.body44 ]
; CHECK-NEXT:   %_unwrap18 = trunc i64 %"iv1'ac.0" to i32
; CHECK-NEXT:   %mul10_unwrap19 = shl i64 %"iv'ac.0", 4
; CHECK-NEXT:   %idxprom_unwrap20 = zext i32 %_unwrap18 to i64
; CHECK-NEXT:   %add25_unwrap21 = add nuw nsw i64 %mul10_unwrap19, %idxprom_unwrap20
; CHECK-NEXT:   %mul26_unwrap22 = mul i64 %add25_unwrap21, %n
; CHECK-NEXT:   %add27_unwrap23 = add i64 %mul26_unwrap22, %conv6
; CHECK-NEXT:   %mul15_unwrap24 = mul i64 %n, %n
; CHECK-NEXT:   %cmp29.not_unwrap25 = icmp ult i64 %add27_unwrap23, %mul15_unwrap24
; CHECK-NEXT:   %arrayidx32_unwrap = getelementptr inbounds float, float* %d_b, i64 %add27_unwrap23
; CHECK-NEXT:   %_unwrap34 = load float, float* %arrayidx32_unwrap, align 4, !tbaa !9
; CHECK-NEXT:   %32 = select fast i1 %cmp29.not_unwrap25, float %_unwrap34, float 0.000000e+00
; CHECK-NEXT:   %m0diffe = fmul fast float %"add56'de.1", %32
; CHECK-NEXT:   %mul9_unwrap35 = mul i64 %conv, %n
; CHECK-NEXT:   %add11_unwrap37 = add i64 %mul9_unwrap35, %idxprom_unwrap20
; CHECK-NEXT:   %add14_unwrap39 = add i64 %add11_unwrap37, %mul10_unwrap19
; CHECK-NEXT:   %cmp16.not_unwrap41 = icmp ult i64 %add14_unwrap39, %mul15_unwrap24
; CHECK-NEXT:   %arrayidx_unwrap = getelementptr inbounds float, float* %d_a, i64 %add14_unwrap39
; CHECK-NEXT:   %_unwrap50 = load float, float* %arrayidx_unwrap, align 4, !tbaa !9
; CHECK-NEXT:   %33 = select fast i1 %cmp16.not_unwrap41, float %_unwrap50, float 0.000000e+00
; CHECK-NEXT:   %m1diffe = fmul fast float %"add56'de.1", %33
; CHECK-NEXT:   %conv13_unwrap53 = zext i32 %12 to i64
; CHECK-NEXT:   %"arrayidx54101'ipg_unwrap" = getelementptr inbounds [16 x [16 x float]], [16 x [16 x float]] addrspace(3)* @_ZZ22gpu_square_matrix_multPfS_S_mE6tile_b_shadow, i64 0, i64 %idxprom_unwrap20, i64 %conv13_unwrap53
; CHECK-NEXT:   %"arrayidx54'ipc_unwrap" = addrspacecast float addrspace(3)* %"arrayidx54101'ipg_unwrap" to float*
; CHECK-NEXT:   %34 = atomicrmw fadd float* %"arrayidx54'ipc_unwrap", float %m1diffe monotonic
; CHECK-NEXT:   %idxprom_unwrap54 = zext i32 %10 to i64
; CHECK-NEXT:   %"arrayidx4999'ipg_unwrap" = getelementptr inbounds [16 x [16 x float]], [16 x [16 x float]] addrspace(3)* @_ZZ22gpu_square_matrix_multPfS_S_mE6tile_a_shadow, i64 0, i64 %idxprom_unwrap54, i64 %idxprom_unwrap20
; CHECK-NEXT:   %"arrayidx49'ipc_unwrap" = addrspacecast float addrspace(3)* %"arrayidx4999'ipg_unwrap" to float*
; CHECK-NEXT:   %35 = atomicrmw fadd float* %"arrayidx49'ipc_unwrap", float %m0diffe monotonic
; CHECK-NEXT:   %36 = icmp eq i64 %"iv1'ac.0", 0
; CHECK-NEXT:   %37 = select fast i1 %36, float 0.000000e+00, float %"add56'de.1"
; CHECK-NEXT:   %38 = fadd fast float %"tmp.0105'de.1", %"add56'de.1"
; CHECK-NEXT:   %39 = select fast i1 %36, float %38, float %"tmp.0105'de.1"
; CHECK-NEXT:   br i1 %36, label %invertcond.end33, label %incinvertfor.body44

; CHECK: incinvertfor.body44:                              ; preds = %invertfor.body44
; CHECK-NEXT:   %40 = add nsw i64 %"iv1'ac.0", -1
; CHECK-NEXT:   br label %invertfor.body44

; CHECK: invertif.then:                                    ; preds = %if.end
; CHECK-NEXT:   %mul62_unwrap = mul i64 %conv, %n
; CHECK-NEXT:   %add63_unwrap = add i64 %mul62_unwrap, %conv6
; CHECK-NEXT:   %"arrayidx64'ipg_unwrap" = getelementptr inbounds float, float* %"d_result'", i64 %add63_unwrap
; CHECK-NEXT:   %41 = load float, float* %"arrayidx64'ipg_unwrap", align 4
; CHECK-NEXT:   store float 0.000000e+00, float* %"arrayidx64'ipg_unwrap", align 4
; CHECK-NEXT:   br label %invertfor.cond.cleanup
; CHECK-NEXT: }