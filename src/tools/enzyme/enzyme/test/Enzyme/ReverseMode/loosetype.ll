; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -mem2reg -instsimplify -simplifycfg -S -enzyme-loose-types | FileCheck %s


%struct.tensor2 = type { %struct.tensor1 }
%struct.tensor1 = type { [3 x double] }

define void @_Z9transposePK7tensor2([3 x double]*  %A, [3 x double]* %ref.tmp) {
entry:
  %a0 = bitcast [3 x double]* %ref.tmp to i8*
  %a1 = bitcast [3 x double]* %A to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %a0, i8* align 8 %a1, i64 24, i1 false)
  ret void
}
define dso_local void @_Z4callPK7tensor2S1_(%struct.tensor2* %A, %struct.tensor2* %dA) {
entry:
  call void (i8*, ...) @_Z17__enzyme_autodiffPvz(i8* bitcast (void ([3 x double]*, [3 x double]*)* @_Z9transposePK7tensor2 to i8*), %struct.tensor2* %A, %struct.tensor2* %dA, %struct.tensor2* %A, %struct.tensor2* %dA)
  ret void
}

declare void @_Z17__enzyme_autodiffPvz(i8*, ...)

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture, i8* noalias nocapture readonly, i64, i1)

; CHECK: define internal void @diffe_Z9transposePK7tensor2([3 x double]* %A, [3 x double]* %"A'", [3 x double]* %ref.tmp, [3 x double]* %"ref.tmp'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"a0'ipc" = bitcast [3 x double]* %"ref.tmp'" to i8*
; CHECK-NEXT:   %a0 = bitcast [3 x double]* %ref.tmp to i8*
; CHECK-NEXT:   %"a1'ipc" = bitcast [3 x double]* %"A'" to i8*
; CHECK-NEXT:   %a1 = bitcast [3 x double]* %A to i8*
; CHECK-NEXT:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %a0, i8* align 8 %a1, i64 24, i1 false)
; CHECK-NEXT:   %0 = bitcast i8* %"a0'ipc" to double*
; CHECK-NEXT:   %1 = bitcast i8* %"a1'ipc" to double*
; CHECK-NEXT:   br label %for.body.i

; CHECK: for.body.i:                                       ; preds = %for.body.i, %entry
; CHECK-NEXT:   %idx.i = phi i64 [ 0, %entry ], [ %idx.next.i, %for.body.i ]
; CHECK-NEXT:   %dst.i.i = getelementptr inbounds double, double* %0, i64 %idx.i
; CHECK-NEXT:   %dst.i.l.i = load double, double* %dst.i.i
; CHECK-NEXT:   store double 0.000000e+00, double* %dst.i.i
; CHECK-NEXT:   %src.i.i = getelementptr inbounds double, double* %1, i64 %idx.i
; CHECK-NEXT:   %src.i.l.i = load double, double* %src.i.i
; CHECK-NEXT:   %2 = fadd fast double %src.i.l.i, %dst.i.l.i
; CHECK-NEXT:   store double %2, double* %src.i.i
; CHECK-NEXT:   %idx.next.i = add nuw i64 %idx.i, 1
; CHECK-NEXT:   %3 = icmp eq i64 3, %idx.next.i
; CHECK-NEXT:   br i1 %3, label %__enzyme_memcpyadd_doubleda8sa8.exit, label %for.body.i

; CHECK: __enzyme_memcpyadd_doubleda8sa8.exit:             ; preds = %for.body.i
; CHECK-NEXT:   ret void
; CHECK-NEXT: }