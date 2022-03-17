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
; CHECK-NEXT:   call void @__enzyme_memcpyadd_doubleda8sa8(double* %0, double* %1, i64 3)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
