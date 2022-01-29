; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -simplifycfg -adce -S | FileCheck %s

declare void @llvm.trap()

declare void @baduse(i64, double*)

; Function Attrs: nounwind readnone uwtable
define double @tester({double, i64}* %x, i1 %cmp) {
entry:
  %gep0 = getelementptr inbounds {double, i64}, {double, i64}* %x, i64 0, i32 0
  %gep1 = getelementptr inbounds {double, i64}, {double, i64}* %x, i64 0, i32 1
  %ld = load i64, i64* %gep1
  br i1 %cmp, label %exit, label %err

err:
  call void @baduse(i64 %ld, double* %gep0)
  call void @llvm.trap()
  unreachable

exit:
  %res = load double, double* %gep0
  ret double %res
}

define double @test_derivative({double, i64}* %x, {double, i64}* %dx, i1 %cmp) {
entry:
  %0 = tail call double (double ({double, i64}*, i1)*, ...) @__enzyme_autodiff(double ({double, i64}*, i1)* nonnull @tester, {double, i64}* %x, {double, i64}* %dx, i1 %cmp)
  ret double %0
}

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double ({double, i64}*, i1)*, ...)

; CHECK: define internal void @diffetester({ double, i64 }* %x, { double, i64 }* %"x'", i1 %cmp, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"gep0'ipg" = getelementptr inbounds { double, i64 }, { double, i64 }* %"x'", i64 0, i32 0
; CHECK-NEXT:   %gep0 = getelementptr inbounds { double, i64 }, { double, i64 }* %x, i64 0, i32 0
; CHECK-NEXT:   %gep1 = getelementptr inbounds { double, i64 }, { double, i64 }* %x, i64 0, i32 1
; CHECK-NEXT:   %ld = load i64, i64* %gep1
; CHECK-NEXT:   br i1 %cmp, label %invertexit, label %err

; CHECK: err:                                              ; preds = %entry
; CHECK-NEXT:   call void @baduse(i64 %ld, double* %gep0)
; CHECK:   unreachable

; CHECK: invertexit:                                       ; preds = %entry
; CHECK-NEXT:   %0 = load double, double* %"gep0'ipg"
; CHECK-NEXT:   %1 = fadd fast double %0, %differeturn
; CHECK-NEXT:   store double %1, double* %"gep0'ipg"
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
