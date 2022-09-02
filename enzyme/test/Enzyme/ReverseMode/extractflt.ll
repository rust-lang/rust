; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -instsimplify -adce -simplifycfg -S | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

declare void @_Z17__enzyme_autodiffPv(...)

define double @f({ double }* %r2) {
entry:
  %r4 = load { double }, { double }* %r2, align 8
  %ex = extractvalue { double } %r4, 0
  %g = fadd double %ex, %ex
  ret double %g
}

define void @caller(i8* %a, i8* %b) {
  call void (...) @_Z17__enzyme_autodiffPv(double ({ double }*)* @f, metadata !"enzyme_dup", i8* %a, i8* %b)
  ret void
}

; CHECK: define internal void @diffef({ double }* %r2, { double }* %"r2'", double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = fadd fast double %differeturn, %differeturn
; CHECK-NEXT:   %1 = bitcast { double }* %"r2'" to double*
; CHECK-NEXT:   %2 = load double, double* %1, align 8
; CHECK-NEXT:   %3 = fadd fast double %2, %0
; CHECK-NEXT:   store double %3, double* %1, align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
