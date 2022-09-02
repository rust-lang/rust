; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -instsimplify -adce -simplifycfg -S | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

declare void @_Z17__enzyme_autodiffPv(...)

define double @f({ double, float, i32 }* %r2) {
entry:
  %r4 = load { double, float, i32 }, { double, float, i32 }* %r2, align 8
  %ex = extractvalue { double, float, i32 } %r4, 0
  %ex2 = extractvalue { double, float, i32 } %r4, 1
  %ex3 = extractvalue { double, float, i32 } %r4, 2
  %si = sitofp i32 %ex3 to float
  %fp = fpext float %ex2 to double
  %g = fadd double %ex, %fp
  ret double %g
}

define void @caller(i8* %a, i8* %b) {
  call void (...) @_Z17__enzyme_autodiffPv(double ({ double, float, i32 }*)* @f, metadata !"enzyme_dup", i8* %a, i8* %b)
  ret void
}

; CHECK: define internal void @diffef({ double, float, i32 }* %r2, { double, float, i32 }* %"r2'", double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = fptrunc double %differeturn to float
; CHECK-NEXT:   %1 = bitcast { double, float, i32 }* %"r2'" to double*
; CHECK-NEXT:   %2 = load double, double* %1, align 8
; CHECK-NEXT:   %3 = fadd fast double %2, %differeturn
; CHECK-NEXT:   store double %3, double* %1, align 8
; CHECK-NEXT:   %4 = bitcast { double, float, i32 }* %"r2'" to i8*
; CHECK-NEXT:   %5 = getelementptr inbounds i8, i8* %4, i64 8
; CHECK-NEXT:   %6 = bitcast i8* %5 to float*
; CHECK-NEXT:   %7 = load float, float* %6, align 8
; CHECK-NEXT:   %8 = fadd fast float %7, %0
; CHECK-NEXT:   store float %8, float* %6, align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
