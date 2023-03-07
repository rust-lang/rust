; RUN: %opt < %s %loadEnzyme -print-activity-analysis -activity-analysis-func=f -o /dev/null | FileCheck %s

define double @f({ i64, double }* %x) {
entry:
  %v = load { i64, double }, { i64, double }* %x, align 8
  %r = extractvalue { i64, double } %v, 1
  %q = fmul double %r, %r
  ret double %q
}

; CHECK: { i64, double }* %x: icv:0
; CHECK-NEXT: entry
; CHECK-NEXT:   %v = load { i64, double }, { i64, double }* %x, align 8: icv:0 ici:0
; CHECK-NEXT:   %r = extractvalue { i64, double } %v, 1: icv:0 ici:0
; CHECK-NEXT:   %q = fmul double %r, %r: icv:0 ici:0
; CHECK-NEXT:   ret double %q: icv:1 ici:1
