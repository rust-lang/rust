; RUN: %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=caller -o /dev/null | FileCheck %s

declare i64 @f()

define void @caller(double* %p) {
entry:
  %int = ptrtoint double* %p to i64
  %ld = load double, double* %p, align 8
  %plus = sub i64 %int, 1
  %foo = call i64 @f()
  %nil = sub i64 %int, %foo
  ret void
}


; CHECK: caller - {} |{[-1]:Pointer, [-1,-1]:Float@double}:{} 
; CHECK-NEXT: double* %p: {[-1]:Pointer, [-1,-1]:Float@double}
; CHECK-NEXT: entry
; CHECK-NEXT:   %int = ptrtoint double* %p to i64: {[-1]:Pointer, [-1,-1]:Float@double}
; CHECK-NEXT:   %ld = load double, double* %p, align 8: {[-1]:Float@double}
; CHECK-NEXT:   %plus = sub i64 %int, 1: {[-1]:Pointer, [-1,-1]:Float@double}
; CHECK-NEXT:   %foo = call i64 @f(): {}
; the below should be unknown since ptr - unknown => unknown
; CHECK-NEXT:   %nil = sub i64 %int, %foo: {}
; CHECK-NEXT:   ret void: {}
