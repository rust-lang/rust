; RUN: %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=d -o /dev/null | FileCheck %s

define i8 @d(double %x) {
entry:
  %bc = bitcast double %x to i64
  %t = trunc i64 %bc to i8
  ret i8 %t
}

; CHECK: d - {[-1]:Integer} |{[-1]:Float@double}:{}
; CHECK-NEXT: double %x: {[-1]:Float@double}
; CHECK-NEXT: entry
; CHECK-NEXT:   %bc = bitcast double %x to i64: {[-1]:Float@double}
; CHECK-NEXT:   %t = trunc i64 %bc to i8: {[-1]:Integer}
; CHECK-NEXT:   ret i8 %t: {}
