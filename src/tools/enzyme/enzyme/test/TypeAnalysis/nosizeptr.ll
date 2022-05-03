; RUN: %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=caller -o /dev/null | FileCheck %s


define void @caller(double* %p) {
entry:
  %z = bitcast double* %p to [0 x double]*
  ret void
}

; CHECK: caller - {} |{[-1]:Pointer, [-1,-1]:Float@double}:{} 
; CHECK-NEXT: double* %p: {[-1]:Pointer, [-1,-1]:Float@double}
; CHECK-NEXT: entry
; CHECK-NEXT:   %z = bitcast double* %p to [0 x double]*: {[-1]:Pointer, [-1,-1]:Float@double}
; CHECK-NEXT:   ret void: {}
