; RUN: %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=square -o /dev/null | FileCheck %s

define internal void @square(double* %in) {
entry:
  %out = getelementptr inbounds double, double* %in
  ret void
}

; CHECK: square - {} |{[-1]:Pointer, [-1,-1]:Float@double}:{} 
; CHECK-NEXT: double* %in: {[-1]:Pointer, [-1,-1]:Float@double}
; CHECK-NEXT: entry
; CHECK-NEXT:   %out = getelementptr inbounds double, double* %in: {[-1]:Pointer, [-1,-1]:Float@double}
; CHECK-NEXT:   ret void: {}
