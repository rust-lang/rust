; RUN: %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=matvec -o /dev/null | FileCheck %s

define internal void @matvec(i64* %lhs, double* %res) {
entry:
  %loaded = load i64, i64* %lhs, align 4
  %a2 = inttoptr i64 %loaded to double*
  %div = lshr i64 %loaded, 3
  %and = and i64 %div, 1
  %gep = getelementptr inbounds double, double* %a2, i64 %and
  %a4 = load double, double* %gep, align 8
  store double %a4, double* %res, align 8
  ret void
}


; CHECK: matvec - {} |{[-1]:Pointer}:{} {[-1]:Pointer, [-1,-1]:Float@double}:{} 
; CHECK-NEXT: i64* %lhs: {[-1]:Pointer, [-1,0]:Pointer}
; CHECK-NEXT: double* %res: {[-1]:Pointer, [-1,-1]:Float@double}
; CHECK-NEXT: entry
; CHECK-NEXT:   %loaded = load i64, i64* %lhs, align 4: {[-1]:Pointer}
; CHECK-NEXT:   %a2 = inttoptr i64 %loaded to double*: {[-1]:Pointer}
; CHECK-NEXT:   %div = lshr i64 %loaded, 3: {}
; CHECK-NEXT:   %and = and i64 %div, 1: {[-1]:Integer}
; CHECK-NEXT:   %gep = getelementptr inbounds double, double* %a2, i64 %and: {[-1]:Pointer, [-1,0]:Float@double}
; CHECK-NEXT:   %a4 = load double, double* %gep, align 8: {[-1]:Float@double}
; CHECK-NEXT:   store double %a4, double* %res, align 8: {}
; CHECK-NEXT:   ret void: {}
