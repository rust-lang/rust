; RUN: %opt < %s %loadEnzyme -print-activity-analysis -activity-analysis-func=matvec -o /dev/null | FileCheck %s

define internal double @matvec(double* noalias %b) {
entry:
  %.fca.1.insert.i = insertvalue { double* } undef, double* %b, 0
  %i7 = extractvalue { double* } %.fca.1.insert.i, 0
  %unusedWeird = bitcast double* %i7 to double*
  %.fca.1.insert.i15 = insertvalue { double* } undef, double* %i7, 0
  %i23 = extractvalue { double* } %.fca.1.insert.i15, 0
  %i29 = load double, double* %i23, align 8
  %arrayidx.i.i.i = getelementptr inbounds double, double* %i23, i64 1
  ret double %i29
}

; CHECK: double* %b: icv:0
; CHECK-NEXT: entry
; CHECK-NEXT:   %.fca.1.insert.i = insertvalue { double* } undef, double* %b, 0: icv:0 ici:1
; CHECK-NEXT:   %i7 = extractvalue { double* } %.fca.1.insert.i, 0: icv:0 ici:1
; CHECK-NEXT:   %unusedWeird = bitcast double* %i7 to double*: icv:0 ici:1
; CHECK-NEXT:   %.fca.1.insert.i15 = insertvalue { double* } undef, double* %i7, 0: icv:0 ici:1
; CHECK-NEXT:   %i23 = extractvalue { double* } %.fca.1.insert.i15, 0: icv:0 ici:1
; CHECK-NEXT:   %i29 = load double, double* %i23, align 8: icv:0 ici:0
; CHECK-NEXT:   %arrayidx.i.i.i = getelementptr inbounds double, double* %i23, i64 1: icv:0 ici:1
; CHECK-NEXT:   ret double %i29: icv:1 ici:1