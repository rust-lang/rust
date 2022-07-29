; RUN: %opt < %s %loadEnzyme -print-activity-analysis -activity-analysis-func=f -enzyme-strict-aliasing=0 -o /dev/null | FileCheck %s

declare i64 @getint(double* %x)

define void @f(double* %x) {
entry:
  %a189 = call i64 @getint(double* %x)
  %a190 = ashr exact i64 %a189, 4
  br i1 true, label %r, label %e

r:
  %g = getelementptr inbounds double, double* %x, i64 %a190
  store double 1.000000e+00, double* %g, align 8
  br label %e

e:
  ret void
}

; CHECK: double* %x: icv:0
; CHECK-NEXT: entry
; CHECK-NEXT:   %a189 = call i64 @getint(double* %x): icv:0 ici:0
; CHECK-NEXT:   %a190 = ashr exact i64 %a189, 4: icv:0 ici:1
; CHECK-NEXT:   br i1 true, label %r, label %e: icv:1 ici:1
; CHECK-NEXT: r
; CHECK-NEXT:   %g = getelementptr inbounds double, double* %x, i64 %a190: icv:0 ici:1
; CHECK-NEXT:   store double 1.000000e+00, double* %g, align 8: icv:1 ici:1
; CHECK-NEXT:   br label %e: icv:1 ici:1
; CHECK-NEXT: e
; CHECK-NEXT:   ret void: icv:1 ici:1

