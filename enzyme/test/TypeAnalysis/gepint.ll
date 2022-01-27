; RUN: %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=callee -o /dev/null | FileCheck %s

declare i64 @val()

define void @callee(i64* %ptr) {
entry:
  %load1 = load i64, i64* %ptr
  %v = call i64 @val()
  %ptr2 = getelementptr i64, i64* %ptr, i64 %v
  %load2 = load i64, i64* %ptr2
  ret void
}

; CHECK: callee - {} |{}:{}
; CHECK-NEXT: i64* %ptr: {[-1]:Pointer}
; CHECK-NEXT: entry
; CHECK-NEXT:   %load1 = load i64, i64* %ptr: {}
; CHECK-NEXT:   %v = call i64 @val(): {[-1]:Integer}
; CHECK-NEXT:   %ptr2 = getelementptr i64, i64* %ptr, i64 %v: {[-1]:Pointer}
; CHECK-NEXT:   %load2 = load i64, i64* %ptr2: {}
; CHECK-NEXT:   ret void: {}
