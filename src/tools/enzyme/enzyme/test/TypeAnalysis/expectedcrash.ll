; RUN: %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=matvec -o /dev/null | FileCheck %s
; This is expected to crash as it learns that a value is both a pointer and an integer which is illegal
; XFAIL: *
define internal void @matvec(i64* %a, i64* %b) {
entry:
  %gloaded = load i64, i64* %a, align 4
  %iloaded = load i64, i64* %b, align 4
  %aint = ptrtoint i64* %a to i64
  %bint = ptrtoint i64* %a to i64
  %sub = sub i64 %aint, %bint
  %subptr = inttoptr i64 %sub to i64*
  %badload = load i64, i64* %subptr
  ret void
}


; CHECK: matvec - {} |
; CHECK-NEXT: entry
; CHECK-NEXT:   %gloaded = load i64, i64* inttoptr (i64 1 to i64*), align 4: {}
; CHECK-NEXT:   %cst = inttoptr i64 1 to i64*: {[-1]:Anything}
; CHECK-NEXT:   %iloaded = load i64, i64* inttoptr (i64 1 to i64*), align 4: {}
; CHECK-NEXT:   br label %next: {}
; CHECK-NEXT: next
; CHECK-NEXT:   %phi = phi i64* [ inttoptr (i64 1 to i64*), %entry ]: {[-1]:Anything}
; CHECK-NEXT:   %ploaded = load i64, i64* %phi, align 4: {}
; CHECK-NEXT:   %ext = zext i1 true to i64: {[-1]:Anything}
; CHECK-NEXT:   %extptr = inttoptr i64 %ext to i64*: {[-1]:Anything}
; CHECK-NEXT:   %zloaded = load i64, i64* %extptr, align 4: {}
; CHECK-NEXT:   ret void: {}
