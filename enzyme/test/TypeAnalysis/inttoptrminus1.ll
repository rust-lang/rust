; RUN: %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=matvec -o /dev/null | FileCheck %s

define internal void @matvec() {
entry:
  %gloaded = load i64, i64* inttoptr (i64 1 to i64*), align 4
  %cst = inttoptr i64 1 to i64*
  %iloaded = load i64, i64* inttoptr (i64 1 to i64*), align 4
  br label %next
next:
  %phi = phi i64* [ inttoptr (i64 1 to i64*), %entry ]
  %ploaded = load i64, i64* %phi, align 4
  %ext = zext i1 1 to i64
  %extptr = inttoptr i64 %ext to i64*
  %zloaded = load i64, i64* %extptr, align 4
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
