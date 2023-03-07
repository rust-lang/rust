; RUN: %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=matvec -o /dev/null | FileCheck %s

define internal void @matvec(i64* %a, i64* %b) {
entry:
  %gloaded = load i64, i64* %a, align 4
  %iloaded = load i64, i64* %b, align 4
  %aint = ptrtoint i64* %a to i64
  %bint = ptrtoint i64* %a to i64
  %sub = sub i64 %aint, %bint
  %subptr = inttoptr i64 %sub to i64*
  br i1 true, label %good, label %bad

bad:
  %badload = load i64, i64* %subptr, align 4
  unreachable

good:
  ret void
}


; CHECK: matvec - {} |{[-1]:Pointer}:{} {[-1]:Pointer}:{} 
; CHECK-NEXT: i64* %a: {[-1]:Pointer}
; CHECK-NEXT: i64* %b: {[-1]:Pointer}
; CHECK-NEXT: entry
; CHECK-NEXT:   %gloaded = load i64, i64* %a, align 4: {}
; CHECK-NEXT:   %iloaded = load i64, i64* %b, align 4: {}
; CHECK-NEXT:   %aint = ptrtoint i64* %a to i64: {[-1]:Pointer}
; CHECK-NEXT:   %bint = ptrtoint i64* %a to i64: {[-1]:Pointer}
; CHECK-NEXT:   %sub = sub i64 %aint, %bint: {[-1]:Integer}
; CHECK-NEXT:   %subptr = inttoptr i64 %sub to i64*: {[-1]:Integer}
; CHECK-NEXT:   br i1 true, label %good, label %bad: {}
; CHECK-NEXT: bad
; CHECK-NEXT:   %badload = load i64, i64* %subptr, align 4: {}
; CHECK-NEXT:   unreachable: {}
; CHECK-NEXT: good
; CHECK-NEXT:   ret void: {}
