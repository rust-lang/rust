; RUN: %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=caller -o /dev/null | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @caller() {
entry:
  %sel = select i1 true, i64 0, i64 8
  %ptr = inttoptr i64 %sel to i64*
  %ld = load i64, i64* %ptr
  ret void
}

; CHECK: caller - {} |
; CHECK-NEXT: entry
; CHECK-NEXT:   %sel = select i1 true, i64 0, i64 8: {[-1]:Anything}
; CHECK-NEXT:   %ptr = inttoptr i64 %sel to i64*: {[-1]:Anything}
; CHECK-NEXT:   %ld = load i64, i64* %ptr: {}
; CHECK-NEXT:   ret void: {}