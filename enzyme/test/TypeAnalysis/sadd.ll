; RUN: %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=caller -o /dev/null | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare { i64, i1 } @llvm.sadd.with.overflow.i64(i64, i64)

define void @caller() {
entry:
  %sadded = call { i64, i1 } @llvm.sadd.with.overflow.i64(i64 3, i64 1)
  %res = extractvalue { i64, i1 } %sadded, 0
  ret void
}

; CHECK: caller - {} |
; CHECK-NEXT: entry
; CHECK-NEXT:   %sadded = call { i64, i1 } @llvm.sadd.with.overflow.i64(i64 3, i64 1): {[0]:Integer, [8]:Integer}
; CHECK-NEXT:   %res = extractvalue { i64, i1 } %sadded, 0: {[0]:Integer}
; CHECK-NEXT:   ret void: {}