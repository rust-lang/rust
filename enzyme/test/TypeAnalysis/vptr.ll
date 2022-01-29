; RUN: %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=caller -o /dev/null | FileCheck %s

declare void @f(i64 %x)

define void @caller() {
entry:
  %.fca.0.insert = insertvalue { i64, i64 } undef, i64 ptrtoint (void (i64)* @f to i64), 0
  %.fca.1.insert = insertvalue { i64, i64 } %.fca.0.insert, i64 0, 1
  %memptr.adj.i = extractvalue { i64, i64 } %.fca.1.insert, 1
  %memptr.ptr.i = extractvalue { i64, i64 } %.fca.1.insert, 0
  %and = and i64 %memptr.ptr.i, 1
  ret void
}

; CHECK: caller - {} |
; CHECK-NEXT: entry
; CHECK-NEXT:   %.fca.0.insert = insertvalue { i64, i64 } undef, i64 ptrtoint (void (i64)* @f to i64), 0: {[0]:Pointer, [8]:Anything, [9]:Anything, [10]:Anything, [11]:Anything, [12]:Anything, [13]:Anything, [14]:Anything, [15]:Anything}
; CHECK-NEXT:   %.fca.1.insert = insertvalue { i64, i64 } %.fca.0.insert, i64 0, 1: {[0]:Pointer, [8]:Anything, [9]:Anything, [10]:Anything, [11]:Anything, [12]:Anything, [13]:Anything, [14]:Anything, [15]:Anything}
; CHECK-NEXT:   %memptr.adj.i = extractvalue { i64, i64 } %.fca.1.insert, 1: {[-1]:Anything}
; CHECK-NEXT:   %memptr.ptr.i = extractvalue { i64, i64 } %.fca.1.insert, 0: {[-1]:Pointer}
; CHECK-NEXT:   %and = and i64 %memptr.ptr.i, 1: {[-1]:Integer}
; CHECK-NEXT:   ret void: {}
