; RUN: %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=caller -o /dev/null | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @caller(double* %inp) {
entry:
  %ptr = bitcast double* %inp to i8*
  %res = call i8* @realloc(i8* %ptr, i64 16)
  ret void
}

; CHECK: caller - {} |{[-1]:Pointer, [-1,-1]:Float@double}:{} 
; CHECK-NEXT: double* %inp: {[-1]:Pointer, [-1,-1]:Float@double}
; CHECK-NEXT: entry
; CHECK-NEXT:   %ptr = bitcast double* %inp to i8*: {[-1]:Pointer, [-1,-1]:Float@double}
; CHECK-NEXT:   %res = call i8* @realloc(i8* %ptr, i64 16): {[-1]:Pointer, [-1,0]:Float@double, [-1,8]:Float@double}
; CHECK-NEXT:   ret void: {}

declare dso_local noalias i8* @realloc(i8* nocapture, i64)
