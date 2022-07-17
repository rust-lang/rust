; RUN: %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=caller -o /dev/null | FileCheck %s

; ModuleID = 'llvm-link'

define void @caller() {
entry:
  %g = getelementptr float, float* null, i32 1
  %pi = ptrtoint float* %g to i64
  %tmp11 = call i8* @malloc(i64 %pi)
  ret void
}

declare i8* @malloc(i64)

; CHECK: caller - {} |
; CHECK-NEXT: entry
; CHECK-NEXT:   %g = getelementptr float, float* null, i32 1: {[-1]:Integer}
; CHECK-NEXT:   %pi = ptrtoint float* %g to i64: {[-1]:Integer}
; CHECK-NEXT:   %tmp11 = call i8* @malloc(i64 %pi): {[-1]:Pointer}
; CHECK-NEXT:   ret void: {}

