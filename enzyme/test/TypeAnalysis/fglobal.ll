; RUN: %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=callee -o /dev/null | FileCheck %s

@ptr = internal constant [16 x i8] c"1234567890123456", align 1

define void @callee() {
entry:
  %ld = load i64, i64* bitcast ([16 x i8]* @ptr to i64*), align 4
  ret void
}

; CHECK: callee - {} |
; CHECK-NEXT: entry
; CHECK-NEXT:   %ld = load i64, i64* bitcast ([16 x i8]* @ptr to i64*), align 4: {[-1]:Anything}
; CHECK-NEXT:   ret void: {}
