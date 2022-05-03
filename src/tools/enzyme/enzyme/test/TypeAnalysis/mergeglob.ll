; RUN: %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=callee -o /dev/null | FileCheck %s

@ptr = internal constant [4 x double] [double 3.14000e+00, double 3.14000e+00, double 3.14000e+00, double 3.14000e+00], align 8

define void @callee() {
entry:
  %self = getelementptr inbounds [4 x double], [4 x double]* @ptr, i32 0, i32 0
  ret void
}

; CHECK: callee - {} |
; CHECK-NEXT: entry
; CHECK-NEXT:   %self = getelementptr inbounds [4 x double], [4 x double]* @ptr, i32 0, i32 0: {[-1]:Pointer, [-1,-1]:Float@double}
; CHECK-NEXT:   ret void: {}
